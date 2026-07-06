//! M2 prepass: decode a wasm function's `indirect-dispatch` op stream into a
//! flat `i64`-word MiniProgram for the majit-traced mainloop (see [`super`]).
//!
//! Under `indirect-dispatch` the op stream is tightly packed as
//! `[u16 OpCode][operand bytes…]` with no inter-op padding, so it is walked
//! linearly: read the `OpCode` (a `u16` discriminant), then decode that
//! opcode's operand struct (`crate::ir::decode::*`).
//!
//! wasmi is not a pure slot machine: many ops read/write an implicit per-type
//! accumulator register (`Reg<i64>`, encoded as zero stream bytes). The
//! MiniProgram models the integer accumulator as a single scalar `ireg`; the
//! frame slots become an indexed `i64` cell array. A `SlotAndReg` result writes
//! both a slot and `ireg`.
//!
//! The prepass is intentionally partial: it maps only the small op subset the
//! PoC kernel implements and returns `None` (function ineligible — fall back to
//! the stock executor) at the first op outside that subset.

#![allow(dead_code)]

use crate::ir::OpCode;
use alloc::vec::Vec;

/// The MiniProgram `env`: a flat `i64`-word instruction stream indexed by the
/// kernel's program counter. Element kind must be a plain indexable integer
/// slice for the `#[jit_interp]` macro (operand reads constant-fold on a green
/// `pc`).
pub(crate) type MiniCode = [i64];

/// MiniProgram opcodes. Each instruction is `[opcode, operands…]` of `i64`
/// words. Cell operands are slot indices into the kernel's `slots` cell array.
/// wasmi keeps type-separated accumulator registers; the kernel models the two
/// it lowers as a two-cell `accum` array: `accum[0]` is the integer accumulator
/// (`Reg<i64>`, `ireg`) and `accum[1]` is the f64 accumulator (`Reg<f64>`,
/// `freg64`, held as raw bits). Integer ops read/write `accum[0]`, f64 ops
/// read/write `accum[1]`, and cross-type ops (int↔f64 converts, f64 compares,
/// f64→int truncations) read one and write the other. Keeping them apart lets an
/// integer value in `ireg` survive across f64 ops (and vice versa) exactly as it
/// does in wasmi — conflating them corrupts a live cross-type value.
pub(crate) const MINI_HALT: i64 = 0;
/// `[MINI_I32_ADD_SI_WB, dst_slot, lhs_slot, imm]` (4 words):
/// `slots[dst] = ireg = wrap_i32(slots[lhs] + imm)`. Lowers wasmi
/// `I32Add_Rs_si`, the `local.tee $dst (i32.add/sub (local.get $lhs) imm)`
/// fusion (a `… - c` is translated as `… + (-c)`).
pub(crate) const MINI_I32_ADD_SI_WB: i64 = 1;
/// `[MINI_BR_I32_NE_RI, target_word, imm]` (3 words): if
/// `wrap_i32(ireg) != imm` jump to `target_word`, else fall through. Lowers
/// wasmi `BranchI32NotEq_Ri`. A jump to a lower word index is a loop back-edge.
pub(crate) const MINI_BR_I32_NE_RI: i64 = 2;
/// `[MINI_RETURN_R]` (1 word): `return ireg`. Lowers wasmi `ReturnU64_R`.
pub(crate) const MINI_RETURN_R: i64 = 3;
/// `[MINI_COPY_SI, dst_slot, imm]` (3 words): `slots[dst] = imm`. Lowers wasmi
/// `U64Copy_Si` (a materialized `i64.const` / 64-bit `local.set`/`.tee`).
pub(crate) const MINI_COPY_SI: i64 = 4;
/// `[MINI_COPY_SS, dst_slot, src_slot]` (3 words): `slots[dst] = slots[src]`.
/// Lowers wasmi `U64Copy_Ss` (a slot-to-slot `local.set`/`local.get`).
pub(crate) const MINI_COPY_SS: i64 = 5;
/// `[MINI_COPY_SR, dst_slot]` (2 words): `slots[dst] = ireg`. Lowers wasmi's
/// local-indexed `U64Copy_S{0..9}r` (accumulator spill to a local).
pub(crate) const MINI_COPY_SR: i64 = 6;
/// `[MINI_I64_ADD_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = slots[lhs] + slots[rhs]` (i64). Lowers wasmi `I64Add_Rss`.
pub(crate) const MINI_I64_ADD_SS_WR: i64 = 7;
// Value 8 is retired: `I64Add_Rs_si` now pre-materializes its immediate into a
// scratch slot and reuses `MINI_I64_ADD_SS_WB`, so no dedicated slot+imm op is
// needed (the first use of operand pre-materialization, see `NUM_SCRATCH`).
/// `[MINI_BR_I64_EQ_SI, target_word, lhs_slot, imm]` (4 words): if
/// `slots[lhs] == imm` jump to `target_word` (i64). Lowers wasmi
/// `BranchI64Eq_Si` (e.g. an `i64.eqz` branch).
pub(crate) const MINI_BR_I64_EQ_SI: i64 = 9;
/// `[MINI_BR_I64_NE_RI, target_word, imm]` (3 words): if `ireg != imm` jump to
/// `target_word` (i64). Lowers wasmi `BranchI64NotEq_Ri`.
pub(crate) const MINI_BR_I64_NE_RI: i64 = 10;
/// `[MINI_RETURN_S, src_slot]` (2 words): `return slots[src]`. Lowers wasmi
/// `ReturnU64_S`.
pub(crate) const MINI_RETURN_S: i64 = 11;
/// `[MINI_I64_MUL_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = slots[lhs] * slots[rhs]` (i64, wrapping). Lowers wasmi `I64Mul_Rss`.
pub(crate) const MINI_I64_MUL_SS_WR: i64 = 12;
/// `[MINI_I32_MUL_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = wrap_i32(slots[lhs] * slots[rhs])`. Lowers wasmi `I32Mul_Rss`.
pub(crate) const MINI_I32_MUL_SS_WR: i64 = 13;
/// `[MINI_I32_ADD_RS_WB, dst_slot, rhs_slot]` (3 words):
/// `slots[dst] = ireg = wrap_i32(ireg + slots[rhs])`. Lowers wasmi
/// `I32Add_Rs_rs` (the `local.tee $dst (i32.add <acc> (local.get $rhs)))` fusion,
/// `lhs` = the implicit accumulator register).
pub(crate) const MINI_I32_ADD_RS_WB: i64 = 14;
/// `[MINI_BR_U32_LE_SS, target_word, lhs_slot, rhs_slot]` (4 words): if
/// `lo32(slots[lhs]) <= lo32(slots[rhs])` (unsigned) jump to `target_word`.
/// Lowers wasmi `BranchU32Le_Ss`.
pub(crate) const MINI_BR_U32_LE_SS: i64 = 15;
/// `[MINI_BR_ALWAYS, target_word]` (2 words): unconditional jump to
/// `target_word`. Lowers wasmi `Branch`; a jump to a lower word is a back-edge.
pub(crate) const MINI_BR_ALWAYS: i64 = 16;
/// `[MINI_BR_I64_LE_SS, target_word, lhs_slot, rhs_slot]` (4 words): if
/// `slots[lhs] <= slots[rhs]` (signed i64) jump to `target_word`. Lowers wasmi
/// `BranchI64Le_Ss`.
pub(crate) const MINI_BR_I64_LE_SS: i64 = 17;
/// `[MINI_I64_XOR_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = slots[lhs] ^ slots[rhs]` (i64). Lowers wasmi `I64BitXor_Rss`.
pub(crate) const MINI_I64_XOR_SS_WR: i64 = 18;
/// `[MINI_I64_AND_RI_WR, imm]` (2 words): `ireg = ireg & imm` (i64). Lowers
/// wasmi `I64BitAnd_Rri` (the accumulator-and-immediate fusion).
pub(crate) const MINI_I64_AND_RI_WR: i64 = 19;
/// `[MINI_I64_AND_SI_WR, lhs_slot, imm]` (3 words): `ireg = slots[lhs] & imm`
/// (i64). Lowers wasmi `I64BitAnd_Rsi`.
pub(crate) const MINI_I64_AND_SI_WR: i64 = 20;
/// `[MINI_I64_ADD_RS_WB, dst_slot, rhs_slot]` (3 words):
/// `slots[dst] = ireg = ireg + slots[rhs]` (i64). Lowers wasmi `I64Add_Rs_rs`
/// (`lhs` = the implicit accumulator register).
pub(crate) const MINI_I64_ADD_RS_WB: i64 = 21;
/// `[MINI_U64_SHR_SI, src_slot, shift, mask]` (4 words): logical shift-right by
/// a constant — `ireg = (slots[src] >> shift) & mask`, where `mask` clears the
/// `shift` high bits the arithmetic `>>` sign-extends. Lowers wasmi
/// `U64Shr_Rsi` (`i64.shr_u` by an immediate). majit has only arithmetic
/// `IntRshift`, so the mask reproduces the logical (zero-fill) shift.
pub(crate) const MINI_U64_SHR_SI: i64 = 22;
/// `[MINI_I64_ADD_SS_WB, dst_slot, lhs_slot, rhs_slot]` (4 words):
/// `slots[dst] = ireg = slots[lhs] + slots[rhs]` (i64). Lowers wasmi
/// `I64Add_Rs_ss` (the slot-and-reg result variant of `MINI_I64_ADD_SS_WR`).
pub(crate) const MINI_I64_ADD_SS_WB: i64 = 23;
/// `[MINI_BR_I64_LE_SI, target_word, lhs_slot, imm]` (4 words): if
/// `slots[lhs] <= imm` (signed i64) jump to `target_word`. Lowers wasmi
/// `BranchI64Le_Si`.
pub(crate) const MINI_BR_I64_LE_SI: i64 = 24;
/// `[MINI_I32_ADD_SS_WB, dst_slot, lhs_slot, rhs_slot]` (4 words):
/// `slots[dst] = ireg = i32-wrap(slots[lhs] + slots[rhs])`. Lowers wasmi
/// `I32Add_Rs_ss`.
pub(crate) const MINI_I32_ADD_SS_WB: i64 = 25;
/// `[MINI_BR_I32_LT_SI, target_word, lhs_slot, imm]` (4 words): if
/// `(i32)slots[lhs] < imm` (signed i32, `imm` pre-sign-extended) jump to
/// `target_word`. Lowers wasmi `BranchI32Lt_Si`.
pub(crate) const MINI_BR_I32_LT_SI: i64 = 26;
/// `[MINI_BR_I64_LT_IR, target_word, imm]` (3 words): if `imm < ireg` (signed
/// i64) jump to `target_word` — the accumulator is the right operand. Lowers
/// wasmi `BranchI64Lt_Ir` (e.g. an `i > 0` bottom-tested loop back-edge, which
/// wasmi rewrites to `0 < i`).
pub(crate) const MINI_BR_I64_LT_IR: i64 = 27;

/// `[MINI_I64_OR_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = slots[lhs] | slots[rhs]` (i64). Lowers wasmi `I64BitOr_Rss`.
pub(crate) const MINI_I64_OR_SS_WR: i64 = 28;
/// `[MINI_I64_SUB_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = slots[lhs] - slots[rhs]` (i64, wrapping; non-commutative). Lowers
/// wasmi `I64Sub_Rss`.
pub(crate) const MINI_I64_SUB_SS_WR: i64 = 29;
/// `[MINI_I32_XOR_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = wrap_i32(slots[lhs] ^ slots[rhs])`. The canonical slot-slot form for
/// i32 xor; reg-operand variants pre-materialize into a scratch slot.
pub(crate) const MINI_I32_XOR_SS_WR: i64 = 30;
/// `[MINI_I32_AND_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = wrap_i32(slots[lhs] & slots[rhs])`.
pub(crate) const MINI_I32_AND_SS_WR: i64 = 31;
/// `[MINI_I32_OR_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = wrap_i32(slots[lhs] | slots[rhs])`.
pub(crate) const MINI_I32_OR_SS_WR: i64 = 32;
/// `[MINI_I32_SUB_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = wrap_i32(slots[lhs] - slots[rhs])` (non-commutative).
pub(crate) const MINI_I32_SUB_SS_WR: i64 = 33;
/// `[MINI_BR_I32_LE_SS, target_word, lhs_slot, rhs_slot]` (4 words): if
/// `(i32)slots[lhs] <= (i32)slots[rhs]` (signed i32) jump to `target_word`.
/// Lowers wasmi `BranchI32Le_Ss`.
pub(crate) const MINI_BR_I32_LE_SS: i64 = 34;
/// `[MINI_BR_I64_EQ_SS, target_word, lhs_slot, rhs_slot]` (4 words): if
/// `slots[lhs] == slots[rhs]` (i64) jump to `target_word`. Lowers wasmi
/// `BranchI64Eq_Ss`.
pub(crate) const MINI_BR_I64_EQ_SS: i64 = 35;
/// `[MINI_I64_SHL_SI, src_slot, shift]` (3 words): `ireg = slots[src] << shift`
/// (i64 left shift by a constant). Lowers wasmi `I64Shl_Rsi`. Unlike the logical
/// right shift, a left shift needs no mask — `<<` zero-fills from the right.
pub(crate) const MINI_I64_SHL_SI: i64 = 36;
/// `[MINI_I32_SHL_SI, src_slot, shift]` (3 words):
/// `ireg = wrap_i32(slots[src] << shift)` (i32 left shift, `shift` mod 32). The
/// `<< 32 >> 32` re-canonicalizes the low 32 bits. Lowers wasmi `I32Shl_Rsi`.
pub(crate) const MINI_I32_SHL_SI: i64 = 37;
/// `[MINI_U32_SHR_RI, shift]` (2 words): `ireg = wrap_i32((u32)ireg >> shift)`
/// (i32 logical right shift of the accumulator by a constant). Masks to the low
/// 32 bits (zero-fill) then sign-extends. Lowers wasmi `U32Shr_Rri`.
pub(crate) const MINI_U32_SHR_RI: i64 = 38;
/// `[MINI_I32_LT_SI_R, lhs_slot, imm]` (3 words): a signed i32 comparison used as
/// a 0/1 VALUE — `ireg = if (i32)slots[lhs] < imm { 1 } else { 0 }` (`imm`
/// pre-sign-extended). Lowers wasmi `I32Lt_Rsi`. The `if … { 1 } else { 0 }`
/// form is how a compare RESULT (vs a compare-and-branch) traces.
pub(crate) const MINI_I32_LT_SI_R: i64 = 39;
/// `[MINI_I64_LT_IS_R, imm, rhs_slot]` (3 words): a signed i64 comparison used as
/// a 0/1 value — `ireg = if imm < slots[rhs] { 1 } else { 0 }` (the immediate is
/// the left operand). Lowers wasmi `I64Lt_Ris` (e.g. `i64.gt_s x c` rewritten to
/// `c < x`). No i32 wraparound since the operands are full i64.
pub(crate) const MINI_I64_LT_IS_R: i64 = 40;
/// `[MINI_I64_LT_SI_R, lhs_slot, imm]` (3 words): a signed i64 comparison used
/// as a 0/1 value — `ireg = if slots[lhs] < imm { 1 } else { 0 }`. Lowers wasmi
/// `I64Lt_Rsi`.
pub(crate) const MINI_I64_LT_SI_R: i64 = 137;
/// `[MINI_I32_EQ_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i32 `==` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I32_EQ_SS_R: i64 = 138;
/// `[MINI_I32_NE_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i32 `!=` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I32_NE_SS_R: i64 = 139;
/// `[MINI_I32_LT_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i32 `<` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I32_LT_SS_R: i64 = 140;
/// `[MINI_I32_LE_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i32 `<=` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I32_LE_SS_R: i64 = 141;
/// `[MINI_U32_LT_SS_R, lhs_slot, rhs_slot]` (3 words): an unsigned i32 `<` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_U32_LT_SS_R: i64 = 142;
/// `[MINI_U32_LE_SS_R, lhs_slot, rhs_slot]` (3 words): an unsigned i32 `<=` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_U32_LE_SS_R: i64 = 143;
/// `[MINI_I64_EQ_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i64 `==` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I64_EQ_SS_R: i64 = 144;
/// `[MINI_I64_NE_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i64 `!=` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I64_NE_SS_R: i64 = 145;
/// `[MINI_I64_LT_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i64 `<` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I64_LT_SS_R: i64 = 146;
/// `[MINI_I64_LE_SS_R, lhs_slot, rhs_slot]` (3 words): a signed i64 `<=` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_I64_LE_SS_R: i64 = 147;
/// `[MINI_U64_LT_SS_R, lhs_slot, rhs_slot]` (3 words): an unsigned i64 `<` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_U64_LT_SS_R: i64 = 148;
/// `[MINI_U64_LE_SS_R, lhs_slot, rhs_slot]` (3 words): an unsigned i64 `<=` used
/// as a 0/1 value into the int accumulator.
pub(crate) const MINI_U64_LE_SS_R: i64 = 149;
/// `[MINI_COPY_RS, src_slot]` (2 words): `ireg = slots[src]`. The inverse of
/// [`MINI_COPY_SR`] — loads a slot value into the integer accumulator.
pub(crate) const MINI_COPY_RS: i64 = 150;
/// `[MINI_COPY_RI, imm]` (2 words): `ireg = imm`. Loads an immediate constant
/// into the integer accumulator.
pub(crate) const MINI_COPY_RI: i64 = 151;
/// `[MINI_I64_REINTERP_F64]` (1 word): `ireg = freg64-bits` — a pure 64-bit
/// bit move from the f64 accumulator to the integer accumulator. Lowers
/// `I64ReinterpretF64_Rr`.
pub(crate) const MINI_I64_REINTERP_F64: i64 = 152;
/// `[MINI_F64_REINTERP_I64]` (1 word): `freg64 = ireg-bits` — a pure 64-bit
/// bit move from the integer accumulator to the f64 accumulator. Lowers
/// `F64ReinterpretI64_Rr`.
pub(crate) const MINI_F64_REINTERP_I64: i64 = 153;
/// `[MINI_BR_I64_NE_SS, target, lhs_slot, rhs_slot]` (4 words): branch to
/// `target` if `slots[lhs] != slots[rhs]` (i64 comparison). Also valid for
/// canonical i32 values (sign-extended i64).
pub(crate) const MINI_BR_I64_NE_SS: i64 = 154;
/// `[MINI_BR_U64_LT_SS, target, lhs_slot, rhs_slot]` (4 words): branch to
/// `target` if `(slots[lhs] as u64) < (slots[rhs] as u64)` (unsigned i64).
pub(crate) const MINI_BR_U64_LT_SS: i64 = 155;
/// `[MINI_RETURN_BAIL]` (1 word): signals to the caller that the kernel hit
/// an instruction it cannot execute (tail call). The kernel returns and the
/// caller falls back to the stock executor from the function start.
pub(crate) const MINI_RETURN_BAIL: i64 = 156;
/// `[MINI_YIELD_STOCK, byte_offset, num_slots]` (3 words): yield to the stock
/// executor AT the indicated byte offset. Unlike [`MINI_RETURN_BAIL`] (which
/// reruns the function from byte 0), this flushes the kernel's computed slots
/// to the real frame and resumes the stock executor at `byte_offset` — the
/// position of a CallInternal the kernel cannot handle. No double-apply of
/// side effects because execution continues from the exact instruction, not
/// from the start.
pub(crate) const MINI_YIELD_STOCK: i64 = 157;
/// `[MINI_CALL_RESIDUAL, func_addr, params_start, params_len]` (4 words):
/// execute an internal function call via a `#[dont_look_inside]` residual.
/// The kernel stages `slots[params_start..params_start+params_len]` into a
/// TLS buffer, then calls `call_internal_residual(func_addr, params_len)`.
/// The return value goes into `ireg` (integer accumulator). The JIT treats
/// the residual as an opaque call — the callee is not traced.
pub(crate) const MINI_CALL_RESIDUAL: i64 = 158;
/// `[MINI_TRAP, trap_code_u8]` (2 words): unconditional trap. Sets the trap
/// code via [`set_residual_trap`] and returns 0. The caller surfaces the trap
/// directly.
pub(crate) const MINI_TRAP: i64 = 159;
/// `[MINI_MEMORY_SIZE]` (1 word): return the current memory size in pages
/// (`mem_len / 65536`) into ireg. Reads from the [`MEM_CTX`] TLS.
pub(crate) const MINI_MEMORY_SIZE: i64 = 160;
/// `[MINI_U64_SHR_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `ireg = (slots[lhs] as u64 >> (slots[rhs] as u64 & 63)) as i64`.
/// Unsigned 64-bit right shift with dynamic shift amount.
pub(crate) const MINI_U64_SHR_SS_WR: i64 = 161;
/// `[MINI_MEM_COPY_WITHIN, dst_slot, src_slot, len_slot]` (4 words):
/// `memory.copy` within memory[0]. Reads dst/src/len from slots, performs
/// bounds-checked `copy_within` through a `#[dont_look_inside]` residual
/// helper. Out-of-bounds sets `MEM_TRAP`. This is same-memory only;
/// cross-memory MemoryCopy bails the function.
pub(crate) const MINI_MEM_COPY_WITHIN: i64 = 162;
/// `[MINI_I64_OR_RI_WR, imm]` (2 words): `ireg = ireg | imm` (i64). Lowers
/// wasmi `I64BitOr_Rsi` (`lhs` = accumulator, `rhs` = sign-extended Imm16).
pub(crate) const MINI_I64_OR_RI_WR: i64 = 163;
/// `[MINI_CALL_IMPORTED, func_idx, params_start, params_len]` (4 words):
/// residual call to an imported function. `func_idx` is the `index::Func`
/// (u32) resolved through the instance's function table at runtime.
/// Result is written to `slots[params_start]` (wasmi calling convention).
/// The residual re-extracts `MEM_CTX` after the call (host may `memory.grow`).
pub(crate) const MINI_CALL_IMPORTED: i64 = 164;
/// `[MINI_CALL_INDIRECT, table, func_type, index_slot, params_start, params_len]`
/// (6 words): residual indirect call. Performs table lookup + null check +
/// type check, then dispatches to Wasm or Host. `index_slot` is the slot
/// holding the runtime table index. Result to `slots[params_start]`.
pub(crate) const MINI_CALL_INDIRECT: i64 = 165;
/// `[MINI_I64_LOAD_MEM0_OFF, offset]` (2 words): an i64 load from the default
/// linear memory — `ireg = *(mem_base + (ireg & 0xffff_ffff) + offset)`. The
/// dynamic address is the accumulator (an unsigned 32-bit wasm address); the
/// static `offset` is added. Lowers wasmi `U64LoadMem0Offset16_Rr`. The access
/// goes through a `#[dont_look_inside]` residual helper (it stays a real call in
/// the compiled trace) that bounds-checks against `mem_len`; an out-of-bounds
/// access sets a trap flag and the caller re-runs the function on the stock
/// executor (which traps faithfully).
pub(crate) const MINI_I64_LOAD_MEM0_OFF: i64 = 41;
/// `[MINI_I32_LOAD_MEM0_OFF, offset]` (2 words): an i32 load from the default
/// linear memory — `ireg = sext32(*(mem_base + (ireg & 0xffff_ffff) + offset))`.
/// Reads 4 bytes (sign-extended to a canonical i32). Lowers wasmi
/// `U32LoadMem0Offset16_Rr`. Bounds-checked in the residual like the i64 load.
pub(crate) const MINI_I32_LOAD_MEM0_OFF: i64 = 42;
/// `[MINI_I64_SEXT32]` (1 word): `ireg = (ireg << 32) >> 32` — sign-extend the
/// low 32 bits. Lowers wasmi `I64Sext32_Rr` (`i64.extend_i32_s`).
pub(crate) const MINI_I64_SEXT32: i64 = 43;
/// `[MINI_U8_LOAD_MEM0_OFF, offset]` (2 words): an unsigned byte load from the
/// default linear memory — `ireg = *(mem_base + (ireg & 0xffff_ffff) + offset)`
/// as a zero-extended byte (0..=255). Lowers wasmi
/// `U32LoadExtend8Mem0Offset16_Rr` (`i32.load8_u`). Bounds-checked in the
/// residual.
pub(crate) const MINI_U8_LOAD_MEM0_OFF: i64 = 44;
/// `[MINI_I8_LOAD_MEM0_OFF, offset]` (2 words): a signed byte load from the
/// default linear memory — `ireg = *(mem_base + (ireg & 0xffff_ffff) + offset)`
/// as a sign-extended byte (-128..=127). Lowers wasmi
/// `I32LoadExtend8Mem0Offset16_Rr` (`i32.load8_s`). Bounds-checked in the
/// residual.
pub(crate) const MINI_I8_LOAD_MEM0_OFF: i64 = 45;
/// `[MINI_U16_LOAD_MEM0_OFF, offset]` (2 words): an unsigned 16-bit load — a
/// zero-extended halfword (0..=65535). Lowers `U32LoadExtend16Mem0Offset16_Rr`
/// (`i32.load16_u`). Bounds-checked in the residual.
pub(crate) const MINI_U16_LOAD_MEM0_OFF: i64 = 46;
/// `[MINI_I16_LOAD_MEM0_OFF, offset]` (2 words): a signed 16-bit load — a
/// sign-extended halfword (-32768..=32767). Lowers
/// `I32LoadExtend16Mem0Offset16_Rr` (`i32.load16_s`). Bounds-checked in the
/// residual.
pub(crate) const MINI_I16_LOAD_MEM0_OFF: i64 = 47;
/// `[MINI_I32_STORE_SR, ptr_slot, offset]` (3 words): a 32-bit store to the
/// default linear memory — `*(mem_base + (slots[ptr_slot] & 0xffff_ffff) +
/// offset) = ireg as u32`. Lowers wasmi `U32StoreMem0Offset16_Sr` (`i32.store`
/// with the pointer in a slot and the value in the accumulator). Bounds-checked
/// in the residual; an out-of-bounds store sets the trap flag and applies no
/// further stores.
pub(crate) const MINI_I32_STORE_SR: i64 = 48;
/// `[MINI_RETURN_VOID]` (1 word): return from a function with no results. Lowers
/// wasmi's bare `Return`. The kernel returns a dummy `0`; the caller of a
/// no-result function ignores the result slot.
pub(crate) const MINI_RETURN_VOID: i64 = 49;
/// `[MINI_I32_STORE_RS, offset, val_slot]` (3 words): a 32-bit store to the
/// default linear memory with the pointer in the accumulator and the value in a
/// slot — `*(mem_base + (ireg & 0xffff_ffff) + offset) = slots[val_slot] as u32`.
/// Lowers wasmi `U32StoreMem0Offset16_Rs` (`i32.store` of a local, where the
/// computed address stays in the accumulator). Same bounds-checked residual as
/// [`MINI_I32_STORE_SR`].
pub(crate) const MINI_I32_STORE_RS: i64 = 50;
/// `[MINI_I64_STORE_RS, offset, val_slot]` (3 words): a 64-bit store to the
/// default linear memory with the pointer in the accumulator and the value in a
/// slot — `*(mem_base + (ireg & 0xffff_ffff) + offset) = slots[val_slot] as u64`
/// (8 bytes). Lowers wasmi `U64StoreMem0Offset16_Rs` (`i64.store` of a local).
/// Bounds-checked (`ea + 8 > len`) like [`MINI_I32_STORE_RS`].
pub(crate) const MINI_I64_STORE_RS: i64 = 51;
/// `[MINI_I32_STORE8_RS, offset, val_slot]` (3 words): an 8-bit (wrapping) store —
/// `*(mem_base + (ireg & 0xffff_ffff) + offset) = slots[val_slot] as u8`. Lowers
/// wasmi `I32StoreWrap8Mem0Offset16_Rs` (`i32.store8`). Bounds-checked `ea + 1 >
/// len`.
pub(crate) const MINI_I32_STORE8_RS: i64 = 52;
/// `[MINI_I32_STORE16_RS, offset, val_slot]` (3 words): a 16-bit (wrapping) store —
/// `*(mem_base + (ireg & 0xffff_ffff) + offset) = slots[val_slot] as u16`. Lowers
/// wasmi `I32StoreWrap16Mem0Offset16_Rs` (`i32.store16`). Bounds-checked `ea + 2 >
/// len`.
pub(crate) const MINI_I32_STORE16_RS: i64 = 53;
/// `[MINI_I64_STORE_SR, ptr_slot, offset]` (3 words): a 64-bit store of a computed
/// value — `*(mem_base + (slots[ptr_slot] & 0xffff_ffff) + offset) = ireg as u64`.
/// Lowers wasmi `U64StoreMem0Offset16_Sr` (pointer in a slot, value in the
/// accumulator). Bounds-checked `ea + 8 > len`.
pub(crate) const MINI_I64_STORE_SR: i64 = 54;
/// `[MINI_I32_STORE8_SR, ptr_slot, offset]` (3 words): an 8-bit (wrapping) store of
/// a computed value — `... = ireg as u8`. Lowers `I32StoreWrap8Mem0Offset16_Sr`.
pub(crate) const MINI_I32_STORE8_SR: i64 = 55;
/// `[MINI_I32_STORE16_SR, ptr_slot, offset]` (3 words): a 16-bit (wrapping) store of
/// a computed value — `... = ireg as u16`. Lowers `I32StoreWrap16Mem0Offset16_Sr`.
pub(crate) const MINI_I32_STORE16_SR: i64 = 56;
/// `[MINI_I32_LT_RS_R, rhs_slot]` (2 words): a signed i32 comparison used as a 0/1
/// VALUE with the left operand in the accumulator — `ireg = if (i32)ireg <
/// (i32)slots[rhs] { 1 } else { 0 }`. Lowers wasmi `I32Lt_Rrs` (e.g. counting
/// `mem[i] < threshold`). Both operands are sign-extended from their low 32 bits.
pub(crate) const MINI_I32_LT_RS_R: i64 = 57;
/// `[MINI_I32_LT_SR_R, lhs_slot]` (2 words): a signed i32 comparison used as a 0/1
/// VALUE with the right operand in the accumulator — `ireg = if (i32)slots[lhs] <
/// (i32)ireg { 1 } else { 0 }`. Lowers wasmi `I32Lt_Rsr` (e.g. counting `mem[i] >
/// threshold`, which lowers to `threshold < mem[i]`). Both operands are
/// sign-extended from their low 32 bits.
pub(crate) const MINI_I32_LT_SR_R: i64 = 58;
/// `[MINI_SELECT, true_slot, false_slot]` (3 words): a branchless `select` —
/// `ireg = if (i32)ireg != 0 { slots[true_slot] } else { slots[false_slot] }`,
/// the condition in the accumulator. Lowers wasmi `U64Select_Rrss`. Implemented
/// as `f + (t - f) * c` with `c ∈ {0, 1}` (a value-`if` over non-constant arms
/// aborts the trace; the arithmetic form is exact in wrapping i64).
pub(crate) const MINI_SELECT: i64 = 59;
/// `[MINI_I32_EQ_RS_R, rhs_slot]` (2 words): an i32 equality used as a 0/1 VALUE
/// with the left operand in the accumulator — `ireg = if (i32)ireg ==
/// (i32)slots[rhs] { 1 } else { 0 }`. Lowers wasmi `I32Eq_Rrs` (e.g. counting
/// occurrences `mem[i] == target`). Both operands are sign-extended from low 32.
pub(crate) const MINI_I32_EQ_RS_R: i64 = 60;
/// `[MINI_I32_NE_RS_R, rhs_slot]` (2 words): an i32 inequality used as a 0/1 VALUE
/// with the left operand in the accumulator — `ireg = if (i32)ireg !=
/// (i32)slots[rhs] { 1 } else { 0 }`. Lowers wasmi `I32NotEq_Rrs`.
pub(crate) const MINI_I32_NE_RS_R: i64 = 61;
/// `[MINI_I64_EQ_RS_R, rhs_slot]` (2 words): an i64 equality used as a 0/1 VALUE
/// with the left operand in the accumulator — `ireg = if ireg == slots[rhs] { 1 }
/// else { 0 }`. Lowers wasmi `I64Eq_Rrs`. Full i64, no sign-extension.
pub(crate) const MINI_I64_EQ_RS_R: i64 = 62;
/// `[MINI_I64_NE_RS_R, rhs_slot]` (2 words): an i64 inequality used as a 0/1 VALUE
/// with the left operand in the accumulator — `ireg = if ireg != slots[rhs] { 1 }
/// else { 0 }`. Lowers wasmi `I64NotEq_Rrs`. Full i64.
pub(crate) const MINI_I64_NE_RS_R: i64 = 63;
/// `[MINI_I64_LT_RS_R, rhs_slot]` (2 words): a signed i64 less-than used as a 0/1
/// VALUE with the left operand in the accumulator — `ireg = if ireg < slots[rhs]
/// { 1 } else { 0 }`. Lowers wasmi `I64Lt_Rrs`. Full i64, no wraparound.
pub(crate) const MINI_I64_LT_RS_R: i64 = 64;
/// `[MINI_I32_LE_RS_R, rhs_slot]` (2 words): a signed i32 `<=` used as a 0/1 VALUE
/// with the left operand in the accumulator — `ireg = if (i32)ireg <=
/// (i32)slots[rhs] { 1 } else { 0 }`. Lowers wasmi `I32Le_Rrs`. Both operands
/// sign-extended from their low 32 bits.
pub(crate) const MINI_I32_LE_RS_R: i64 = 65;
/// `[MINI_U32_LT_RS_R, rhs_slot]` (2 words): an UNSIGNED i32 `<` used as a 0/1
/// VALUE with the left operand in the accumulator — `ireg = if (ireg & 0xffffffff)
/// < (slots[rhs] & 0xffffffff) { 1 } else { 0 }`. Lowers wasmi `U32Lt_Rrs`. The
/// mask makes both operands non-negative i64 so the signed `<` realizes the
/// unsigned order.
pub(crate) const MINI_U32_LT_RS_R: i64 = 66;
/// `[MINI_U32_LE_RS_R, rhs_slot]` (2 words): an UNSIGNED i32 `<=` used as a 0/1
/// VALUE — `ireg = if (ireg & 0xffffffff) <= (slots[rhs] & 0xffffffff) { 1 } else
/// { 0 }`. Lowers wasmi `U32Le_Rrs`.
pub(crate) const MINI_U32_LE_RS_R: i64 = 67;
/// `[MINI_F64_ARITH_RS, sel, rhs_slot]` (3 words): `freg64 = freg64 OP slots[rhs]`
/// where `sel` is 0=add, 1=sub, 2=mul, 3=div — all as raw f64 bit-patterns. One
/// selector-dispatched residual (`f64_arith`) folds the four operations into a
/// single force-inlined arm, keeping the dispatch below the 256 register/const
/// ceiling. Add is commutative, so an `slot + acc` form (`F64Add_Rsr`) lowers here
/// too; sub/div take the accumulator as the left operand (wasm order). Lowers
/// `F64Add_Rsr`, `F64Sub_Rrs`, `F64Mul_Rrs`, `F64Div_Rrs` and the `_Rri` immediate
/// forms (via a pre-materialized scratch slot).
pub(crate) const MINI_F64_ARITH_RS: i64 = 68;
/// `[MINI_F64_CMP_RS_R, sel, rhs_slot]` (3 words): an f64 compare used as a 0/1
/// VALUE, left operand in `freg64`; the integer 0/1 result lands in `ireg`. `sel`
/// is 0=lt, 1=le, 2=eq, 3=ne, 4=lt swapped, 5=le swapped. One selector residual
/// (`f64_cmp`) folds the compare forms. Lowers `F64Lt_Rrs`, `F64Le_Rrs`,
/// `F64Eq_Rrs`, `F64NotEq_Rrs`, `F64Lt_Rsr`, `F64Le_Rsr`.
pub(crate) const MINI_F64_CMP_RS_R: i64 = 69;
/// `[MINI_F64_UNARY_S, sel, src_slot]` (3 words): `freg64 = OP(slots[src])` (f64).
/// `sel` is 0=abs, 1=neg, 2=sqrt, 3=ceil, 4=floor, 5=trunc, 6=nearest. One selector
/// residual (`f64_unary`) folds the seven ops. The accumulator-input forms (`_Rr`)
/// copy `freg64` into a scratch slot first via [`MINI_COPY_S_FR`]; the slot forms
/// (`_Rs`) map directly.
pub(crate) const MINI_F64_UNARY_S: i64 = 70;
/// `[MINI_F64_MINMAX_RS, sel, rhs_slot]` (3 words): `freg64 = f64_minmax(sel,
/// freg64, slots[rhs])`. `sel` is 0=min, 1=max (wasm NaN-propagating /
/// signed-zero semantics), 2=copysign(freg64, slot), 3=copysign(slot, freg64).
/// min/max are commutative, so the slot-then-accumulator (`_Rsr`) and constant
/// (`_Rri`, pre-materialized) forms also map here; copysign is not commutative,
/// so its `_Rsr` form uses `sel` 3 to swap the magnitude and sign operands.
pub(crate) const MINI_F64_MINMAX_RS: i64 = 71;
/// `[MINI_F64_CVT_S, sel, src_slot]` (3 words): int (slot) → f64 (`freg64`). `sel`
/// is 0=i32_s, 1=u32, 2=i64_s, 3=u64. One selector residual (`f64_convert`) folds
/// the four. The accumulator-input form (`_Rr`) copies the *integer* accumulator
/// (`ireg`) into a scratch slot first via [`MINI_COPY_SR`]; the slot form (`_Rs`)
/// maps directly.
pub(crate) const MINI_F64_CVT_S: i64 = 85;
/// `[MINI_I64_SEXT32_S, src_slot]` (2 words): `ireg = (slots[src] << 32) >> 32` —
/// sign-extend the low 32 bits of a slot. Lowers wasmi `I64Sext32_Rs`
/// (`i64.extend_i32_s` of a local).
pub(crate) const MINI_I64_SEXT32_S: i64 = 89;
/// `[MINI_F64_TRUNC_SAT_S, sel, src_slot]` (3 words): saturating f64→int (NaN→0,
/// out-of-range clamps; never traps). `sel` is 0=i32_s, 1=u32, 2=i64_s, 3=u64. The
/// f64 source is a slot, the integer result the integer accumulator; one selector
/// residual (`f64_trunc_sat`) folds the four. The accumulator-input form (`_Rr`)
/// copies the *f64* accumulator (`freg64`) into a scratch slot first via
/// [`MINI_COPY_S_FR`]; the slot form (`_Rs`) maps directly.
pub(crate) const MINI_F64_TRUNC_SAT_S: i64 = 90;
/// `[MINI_F64_TRUNC_S, sel, src_slot]` (3 words): trapping f64→int (NaN /
/// out-of-range trap with the exact wasm trap code via the residual's
/// `set_residual_trap`). `sel` is 0=i32_s, 1=u32, 2=i64_s, 3=u64. Same `_Rr`
/// scratch-copy / `_Rs` direct lowering as the saturating forms.
pub(crate) const MINI_F64_TRUNC_S: i64 = 94;
/// `[MINI_I32_ROTL_SI, src_slot, k]` (3 words): `ireg = wrap_i32(rotl32(slots[src], k))`
/// with `k` in `1..=31`. Rotates the low 32 bits left by `k`:
/// `(x << k) | (x >> (32 - k))` on the zero-extended low 32 bits, then
/// `<< 32 >> 32` re-canonicalizes. Lowers wasmi `I32Rotl_Rsi`; the prepass
/// folds `k == 0` to an identity `MINI_I32_SHL_SI src 0`.
pub(crate) const MINI_I32_ROTL_SI: i64 = 98;
/// `[MINI_I32_ROTR_SI, src_slot, k]` (3 words): `ireg = wrap_i32(rotr32(slots[src], k))`
/// with `k` in `1..=31`. Rotates the low 32 bits right by `k`:
/// `(x >> k) | (x << (32 - k))` on the zero-extended low 32 bits, then
/// `<< 32 >> 32` re-canonicalizes. Lowers wasmi `I32Rotr_Rsi`; the prepass
/// folds `k == 0` to an identity `MINI_I32_SHL_SI src 0`.
pub(crate) const MINI_I32_ROTR_SI: i64 = 99;
/// `[MINI_COPY_S_FR, dst_slot]` (2 words): `slots[dst] = freg64` (the f64
/// accumulator, `accum[1]`). The f64 counterpart of [`MINI_COPY_SR`], which
/// spills the integer accumulator (`accum[0]`). Lowers wasmi's f64 accumulator
/// spills (`F64Copy_S{0..9}r`, `F64Copy_Sr`) and the scratch-copy that f64-input
/// `_Rr` ops (f64 unary, f64→int truncation) emit before their slot-form arm.
pub(crate) const MINI_COPY_S_FR: i64 = 100;
/// `[MINI_F64_LOAD_MEM0_OFF, offset]` (2 words): an f64 load from the default
/// linear memory — `freg64 = *(mem_base + (ireg & 0xffff_ffff) + offset)` (a
/// bit-identical 8-byte read; the address is the integer accumulator, the result
/// lands in the f64 accumulator `accum[1]`). Lowers wasmi
/// `F64LoadMem0Offset16_Rr`. Bounds-checked in the same residual as
/// [`MINI_I64_LOAD_MEM0_OFF`].
pub(crate) const MINI_F64_LOAD_MEM0_OFF: i64 = 101;
/// `[MINI_F64_STORE_SR, ptr_slot, offset]` (3 words): an f64 store of a computed
/// value — `*(mem_base + (slots[ptr_slot] & 0xffff_ffff) + offset) = freg64` (a
/// bit-identical 8-byte store; the pointer is a slot, the value the f64
/// accumulator `accum[1]`). Lowers wasmi `F64StoreMem0Offset16_Sr`.
/// Bounds-checked in the same residual as [`MINI_I64_STORE_SR`].
pub(crate) const MINI_F64_STORE_SR: i64 = 102;
/// `[MINI_I32_BITCOUNT_S, sel, src_slot]` (3 words): `ireg = bitcount(sel, low 32
/// bits of slots[src])`, sel 0=clz / 1=ctz / 2=popcnt (all results 0..=32). One
/// selector residual (`i32_bitcount`) folds the three; lowers wasmi
/// `I32{Clz,Ctz,Popcnt}_Rs`, the accumulator-input `_Rr` forms copying `ireg` into
/// a scratch slot first via [`MINI_COPY_SR`]. The intrinsics live in a residual
/// (never traced, never trap). Opcode values 104/105 are now unused gaps.
pub(crate) const MINI_I32_BITCOUNT_S: i64 = 103;
/// `[MINI_I64_BITCOUNT_S, sel, src_slot]` (3 words): `ireg = bitcount(sel,
/// slots[src])` over all 64 bits, sel 0=clz / 1=ctz / 2=popcnt (all 0..=64). Lowers
/// wasmi `I64{Clz,Ctz,Popcnt}_Rs` / `_Rr`. Opcode values 107/108 are unused gaps.
pub(crate) const MINI_I64_BITCOUNT_S: i64 = 106;
/// Integer division / remainder, `[MINI_*, lhs_slot, rhs_slot]` (3 words):
/// `ireg = op(slots[lhs], slots[rhs])`. Non-commutative, so the prepass
/// pre-materializes any reg/immediate operand into a scratch slot preserving wasm
/// operand order, then emits the uniform two-slot form. These CAN trap (division
/// by zero, signed `INT_MIN / -1` overflow); the residual latches the trap and
/// `run_jit` recovers via the stock executor — see the `i32_div_s` residual.
/// `MINI_I32_DIV_S` lowers wasmi `I32Div_R{ss,rs,sr,ri,si,ir,is}` (`i32.div_s`).
pub(crate) const MINI_I32_DIV_S: i64 = 109;
/// `i32.div_u` (unsigned), result canonicalized to a sign-extended i64.
pub(crate) const MINI_I32_DIV_U: i64 = 110;
/// `i32.rem_s` (signed remainder).
pub(crate) const MINI_I32_REM_S: i64 = 111;
/// `i32.rem_u` (unsigned remainder), result sign-extended.
pub(crate) const MINI_I32_REM_U: i64 = 112;
/// `i64.div_s` (signed 64-bit division).
pub(crate) const MINI_I64_DIV_S: i64 = 113;
/// `i64.div_u` (unsigned 64-bit division).
pub(crate) const MINI_I64_DIV_U: i64 = 114;
/// `i64.rem_s` (signed 64-bit remainder).
pub(crate) const MINI_I64_REM_S: i64 = 115;
/// `i64.rem_u` (unsigned 64-bit remainder).
pub(crate) const MINI_I64_REM_U: i64 = 116;

/// f32 arithmetic / memory ops. An f32 value is held as its raw 32 bits in the
/// low half of an i64 (a slot, or the f32 accumulator `freg32` = `accum[2]`).
/// The bit-casts live in `#[dont_look_inside]` residuals (`f32_add` …), never in
/// the trace — the same pattern as the f64 ops, but routed to `accum[2]`.
///
/// `[MINI_F32_LOAD_MEM0_OFF, offset]` (2 words): `freg32 = load4(ireg + offset)`
/// (reuses the bounds-checked 4-byte i32 load). Lowers `F32LoadMem0Offset16_Rr`.
pub(crate) const MINI_F32_LOAD_MEM0_OFF: i64 = 117;
/// `[MINI_F32_STORE_SR, ptr_slot, offset]` (3 words): store the low 4 bytes of
/// `freg32` at `slots[ptr_slot] + offset`. Lowers `F32StoreMem0Offset16_Sr`.
pub(crate) const MINI_F32_STORE_SR: i64 = 118;
/// `[MINI_COPY_S_F32R, dst_slot]` (2 words): `slots[dst] = freg32` (spill the f32
/// accumulator). The f32 counterpart of `MINI_COPY_S_FR`. Lowers `F32Copy_*`.
pub(crate) const MINI_COPY_S_F32R: i64 = 119;
/// `[MINI_F32_ARITH_RS, sel, rhs_slot]` (3 words): `freg32 = freg32 OP slots[rhs]`
/// where `sel` is 0=add, 1=sub, 2=mul, 3=div. One selector-dispatched residual
/// (`f32_arith`) folds the four operations into a single force-inlined arm, keeping
/// the dispatch below the 256 register/const ceiling. Add is commutative, so an
/// `slot + acc` form lowers here too; sub/div take the accumulator as the left
/// operand (wasm order). Lowers `F32Add_Rsr`, `F32Sub_Rrs`, `F32Mul_Rrs`, `F32Div_Rrs`.
pub(crate) const MINI_F32_ARITH_RS: i64 = 120;
/// `[MINI_F32_MINMAX_RS, sel, rhs_slot]` (3 words): `freg32 = f32_minmax(sel,
/// freg32, slots[rhs])`. `sel` is 0=min, 1=max (wasm NaN-propagating /
/// signed-zero semantics), 2=copysign(freg32, slot), 3=copysign(slot, freg32).
/// min/max are commutative, so the slot-then-accumulator (`_Rsr`) and constant
/// (`_Rri`, pre-materialized) forms also map here; copysign is not commutative,
/// so its `_Rsr` form uses `sel` 3 to swap the magnitude and sign operands.
pub(crate) const MINI_F32_MINMAX_RS: i64 = 136;
/// `[MINI_F32_CMP_RS_R, sel, rhs_slot]` (3 words): an f32 compare used as a 0/1
/// VALUE, left operand in `freg32`; the integer 0/1 result lands in `ireg`. `sel`
/// is 0=lt, 1=le, 2=eq, 3=ne, 4=lt swapped, 5=le swapped. One selector residual
/// (`f32_cmp`) folds the compare forms. Lowers `F32Lt_Rrs`, `F32Le_Rrs`,
/// `F32Eq_Rrs`, `F32NotEq_Rrs`, `F32Lt_Rsr`, `F32Le_Rsr`.
pub(crate) const MINI_F32_CMP_RS_R: i64 = 121;
/// `[MINI_F32_UNARY_S, sel, src_slot]` (3 words): `freg32 = OP(slots[src])` (f32).
/// `sel` is 0=abs, 1=neg, 2=sqrt, 3=ceil, 4=floor, 5=trunc, 6=nearest. One selector
/// residual (`f32_unary`) folds the seven ops. The accumulator-input forms (`_Rr`)
/// copy `freg32` into a scratch slot first via [`MINI_COPY_S_F32R`]; the slot forms
/// (`_Rs`) map directly.
pub(crate) const MINI_F32_UNARY_S: i64 = 122;
/// `[MINI_F32_CVT_S, sel, src_slot]` (3 words): integer (slot) → f32 (`freg32`).
/// `sel` is 0=i32_s, 1=u32, 2=i64_s, 3=u64. One selector residual (`f32_convert`)
/// folds the four. Lowers `F32Convert{I,U}{32,64}_R{r,s}` (`_Rr` copies `ireg` into
/// a scratch slot via [`MINI_COPY_SR`]).
pub(crate) const MINI_F32_CVT_S: i64 = 123;
/// `[MINI_F64_PROMOTE_S, src_slot]` (2 words): f32 (slot) → f64 (`freg64`) via the
/// `promote_f32_f64` residual. Lowers `F64PromoteF32_R{r,s}` (`_Rr` copies `freg32`
/// into a scratch slot via [`MINI_COPY_S_F32R`]).
pub(crate) const MINI_F64_PROMOTE_S: i64 = 124;
/// `[MINI_F32_DEMOTE_S, src_slot]` (2 words): f64 (slot) → f32 (`freg32`) via the
/// `demote_f64_f32` residual. Lowers `F32DemoteF64_R{r,s}` (`_Rr` copies `freg64`
/// into a scratch slot via [`MINI_COPY_S_FR`]).
pub(crate) const MINI_F32_DEMOTE_S: i64 = 125;
/// `[MINI_I32_REINTERP_F32]` (1 word): `ireg = (i32) freg32-bits` — a pure bit move
/// (no residual). Lowers `I32ReinterpretF32_Rr`.
pub(crate) const MINI_I32_REINTERP_F32: i64 = 126;
/// `[MINI_F32_REINTERP_I32]` (1 word): `freg32 = low32(ireg)` — a pure bit move.
/// Lowers `F32ReinterpretI32_Rr`.
pub(crate) const MINI_F32_REINTERP_I32: i64 = 127;
/// `[MINI_F32_TRUNC_SAT_S, sel, src_slot]` (3 words): saturating f32→int (NaN→0,
/// out-of-range clamps; never traps). `sel` is 0=i32_s, 1=u32, 2=i64_s, 3=u64. The
/// f32 source is a slot, the integer result `ireg`; one selector residual
/// (`f32_trunc_sat`) folds the four. `_Rr` copies `freg32` into a scratch slot via
/// [`MINI_COPY_S_F32R`].
pub(crate) const MINI_F32_TRUNC_SAT_S: i64 = 128;
/// `[MINI_F32_TRUNC_S, sel, src_slot]` (3 words): trapping f32→int (NaN /
/// out-of-range trap via the residual's `set_residual_trap`). `sel` is 0=i32_s,
/// 1=u32, 2=i64_s, 3=u64. Same `_Rr` scratch-copy / `_Rs` direct lowering.
pub(crate) const MINI_F32_TRUNC_S: i64 = 129;

/// `[MINI_GLOBAL_GET_R, global_idx]` (2 words): `ireg = global_get(global_idx)`,
/// reading the raw `lo64` bits of an integer global into the accumulator. This is
/// bit-identical to the stock `GlobalGetU64_R` handler (which reads the same raw
/// `u64` into `ireg`); `GlobalGetU64_R` is only emitted for `i32`/`i64` globals
/// (`f32`/`f64` globals use `GlobalGet{F32,F64}_R`, which the prepass leaves
/// unlowered), so the integer accumulator is always the right destination. The
/// per-run global raw pointers are read from `GLOBALS_CTX`, set before each run.
pub(crate) const MINI_GLOBAL_GET_R: i64 = 130;
/// `[MINI_GLOBAL_SET_S, global_idx, src_slot]` (3 words): writes `slots[src_slot]`
/// raw bits into integer global `global_idx`. Register and immediate value sources
/// are pre-materialized into a scratch slot (via [`MINI_COPY_SR`] / [`MINI_COPY_SI`])
/// so this one slot-sourced arm covers every `GlobalSet{U64_R,U64_S,U64_I,U32_I}`
/// form. Writing the accumulator's full `i64` bits matches the stock handler for
/// `i64` globals exactly, and for `i32` globals only the low 32 bits are ever read
/// back (by both the JIT and the host `TypedRawVal`), so the high bits are unobserved.
pub(crate) const MINI_GLOBAL_SET_S: i64 = 131;
/// `[MINI_GLOBAL_GET_F32, global_idx]` (2 words): `freg32 = global_get(idx) low 32
/// bits`, reading an `f32` global's raw bits into the f32 accumulator. Reuses the
/// integer `global_get` residual (raw `lo64`); only the placement differs.
pub(crate) const MINI_GLOBAL_GET_F32: i64 = 132;
/// `[MINI_GLOBAL_GET_F64, global_idx]` (2 words): `freg64 = global_get(idx)`,
/// reading an `f64` global's raw bits into the f64 accumulator. `f32`/`f64` global
/// SETs reuse [`MINI_GLOBAL_SET_S`] after spilling the float accumulator into a
/// scratch slot (via [`MINI_COPY_S_F32R`] / [`MINI_COPY_S_FR`]).
pub(crate) const MINI_GLOBAL_GET_F64: i64 = 133;
/// `[MINI_RETURN_F_R]` (1 word): return the f64 accumulator (`freg64`) bits as the
/// function result. The bit pattern is written to result slot 0 and read back as
/// `f64` by the caller, matching the stock `ReturnF64_R`.
pub(crate) const MINI_RETURN_F_R: i64 = 134;
/// `[MINI_RETURN_F32_R]` (1 word): return the f32 accumulator (`freg32`) bits (low
/// 32) as the function result, matching the stock `ReturnF32_R`.
pub(crate) const MINI_RETURN_F32_R: i64 = 135;

/// Scratch slots reserved past the wasm frame's real slots, used by the prepass
/// to PRE-MATERIALIZE a reg/immediate operand into a slot so an op with mixed
/// addressing modes can reuse one uniform slot-slot kernel arm (a `MINI_COPY_S*`
/// into a scratch slot, then the slot-slot op). The copies fold away in the
/// compiled trace. The kernel's `slots` array (and the caller's seed) is sized
/// `num_slots + NUM_SCRATCH`; scratch indices are `num_slots .. num_slots + N`.
/// Four covers a binary op whose two operands both need materializing.
pub(crate) const NUM_SCRATCH: usize = 4;

// ── Scratch-dedicated ops ──────────────────────────────────────────────────
//
// These ops use the kernel's `state.scratch0` / `state.scratch1` scalar
// fields directly instead of routing through the `slots` virtualizable
// array. Keeping scratch out of `slots` reduces the JIT's close-loop
// inputargs count and fixes the LABEL/JUMP arity mismatch that the POC
// surfaced on trivial loops.

/// `[MINI_COPY_SCRATCH0_R]` (1 word): `scratch0 = accum0` — save the integer
/// accumulator to the scratch0 scalar register.
pub(crate) const MINI_COPY_SCRATCH0_R: i64 = 171;
/// `[MINI_COPY_SCRATCH0_I, imm]` (2 words): `scratch0 = imm` — load an
/// immediate constant into scratch0.
pub(crate) const MINI_COPY_SCRATCH0_I: i64 = 172;
/// `[MINI_COPY_SCRATCH0_FR]` (1 word): `scratch0 = accum1` — save the f64
/// accumulator bits to scratch0.
pub(crate) const MINI_COPY_SCRATCH0_FR: i64 = 173;
/// `[MINI_COPY_SCRATCH0_F32R]` (1 word): `scratch0 = accum2` — save the f32
/// accumulator bits to scratch0.
pub(crate) const MINI_COPY_SCRATCH0_F32R: i64 = 174;
/// `[MINI_COPY_SCRATCH1_R]` (1 word): `scratch1 = accum0` — save the integer
/// accumulator to the scratch1 scalar register.
pub(crate) const MINI_COPY_SCRATCH1_R: i64 = 175;
/// `[MINI_COPY_SCRATCH1_I, imm]` (2 words): `scratch1 = imm` — load an
/// immediate constant into scratch1.
pub(crate) const MINI_COPY_SCRATCH1_I: i64 = 176;

// ── Binary RS ops (accum OP slot → accum) ──────────────────────────────────
/// `[MINI_I32_AND_RS_WR, rhs_slot]` (2 words): `accum = wrap_i32(accum & slot)`.
pub(crate) const MINI_I32_AND_RS_WR: i64 = 177;
/// `[MINI_I32_OR_RS_WR, rhs_slot]` (2 words): `accum = wrap_i32(accum | slot)`.
pub(crate) const MINI_I32_OR_RS_WR: i64 = 178;
/// `[MINI_I32_SUB_RS_WR, rhs_slot]` (2 words): `accum = wrap_i32(accum - slot)`.
pub(crate) const MINI_I32_SUB_RS_WR: i64 = 179;
/// `[MINI_I32_MUL_RS_WR, rhs_slot]` (2 words): `accum = wrap_i32(accum * slot)`.
pub(crate) const MINI_I32_MUL_RS_WR: i64 = 180;
/// `[MINI_I32_XOR_RS_WR, rhs_slot]` (2 words): `accum = wrap_i32(accum ^ slot)`.
pub(crate) const MINI_I32_XOR_RS_WR: i64 = 181;
/// `[MINI_I64_SUB_RS_WR, rhs_slot]` (2 words): `accum = accum - slot`.
pub(crate) const MINI_I64_SUB_RS_WR: i64 = 182;
/// `[MINI_I64_MUL_RS_WR, rhs_slot]` (2 words): `accum = accum * slot`.
pub(crate) const MINI_I64_MUL_RS_WR: i64 = 183;
/// `[MINI_I64_OR_RS_WR, rhs_slot]` (2 words): `accum = accum | slot`.
pub(crate) const MINI_I64_OR_RS_WR: i64 = 184;
/// `[MINI_I64_XOR_RS_WR, rhs_slot]` (2 words): `accum = accum ^ slot`.
pub(crate) const MINI_I64_XOR_RS_WR: i64 = 185;

// ── Binary SR ops (slot OP accum → accum) ──────────────────────────────────
/// `[MINI_I32_SUB_SR_WR, lhs_slot]` (2 words): `accum = wrap_i32(slot - accum)`.
pub(crate) const MINI_I32_SUB_SR_WR: i64 = 186;
/// `[MINI_I64_SUB_SR_WR, lhs_slot]` (2 words): `accum = slot - accum`.
pub(crate) const MINI_I64_SUB_SR_WR: i64 = 187;

// ── Add without slot writeback (replaces throwaway-scratch WB) ─────────────
/// `[MINI_I32_ADD_RS_WR, rhs_slot]` (2 words): `accum = wrap_i32(accum + slot)`.
pub(crate) const MINI_I32_ADD_RS_WR: i64 = 188;
/// `[MINI_I64_ADD_RS_WR, rhs_slot]` (2 words): `accum = accum + slot`.
pub(crate) const MINI_I64_ADD_RS_WR: i64 = 189;
/// `[MINI_I32_ADD_SS_WR, lhs_slot, rhs_slot]` (3 words):
/// `accum = wrap_i32(slot_l + slot_r)` — no slot writeback.
pub(crate) const MINI_I32_ADD_SS_WR: i64 = 190;

// ── Div/rem with scratch operands (selector-dispatched) ────────────────────
/// `[MINI_DIVREM_SCRATCH0_S, sel, rhs_slot]` (3 words):
/// `accum = divrem(sel, scratch0, slot)`. `sel` encodes which div/rem variant.
pub(crate) const MINI_DIVREM_SCRATCH0_S: i64 = 191;
/// `[MINI_DIVREM_S_SCRATCH0, sel, lhs_slot]` (3 words):
/// `accum = divrem(sel, slot, scratch0)`. `sel` encodes which div/rem variant.
pub(crate) const MINI_DIVREM_S_SCRATCH0: i64 = 192;
/// `[MINI_DIVREM_SCRATCH01, sel]` (2 words):
/// `accum = divrem(sel, scratch0, scratch1)`. `sel` encodes which div/rem variant.
pub(crate) const MINI_DIVREM_SCRATCH01: i64 = 193;

// ── Bitcount/unary/convert from scratch0 ───────────────────────────────────
/// `[MINI_I32_BITCOUNT_SCRATCH0, sel]` (2 words): `accum = bitcount(sel, scratch0)`.
pub(crate) const MINI_I32_BITCOUNT_SCRATCH0: i64 = 194;
/// `[MINI_I64_BITCOUNT_SCRATCH0, sel]` (2 words): `accum = bitcount(sel, scratch0)`.
pub(crate) const MINI_I64_BITCOUNT_SCRATCH0: i64 = 195;
/// `[MINI_F32_UNARY_SCRATCH0, sel]` (2 words): `freg32 = unary(sel, scratch0)`.
pub(crate) const MINI_F32_UNARY_SCRATCH0: i64 = 196;
/// `[MINI_F64_UNARY_SCRATCH0, sel]` (2 words): `freg64 = unary(sel, scratch0)`.
pub(crate) const MINI_F64_UNARY_SCRATCH0: i64 = 197;
/// `[MINI_F32_CVT_SCRATCH0, sel]` (2 words): `freg32 = convert(sel, scratch0)`.
pub(crate) const MINI_F32_CVT_SCRATCH0: i64 = 198;
/// `[MINI_F64_CVT_SCRATCH0, sel]` (2 words): `freg64 = convert(sel, scratch0)`.
pub(crate) const MINI_F64_CVT_SCRATCH0: i64 = 199;
/// `[MINI_F64_PROMOTE_SCRATCH0]` (1 word): `freg64 = promote(scratch0)`.
pub(crate) const MINI_F64_PROMOTE_SCRATCH0: i64 = 200;
/// `[MINI_F32_DEMOTE_SCRATCH0]` (1 word): `freg32 = demote(scratch0)`.
pub(crate) const MINI_F32_DEMOTE_SCRATCH0: i64 = 201;
/// `[MINI_F32_TRUNC_SAT_SCRATCH0, sel]` (2 words): `accum = trunc_sat(sel, scratch0)`.
pub(crate) const MINI_F32_TRUNC_SAT_SCRATCH0: i64 = 202;
/// `[MINI_F64_TRUNC_SAT_SCRATCH0, sel]` (2 words): `accum = trunc_sat(sel, scratch0)`.
pub(crate) const MINI_F64_TRUNC_SAT_SCRATCH0: i64 = 203;
/// `[MINI_F32_TRUNC_SCRATCH0, sel]` (2 words): `accum = trunc(sel, scratch0)`.
pub(crate) const MINI_F32_TRUNC_SCRATCH0: i64 = 204;
/// `[MINI_F64_TRUNC_SCRATCH0, sel]` (2 words): `accum = trunc(sel, scratch0)`.
pub(crate) const MINI_F64_TRUNC_SCRATCH0: i64 = 205;

// ── Branch ops with accumulator operand ────────────────────────────────────
/// `[MINI_BR_I64_NE_RS, target, rhs_slot]` (3 words): if `accum != slot` jump.
pub(crate) const MINI_BR_I64_NE_RS: i64 = 206;
/// `[MINI_BR_I64_EQ_RS, target, rhs_slot]` (3 words): if `accum == slot` jump.
pub(crate) const MINI_BR_I64_EQ_RS: i64 = 207;
/// `[MINI_BR_I64_EQ_RI, target, imm]` (3 words): if `accum == imm` jump.
pub(crate) const MINI_BR_I64_EQ_RI: i64 = 208;
/// `[MINI_BR_U32_LE_RS, target, rhs_slot]` (3 words): if `u32(accum) <= u32(slot)` jump.
pub(crate) const MINI_BR_U32_LE_RS: i64 = 209;

// ── Float store from accum address ─────────────────────────────────────────
/// `[MINI_F64_STORE_RR, offset]` (2 words): `*(accum0 + offset) = freg64`.
pub(crate) const MINI_F64_STORE_RR: i64 = 210;
/// `[MINI_F32_STORE_RR, offset]` (2 words): `*(accum0 + offset) = freg32`.
pub(crate) const MINI_F32_STORE_RR: i64 = 211;

// ── Global set from accum/imm ──────────────────────────────────────────────
/// `[MINI_GLOBAL_SET_R, idx]` (2 words): `global[idx] = accum0`.
pub(crate) const MINI_GLOBAL_SET_R: i64 = 212;
/// `[MINI_GLOBAL_SET_I, idx, imm]` (3 words): `global[idx] = imm`.
pub(crate) const MINI_GLOBAL_SET_I: i64 = 213;
/// `[MINI_GLOBAL_SET_FR, idx]` (2 words): `global[idx] = accum1` (f64 bits).
pub(crate) const MINI_GLOBAL_SET_FR: i64 = 214;
/// `[MINI_GLOBAL_SET_F32R, idx]` (2 words): `global[idx] = accum2` (f32 bits).
pub(crate) const MINI_GLOBAL_SET_F32R: i64 = 215;

// ── Misc scratch-elimination ops ───────────────────────────────────────────
/// `[MINI_I64_LT_SR_R, lhs_slot]` (2 words): `accum = if slot < accum { 1 } else { 0 }`.
pub(crate) const MINI_I64_LT_SR_R: i64 = 216;
/// `[MINI_I32_SHL_RI, shift]` (2 words): `accum = wrap_i32(accum << shift)`.
pub(crate) const MINI_I32_SHL_RI: i64 = 217;
/// `[MINI_CALL_INDIRECT_SCRATCH0, table, func_type, params_start, params_len]`
/// (5 words): indirect call with index from scratch0 instead of a slot.
pub(crate) const MINI_CALL_INDIRECT_SCRATCH0: i64 = 218;
/// `[MINI_I64_LT_SCRATCH0_I_R, imm]` (2 words): `accum = if scratch0 < imm { 1 } else { 0 }`.
pub(crate) const MINI_I64_LT_SCRATCH0_I_R: i64 = 219;

// ── Integer store with scratch / accum ─────────────────────────────────────
/// `[MINI_I32_STORE8_SCRATCH0_R, offset]` (2 words): 8-bit store, ptr from
/// scratch0, value from accum.
pub(crate) const MINI_I32_STORE8_SCRATCH0_R: i64 = 220;
/// `[MINI_I32_STORE16_SCRATCH0_R, offset]` (2 words): 16-bit store, ptr from
/// scratch0, value from accum.
pub(crate) const MINI_I32_STORE16_SCRATCH0_R: i64 = 221;
/// `[MINI_I64_STORE_SCRATCH0_R, offset]` (2 words): 64-bit store, ptr from
/// scratch0, value from accum. Used when the store value must be loaded after
/// the pointer has already been moved to scratch0.
pub(crate) const MINI_I64_STORE_SCRATCH0_R: i64 = 222;
/// `[MINI_I32_STORE_SCRATCH0_R, offset]` (2 words): 32-bit store, ptr from
/// scratch0, value from accum.
pub(crate) const MINI_I32_STORE_SCRATCH0_R: i64 = 223;

// ── Comparison ops with scratch0 ───────────────────────────────────────────
/// `[MINI_U32_LT_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if u32(scratch0) < u32(slot) { 1 } else { 0 }`.
pub(crate) const MINI_U32_LT_SCRATCH0_S_R: i64 = 224;
/// `[MINI_U32_LE_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if u32(scratch0) <= u32(slot) { 1 } else { 0 }`.
pub(crate) const MINI_U32_LE_SCRATCH0_S_R: i64 = 225;
/// `[MINI_U32_LT_S_SCRATCH0_R, lhs_slot]` (2 words):
/// `accum = if u32(slot) < u32(scratch0) { 1 } else { 0 }`.
pub(crate) const MINI_U32_LT_S_SCRATCH0_R: i64 = 226;
/// `[MINI_U32_LE_S_SCRATCH0_R, lhs_slot]` (2 words):
/// `accum = if u32(slot) <= u32(scratch0) { 1 } else { 0 }`.
pub(crate) const MINI_U32_LE_S_SCRATCH0_R: i64 = 227;
/// `[MINI_I64_LE_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if scratch0 <= slot { 1 } else { 0 }`.
pub(crate) const MINI_I64_LE_SCRATCH0_S_R: i64 = 228;
/// `[MINI_U64_LT_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if u64(scratch0) < u64(slot) { 1 } else { 0 }`.
pub(crate) const MINI_U64_LT_SCRATCH0_S_R: i64 = 229;
/// `[MINI_U64_LT_S_SCRATCH0_R, lhs_slot]` (2 words):
/// `accum = if u64(slot) < u64(scratch0) { 1 } else { 0 }`.
pub(crate) const MINI_U64_LT_S_SCRATCH0_R: i64 = 230;
/// `[MINI_I32_LT_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if i32(scratch0) < i32(slot) { 1 } else { 0 }`.
pub(crate) const MINI_I32_LT_SCRATCH0_S_R: i64 = 231;

// ── Scratch0-aware compare+branch combos ───────────────────────────────────
/// `[MINI_BR_U64_LT_SCRATCH0_S, target, rhs_slot]` (3 words):
/// branch if `u64(scratch0) < u64(slot)`.
pub(crate) const MINI_BR_U64_LT_SCRATCH0_S: i64 = 232;

// ── Add with scratch operands ──────────────────────────────────────────────
/// `[MINI_I32_ADD_SCRATCH01_WB, dst_slot]` (2 words):
/// `slots[dst] = accum = wrap_i32(scratch0 + scratch1)`.
pub(crate) const MINI_I32_ADD_SCRATCH01_WB: i64 = 233;
/// `[MINI_I64_ADD_SCRATCH01_WB, dst_slot]` (2 words):
/// `slots[dst] = accum = scratch0 + scratch1`.
pub(crate) const MINI_I64_ADD_SCRATCH01_WB: i64 = 234;
/// `[MINI_I32_ADD_SCRATCH0_S_WB, dst_slot, rhs_slot]` (3 words):
/// `slots[dst] = accum = wrap_i32(scratch0 + slot)`.
pub(crate) const MINI_I32_ADD_SCRATCH0_S_WB: i64 = 235;
/// `[MINI_I64_ADD_SCRATCH0_S_WB, dst_slot, rhs_slot]` (3 words):
/// `slots[dst] = accum = scratch0 + slot`.
pub(crate) const MINI_I64_ADD_SCRATCH0_S_WB: i64 = 236;

// ── Select with scratch0 operand ───────────────────────────────────────────
/// `[MINI_SELECT_SCRATCH0_S, false_slot]` (2 words):
/// `accum = if accum != 0 { scratch0 } else { slot }`.
pub(crate) const MINI_SELECT_SCRATCH0_S: i64 = 237;
/// `[MINI_SELECT_S_SCRATCH0, true_slot]` (2 words):
/// `accum = if accum != 0 { slot } else { scratch0 }`.
pub(crate) const MINI_SELECT_S_SCRATCH0: i64 = 238;
/// `[MINI_SELECT_SCRATCH01]` (1 word):
/// `accum = if accum != 0 { scratch0 } else { scratch1 }`.
pub(crate) const MINI_SELECT_SCRATCH01: i64 = 239;

// ── Shift/rotate with scratch0 ─────────────────────────────────────────────
/// `[MINI_U64_SHR_SCRATCH01_WR]` (1 word):
/// `accum = (scratch0 as u64 >> (scratch1 as u64 & 63)) as i64`.
pub(crate) const MINI_U64_SHR_SCRATCH01_WR: i64 = 240;

// ── I32 store from accum to accum address ──────────────────────────────────
/// `[MINI_I32_STORE_RR, offset]` (2 words): `*(accum0 + offset) = accum0`.
/// For self-store patterns where pointer and value are both in accum.
/// Actually used for: ptr in SCRATCH0, value loaded via COPY_RI then stored.
/// Redefined: `[MINI_I32_STORE_SCRATCH0_I, offset, imm]` (3 words):
/// `*(scratch0 + offset) = imm`.
pub(crate) const MINI_I32_STORE_SCRATCH0_I: i64 = 241;

// ── Comparison ops: scratch0 vs scratch1 ───────────────────────────────────
/// `[MINI_U32_LT_SCRATCH01_R]` (1 word):
/// `accum = if u32(scratch0) < u32(scratch1) { 1 } else { 0 }`.
pub(crate) const MINI_U32_LT_SCRATCH01_R: i64 = 242;
/// `[MINI_U32_LE_SCRATCH01_R]` (1 word):
/// `accum = if u32(scratch0) <= u32(scratch1) { 1 } else { 0 }`.
pub(crate) const MINI_U32_LE_SCRATCH01_R: i64 = 243;
/// `[MINI_U64_LT_SCRATCH01_R]` (1 word):
/// `accum = if u64(scratch0) < u64(scratch1) { 1 } else { 0 }`.
pub(crate) const MINI_U64_LT_SCRATCH01_R: i64 = 244;

// ── Misc with scratch0 ────────────────────────────────────────────────────
/// `[MINI_I32_EQ_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if i32(scratch0) == i32(slot) { 1 } else { 0 }`.
pub(crate) const MINI_I32_EQ_SCRATCH0_S_R: i64 = 245;
/// `[MINI_I32_NE_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if i32(scratch0) != i32(slot) { 1 } else { 0 }`.
pub(crate) const MINI_I32_NE_SCRATCH0_S_R: i64 = 246;
/// `[MINI_I32_LE_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if i32(scratch0) <= i32(slot) { 1 } else { 0 }`.
pub(crate) const MINI_I32_LE_SCRATCH0_S_R: i64 = 247;
/// `[MINI_I64_EQ_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if scratch0 == slot { 1 } else { 0 }`.
pub(crate) const MINI_I64_EQ_SCRATCH0_S_R: i64 = 248;
/// `[MINI_I64_NE_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if scratch0 != slot { 1 } else { 0 }`.
pub(crate) const MINI_I64_NE_SCRATCH0_S_R: i64 = 249;
/// `[MINI_U64_LE_SCRATCH0_S_R, rhs_slot]` (2 words):
/// `accum = if u64(scratch0) <= u64(slot) { 1 } else { 0 }`.
pub(crate) const MINI_U64_LE_SCRATCH0_S_R: i64 = 251;
/// `[MINI_U32_LE_SCRATCH0_S_R_2, rhs_slot]` (2 words): duplicate alias — NOT
/// needed, reuse 225. Removed.

// ── F32/F64 arith with scratch0 (for immediate forms) ──────────────────────
/// `[MINI_F32_ARITH_SCRATCH0_S, sel, rhs_slot]` (3 words):
/// `freg32 = freg32 OP slots[rhs]` after loading imm into scratch0.
/// Actually: `freg32 = f32_arith(sel, freg32, scratch0)`.
pub(crate) const MINI_F32_ARITH_SCRATCH0: i64 = 252;
/// `[MINI_F64_ARITH_SCRATCH0, sel]` (2 words):
/// `freg64 = f64_arith(sel, freg64, scratch0)`.
pub(crate) const MINI_F64_ARITH_SCRATCH0: i64 = 253;
/// `[MINI_F32_MINMAX_SCRATCH0, sel]` (2 words):
/// `freg32 = f32_minmax(sel, freg32, scratch0)`.
pub(crate) const MINI_F32_MINMAX_SCRATCH0: i64 = 254;
/// `[MINI_F64_MINMAX_SCRATCH0, sel]` (2 words):
/// `freg64 = f64_minmax(sel, freg64, scratch0)`.
pub(crate) const MINI_F64_MINMAX_SCRATCH0: i64 = 255;

/// `[MINI_I64_EQ_SCRATCH0_R, rhs_slot]` (2 words):
/// `accum = if scratch0 == slot { 1 } else { 0 }`. Used by I64Eq_Rri.
/// NOTE: Same as MINI_I64_EQ_SCRATCH0_S_R (248). Use 248 instead.

/// `[MINI_I32_EQ_SCRATCH0_R, rhs_slot]` (2 words):
/// `accum = if i32(scratch0) == i32(slot) { 1 } else { 0 }`. Same as 245.

/// `[MINI_I32_STORE_RR_2, offset, val_slot]` — store of val at ireg+offset.
/// Actually: `[MINI_I32_STORE_SCRATCH0_S, offset, val_slot]` (3 words):
/// `*(scratch0 + offset) = slots[val_slot]`.
pub(crate) const MINI_I32_STORE_SCRATCH0_S: i64 = 256;
/// `[MINI_I64_STORE_SCRATCH0_I, offset, imm]` (3 words):
/// `*(scratch0 + offset) = imm`.
pub(crate) const MINI_I64_STORE_SCRATCH0_I: i64 = 257;

/// `[MINI_I64_AND_SCRATCH0_I_WR, imm]` (2 words):
/// `accum = scratch0 & imm`. Used by U64LoadExtend32.
pub(crate) const MINI_I64_AND_SCRATCH0_I_WR: i64 = 258;

/// `[MINI_F64_NOTGT_SCRATCH0_S_R, rhs_slot]` (2 words):
/// For F64NotLe_Rss negate pattern: `accum = !(f64_le(...))` using scratch0.
/// Actually simpler: this is `accum = (scratch0 == 0) ? 1 : 0` = `i32_eq 0`.
/// Use `MINI_I32_EQ_SCRATCH0_S_R` with a slot containing 0, or just inline.
/// Dropping this — the pattern will use COPY_SCRATCH0_I 0 + I32_EQ_RS_R.

// Highest op value used: 258

/// A function lowered to flat `i64` MiniProgram words plus the metadata the
/// kernel needs to set up its reds and merge point.
pub(crate) struct MiniProgram {
    /// Flat instruction stream (see the `MINI_*` opcodes).
    pub words: Vec<i64>,
    /// Number of dense slots (unique slot indices actually referenced); the
    /// kernel's `slots` array length. `ireg` is a separate scalar and is not
    /// counted here.
    pub num_slots: usize,
    /// Dense-index → original frame slot index. Used by the caller to seed the
    /// kernel's slots from the correct frame positions and to write back on
    /// yield. Length == `num_slots`.
    pub slot_map: Vec<u16>,
    /// Word index of the loop header (back-edge target), if the function has a
    /// back-edge; this is the trace merge point.
    pub loop_header_word: Option<usize>,
    /// Whether the function returns a value (writes a result slot). `false` for a
    /// no-result function (a bare `Return`); the caller must not write the result
    /// slot, which a 0-result frame does not reserve.
    pub writes_result: bool,
    /// Whether the function references any global (`global.get`/`global.set`). When
    /// `false` the caller skips resolving the instance's global raw pointers before
    /// a run, so a globals-free function pays nothing for the capability.
    pub uses_globals: bool,
    /// Whether the program contains any yield-to-stock, bail, or trap ops
    /// (MINI_YIELD_STOCK, MINI_RETURN_BAIL, MINI_TRAP). Functions with these
    /// ops cannot run as callees on the CALL_ASSEMBLER path because there is
    /// no stock executor to fall back to.
    pub has_yield_or_bail: bool,
}

/// Decode `ops` (an `indirect-dispatch` op stream) into a [`MiniProgram`].
///
/// Returns `None` if the function uses any op outside the supported subset, or
/// if a branch targets a non-op boundary — in either case the caller falls back
/// to the stock executor.
pub(crate) fn prepass(
    ops: &[u8],
    len_local_slots: u16,
    len_stack_slots: u16,
) -> Option<MiniProgram> {
    use crate::ir::{Decode, Slot, decode};
    use alloc::collections::BTreeMap;

    let total = ops.len();
    // Track the highest real-slot index referenced by any lowered op so the
    // kernel's `slots` Vec can be shrunk from the full `locals + stack` count
    // to cover only the actually-used range. A sentinel base replaces the
    // original `scratch_base` during emission so scratch indices can be
    // relocated in a single post-pass once `max_slot_seen` is known.
    let mut max_slot_seen: i64 = -1;
    let mut unique_slots: alloc::collections::BTreeSet<i64> = alloc::collections::BTreeSet::new();
    // Sentinel base for real-slot references. During emission every s!() call
    // emits `SLOT_SENTINEL + orig_idx` instead of the raw index. After the
    // main decode loop, a post-pass replaces every SLOT_SENTINEL occurrence
    // with its dense index (position in the sorted unique_slots set). This is
    // the same pattern as SCRATCH_SENTINEL below but for real slots.
    const SLOT_SENTINEL: i64 = i64::MIN / 4; // −2305843009213693952
    /// Convert a u16 slot operand to a sentinel-tagged i64, tracking unique set.
    macro_rules! s {
        ($v:expr) => {{
            let idx = i64::from(u16::from($v));
            if idx > max_slot_seen {
                max_slot_seen = idx;
            }
            unique_slots.insert(idx);
            SLOT_SENTINEL + idx
        }};
    }
    /// Convert a literal slot index (i64) to a sentinel-tagged value, tracking
    /// unique set. Used for opcodes with baked-in slot indices (U64Copy_S{N}r etc.).
    macro_rules! si {
        ($idx:expr) => {{
            let idx: i64 = $idx;
            if idx > max_slot_seen {
                max_slot_seen = idx;
            }
            unique_slots.insert(idx);
            SLOT_SENTINEL + idx
        }};
    }
    /// Register a contiguous range of slot indices in the unique set (for CALL
    /// ops whose params span consecutive frame slots). Returns the sentinel
    /// for the first slot; the kernel's params staging relies on the sorted
    /// BTreeSet preserving contiguous originals as contiguous dense indices.
    macro_rules! s_contig {
        ($head:expr, $len:expr) => {{
            let head_idx = i64::from(u16::from($head));
            let len = $len as i64;
            for off in 0..len {
                let idx = head_idx + off;
                if idx > max_slot_seen {
                    max_slot_seen = idx;
                }
                unique_slots.insert(idx);
            }
            SLOT_SENTINEL + head_idx
        }};
    }
    // Sentinel base for scratch slots. During emission every reference to a
    // scratch slot uses `SCRATCH_SENTINEL + offset` instead of the real index.
    // After the main decode loop, a single `words.iter_mut()` scan replaces
    // every sentinel with `compacted_scratch_base + offset`. The sentinel is
    // chosen so it cannot collide with any real opcode, slot index, branch
    // target, or plausible i64 immediate.
    const SCRATCH_SENTINEL: i64 = i64::MIN / 2; // −4611686018427387904
    let scratch_base = SCRATCH_SENTINEL;
    let mut cursor: &[u8] = ops;
    let mut words: Vec<i64> = Vec::new();
    // Maps each op's start byte offset to its emitted MiniProgram word index, so
    // byte-relative branch targets can be rewritten to word indices.
    let mut byte_to_word: BTreeMap<usize, usize> = BTreeMap::new();
    // Deferred branch-target rewrites: (target_field_word, target_byte, is_back_edge).
    let mut fixups: Vec<(usize, usize, bool)> = Vec::new();
    // Whether the caller writes the trace's result back to callee slot 0. A
    // no-result function has a zero-slot frame, where slot 0 is out of the
    // callee frame; guarding on a non-empty frame keeps the write in bounds
    // (every value-returning function reserves slot 0 for its result).
    let writes_result = i64::from(len_local_slots) + i64::from(len_stack_slots) >= 1;
    // Set true by the first lowered `global.get`/`global.set` so the caller only
    // resolves the instance's global raw pointers when the function needs them.
    let mut uses_globals = false;
    // Set true when the prepass emits any MINI_YIELD_STOCK, MINI_RETURN_BAIL,
    // or MINI_TRAP word. Functions with these ops cannot run as callees on the
    // CALL_ASSEMBLER path (no stock executor to fall back to).
    let mut has_yield_or_bail = false;

    // Integer division / remainder emission helpers, one per addressing form.
    // Each lowers onto a two-slot `$m` kernel arm (`ireg = op(slots[lhs],
    // slots[rhs])`), pre-materializing accumulator (`ireg`, via `MINI_COPY_SR`)
    // and immediate (via `MINI_COPY_SI`) operands into scratch slots while
    // preserving the non-commutative wasm `lhs, rhs` order. `$l`/`$r` are
    // already-resolved slot indices; `$imm` an already-extended i64 immediate.
    macro_rules! dr_ss {
        ($m:expr, $l:expr, $r:expr) => {
            words.extend_from_slice(&[$m, $l, $r])
        };
    }
    macro_rules! dr_rs {
        ($m:expr, $r:expr) => {
            words.extend_from_slice(&[MINI_COPY_SR, scratch_base, $m, scratch_base, $r])
        };
    }
    macro_rules! dr_sr {
        ($m:expr, $l:expr) => {
            words.extend_from_slice(&[MINI_COPY_SR, scratch_base, $m, $l, scratch_base])
        };
    }
    macro_rules! dr_si {
        ($m:expr, $l:expr, $imm:expr) => {
            words.extend_from_slice(&[MINI_COPY_SI, scratch_base, $imm, $m, $l, scratch_base])
        };
    }
    macro_rules! dr_ri {
        ($m:expr, $imm:expr) => {
            words.extend_from_slice(&[
                MINI_COPY_SR,
                scratch_base,
                MINI_COPY_SI,
                scratch_base + 1,
                $imm,
                $m,
                scratch_base,
                scratch_base + 1,
            ])
        };
    }
    macro_rules! dr_is {
        ($m:expr, $imm:expr, $r:expr) => {
            words.extend_from_slice(&[MINI_COPY_SI, scratch_base, $imm, $m, scratch_base, $r])
        };
    }
    macro_rules! dr_ir {
        ($m:expr, $imm:expr) => {
            words.extend_from_slice(&[
                MINI_COPY_SI,
                scratch_base,
                $imm,
                MINI_COPY_SR,
                scratch_base + 1,
                $m,
                scratch_base,
                scratch_base + 1,
            ])
        };
    }

    while !cursor.is_empty() {
        let pos = total - cursor.len();
        byte_to_word.insert(pos, words.len());
        let code = OpCode::decode(&mut cursor).ok()?;
        match code {
            OpCode::I32Add_Rs_si => {
                let op = decode::I32Add_Rs_si::decode(&mut cursor).ok()?;
                let dst = s!(Slot::from(op.result));
                let lhs = s!(op.lhs);
                let imm = i64::from(op.rhs);
                words.extend_from_slice(&[MINI_I32_ADD_SI_WB, dst, lhs, imm]);
            }
            OpCode::BranchI32NotEq_Ri => {
                let op = decode::BranchI32NotEq_Ri::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let imm = i64::from(op.rhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I32_NE_RI, 0, imm]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Branch if slot != immediate: load the slot into ireg and use the
            // accumulator-sourced not-equal branch.
            OpCode::BranchI32NotEq_Si => {
                let op = decode::BranchI32NotEq_Si::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = i64::from(u16::from(op.lhs));
                let imm = i64::from(op.rhs);
                words.push(MINI_COPY_RS);
                words.push(lhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I32_NE_RI, 0, imm]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Branch if ireg != slot (i32): spill ireg to scratch, then use
            // the i64 slot-sourced not-equal branch (canonical i32 = sign-extended
            // i64, so i32 NE = i64 NE).
            OpCode::BranchI32NotEq_Rs => {
                let op = decode::BranchI32NotEq_Rs::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let rhs = i64::from(u16::from(op.rhs));
                words.push(MINI_COPY_SR);
                words.push(scratch_base);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_NE_SS, 0, scratch_base, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Branch if slot == immediate (i32): spill slot to ireg, then spill
            // ireg to scratch, and use the i64 slot-sourced equality branch
            // (canonical i32 is sign-extended so the i64 comparison is correct).
            OpCode::BranchI32Eq_Si => {
                let op = decode::BranchI32Eq_Si::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = i64::from(u16::from(op.lhs));
                let imm = i64::from(op.rhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_EQ_SI, 0, lhs, imm]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Branch if ireg == immediate (i32): spill ireg to scratch and use
            // the i64 slot-sourced equality branch.
            OpCode::BranchI32Eq_Ri => {
                let op = decode::BranchI32Eq_Ri::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let imm = i64::from(op.rhs);
                words.push(MINI_COPY_SR);
                words.push(scratch_base);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_EQ_SI, 0, scratch_base, imm]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::Return => {
                // Every `return` is a bare operand-less op: the translator has
                // already copied the result (if any) into slot 0 (`Slot::from(0)`,
                // the caller's result location), so the trace returns `slots[0]`.
                // For a no-result function emit MINI_RETURN_VOID instead.
                if writes_result {
                    // Use SLOT_SENTINEL so the post-pass remaps slot 0 to
                    // its dense index. Also update max_slot_seen since we
                    // bypass the s!() macro here.
                    unique_slots.insert(0);
                    if max_slot_seen < 0 {
                        max_slot_seen = 0;
                    }
                    words.extend_from_slice(&[MINI_RETURN_S, SLOT_SENTINEL + 0]);
                } else {
                    words.push(MINI_RETURN_VOID);
                }
            }
            OpCode::U64Copy_Si => {
                let op = decode::U64Copy_Si::decode(&mut cursor).ok()?;
                let dst = s!(op.result);
                words.extend_from_slice(&[MINI_COPY_SI, dst, op.value as i64]);
            }
            OpCode::U64Copy_Ss => {
                let op = decode::U64Copy_Ss::decode(&mut cursor).ok()?;
                let dst = s!(op.result);
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_COPY_SS, dst, src]);
            }
            // Local-indexed accumulator spills (`U64Copy_S{N}r`): the slot index
            // N is baked into the opcode and the operands occupy zero bytes.
            OpCode::U64Copy_S0r => {
                words.push(MINI_COPY_SR);
                words.push(si!(0));
            }
            OpCode::U64Copy_S1r => {
                words.push(MINI_COPY_SR);
                words.push(si!(1));
            }
            OpCode::U64Copy_S2r => {
                words.push(MINI_COPY_SR);
                words.push(si!(2));
            }
            OpCode::U64Copy_S3r => {
                words.push(MINI_COPY_SR);
                words.push(si!(3));
            }
            OpCode::U64Copy_S4r => {
                words.push(MINI_COPY_SR);
                words.push(si!(4));
            }
            OpCode::U64Copy_S5r => {
                words.push(MINI_COPY_SR);
                words.push(si!(5));
            }
            OpCode::U64Copy_S6r => {
                words.push(MINI_COPY_SR);
                words.push(si!(6));
            }
            OpCode::U64Copy_S7r => {
                words.push(MINI_COPY_SR);
                words.push(si!(7));
            }
            OpCode::U64Copy_S8r => {
                words.push(MINI_COPY_SR);
                words.push(si!(8));
            }
            OpCode::U64Copy_S9r => {
                words.push(MINI_COPY_SR);
                words.push(si!(9));
            }
            // Local-to-local copies (`U64Copy_S{N}s{M}`): both the destination
            // slot N and source slot M are baked into the opcode, so the operands
            // occupy zero bytes. Bit-identical slot move.
            OpCode::U64Copy_S0s1 => {
                words.push(MINI_COPY_SS);
                words.push(si!(0));
                words.push(si!(1));
            }
            OpCode::U64Copy_S0s2 => {
                words.push(MINI_COPY_SS);
                words.push(si!(0));
                words.push(si!(2));
            }
            OpCode::U64Copy_S0s3 => {
                words.push(MINI_COPY_SS);
                words.push(si!(0));
                words.push(si!(3));
            }
            OpCode::U64Copy_S0s4 => {
                words.push(MINI_COPY_SS);
                words.push(si!(0));
                words.push(si!(4));
            }
            OpCode::U64Copy_S0s5 => {
                words.push(MINI_COPY_SS);
                words.push(si!(0));
                words.push(si!(5));
            }
            OpCode::U64Copy_S1s0 => {
                words.push(MINI_COPY_SS);
                words.push(si!(1));
                words.push(si!(0));
            }
            OpCode::U64Copy_S1s2 => {
                words.push(MINI_COPY_SS);
                words.push(si!(1));
                words.push(si!(2));
            }
            OpCode::U64Copy_S1s3 => {
                words.push(MINI_COPY_SS);
                words.push(si!(1));
                words.push(si!(3));
            }
            OpCode::U64Copy_S1s4 => {
                words.push(MINI_COPY_SS);
                words.push(si!(1));
                words.push(si!(4));
            }
            OpCode::U64Copy_S1s5 => {
                words.push(MINI_COPY_SS);
                words.push(si!(1));
                words.push(si!(5));
            }
            OpCode::U64Copy_S2s0 => {
                words.push(MINI_COPY_SS);
                words.push(si!(2));
                words.push(si!(0));
            }
            OpCode::U64Copy_S2s1 => {
                words.push(MINI_COPY_SS);
                words.push(si!(2));
                words.push(si!(1));
            }
            OpCode::U64Copy_S2s3 => {
                words.push(MINI_COPY_SS);
                words.push(si!(2));
                words.push(si!(3));
            }
            OpCode::U64Copy_S2s4 => {
                words.push(MINI_COPY_SS);
                words.push(si!(2));
                words.push(si!(4));
            }
            OpCode::U64Copy_S2s5 => {
                words.push(MINI_COPY_SS);
                words.push(si!(2));
                words.push(si!(5));
            }
            OpCode::U64Copy_S3s0 => {
                words.push(MINI_COPY_SS);
                words.push(si!(3));
                words.push(si!(0));
            }
            OpCode::U64Copy_S3s1 => {
                words.push(MINI_COPY_SS);
                words.push(si!(3));
                words.push(si!(1));
            }
            OpCode::U64Copy_S3s2 => {
                words.push(MINI_COPY_SS);
                words.push(si!(3));
                words.push(si!(2));
            }
            OpCode::U64Copy_S3s4 => {
                words.push(MINI_COPY_SS);
                words.push(si!(3));
                words.push(si!(4));
            }
            OpCode::U64Copy_S3s5 => {
                words.push(MINI_COPY_SS);
                words.push(si!(3));
                words.push(si!(5));
            }
            OpCode::U64Copy_S4s0 => {
                words.push(MINI_COPY_SS);
                words.push(si!(4));
                words.push(si!(0));
            }
            OpCode::U64Copy_S4s1 => {
                words.push(MINI_COPY_SS);
                words.push(si!(4));
                words.push(si!(1));
            }
            OpCode::U64Copy_S4s2 => {
                words.push(MINI_COPY_SS);
                words.push(si!(4));
                words.push(si!(2));
            }
            OpCode::U64Copy_S4s3 => {
                words.push(MINI_COPY_SS);
                words.push(si!(4));
                words.push(si!(3));
            }
            OpCode::U64Copy_S4s5 => {
                words.push(MINI_COPY_SS);
                words.push(si!(4));
                words.push(si!(5));
            }
            OpCode::U64Copy_S5s0 => {
                words.push(MINI_COPY_SS);
                words.push(si!(5));
                words.push(si!(0));
            }
            OpCode::U64Copy_S5s1 => {
                words.push(MINI_COPY_SS);
                words.push(si!(5));
                words.push(si!(1));
            }
            OpCode::U64Copy_S5s2 => {
                words.push(MINI_COPY_SS);
                words.push(si!(5));
                words.push(si!(2));
            }
            OpCode::U64Copy_S5s3 => {
                words.push(MINI_COPY_SS);
                words.push(si!(5));
                words.push(si!(3));
            }
            OpCode::U64Copy_S5s4 => {
                words.push(MINI_COPY_SS);
                words.push(si!(5));
                words.push(si!(4));
            }
            // f64 accumulator spills read the f64 accumulator (`freg64`), not the
            // integer one, so they lower to `MINI_COPY_S_FR` (bit-identical move).
            OpCode::F64Copy_S0r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(0));
            }
            OpCode::F64Copy_S1r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(1));
            }
            OpCode::F64Copy_S2r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(2));
            }
            OpCode::F64Copy_S3r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(3));
            }
            OpCode::F64Copy_S4r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(4));
            }
            OpCode::F64Copy_S5r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(5));
            }
            OpCode::F64Copy_S6r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(6));
            }
            OpCode::F64Copy_S7r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(7));
            }
            OpCode::F64Copy_S8r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(8));
            }
            OpCode::F64Copy_S9r => {
                words.push(MINI_COPY_S_FR);
                words.push(si!(9));
            }
            // Generic accumulator spills with an explicit destination slot (used
            // when the slot index exceeds the dedicated `S{0..9}r` opcodes).
            OpCode::U64Copy_Sr => {
                let op = decode::U64Copy_Sr::decode(&mut cursor).ok()?;
                let dst = s!(op.result);
                words.extend_from_slice(&[MINI_COPY_SR, dst]);
            }
            // Slot-to-accumulator and immediate-to-accumulator copies.
            OpCode::U64Copy_Rs => {
                let op = decode::U64Copy_Rs::decode(&mut cursor).ok()?;
                let src = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_COPY_RS, src]);
            }
            OpCode::U64Copy_Ri => {
                let op = decode::U64Copy_Ri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_COPY_RI, op.value as i64]);
            }
            // i32 copy forms: sign-extend to canonical i32 representation.
            OpCode::U32Copy_Si => {
                let op = decode::U32Copy_Si::decode(&mut cursor).ok()?;
                let dst = i64::from(u16::from(op.result));
                let imm = i64::from(op.value as i32);
                words.extend_from_slice(&[MINI_COPY_SI, dst, imm]);
            }
            OpCode::U32Copy_Ri => {
                let op = decode::U32Copy_Ri::decode(&mut cursor).ok()?;
                let imm = i64::from(op.value as i32);
                words.extend_from_slice(&[MINI_COPY_RI, imm]);
            }
            OpCode::F64Copy_Sr => {
                let op = decode::F64Copy_Sr::decode(&mut cursor).ok()?;
                let dst = s!(op.result);
                words.extend_from_slice(&[MINI_COPY_S_FR, dst]);
            }
            // f32 accumulator spills read the f32 accumulator (`freg32`), so they
            // lower to `MINI_COPY_S_F32R` (a 32-bit move; the low 32 hold the f32).
            OpCode::F32Copy_S0r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(0));
            }
            OpCode::F32Copy_S1r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(1));
            }
            OpCode::F32Copy_S2r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(2));
            }
            OpCode::F32Copy_S3r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(3));
            }
            OpCode::F32Copy_S4r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(4));
            }
            OpCode::F32Copy_S5r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(5));
            }
            OpCode::F32Copy_S6r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(6));
            }
            OpCode::F32Copy_S7r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(7));
            }
            OpCode::F32Copy_S8r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(8));
            }
            OpCode::F32Copy_S9r => {
                words.push(MINI_COPY_S_F32R);
                words.push(si!(9));
            }
            OpCode::F32Copy_Sr => {
                let op = decode::F32Copy_Sr::decode(&mut cursor).ok()?;
                let dst = s!(op.result);
                words.extend_from_slice(&[MINI_COPY_S_F32R, dst]);
            }
            OpCode::I64Add_Rss => {
                let op = decode::I64Add_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_ADD_SS_WR, lhs, rhs]);
            }
            OpCode::I64Mul_Rss => {
                let op = decode::I64Mul_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_MUL_SS_WR, lhs, rhs]);
            }
            OpCode::I64BitOr_Rss => {
                let op = decode::I64BitOr_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_OR_SS_WR, lhs, rhs]);
            }
            OpCode::I64Sub_Rss => {
                let op = decode::I64Sub_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_SUB_SS_WR, lhs, rhs]);
            }
            OpCode::I32BitXor_Rss => {
                let op = decode::I32BitXor_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_XOR_SS_WR, lhs, rhs]);
            }
            // The `Rrs` (accumulator OP slot) i32 forms use the dedicated
            // accumulator-slot ops — no scratch copy needed.
            OpCode::I32BitAnd_Rrs => {
                let op = decode::I32BitAnd_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_AND_RS_WR, rhs]);
            }
            OpCode::I32BitOr_Rrs => {
                let op = decode::I32BitOr_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_OR_RS_WR, rhs]);
            }
            OpCode::I32Sub_Rrs => {
                let op = decode::I32Sub_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_SUB_RS_WR, rhs]);
            }
            // Remaining value-position `sub` forms (result in the accumulator).
            // Sub is non-commutative, so accumulator (`ireg`, via `MINI_COPY_SR`)
            // and immediate (via `MINI_COPY_SI`) operands are pre-materialized into
            // scratch slots preserving the `lhs - rhs` order, then the two-slot
            // sub MINI op computes into `ireg`.
            OpCode::I32Sub_Rss => {
                let op = decode::I32Sub_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_SUB_SS_WR, lhs, rhs]);
            }
            OpCode::I32Sub_Rsr => {
                let op = decode::I32Sub_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_I32_SUB_SR_WR, lhs]);
            }
            OpCode::I32Sub_Ris => {
                let op = decode::I32Sub_Ris::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    i64::from(op.lhs),
                    MINI_I32_SUB_SS_WR,
                    scratch_base,
                    rhs,
                ]);
            }
            OpCode::I32Sub_Rir => {
                let op = decode::I32Sub_Rir::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    i64::from(op.lhs),
                    MINI_I32_SUB_RS_WR,
                    scratch_base,
                ]);
            }
            OpCode::I64Sub_Rrs => {
                let op = decode::I64Sub_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_SUB_RS_WR, rhs]);
            }
            OpCode::I64Sub_Rsr => {
                let op = decode::I64Sub_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_I64_SUB_SR_WR, lhs]);
            }
            OpCode::I64Sub_Ris => {
                let op = decode::I64Sub_Ris::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    op.lhs,
                    MINI_I64_SUB_SS_WR,
                    scratch_base,
                    rhs,
                ]);
            }
            OpCode::I64Sub_Rir => {
                let op = decode::I64Sub_Rir::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    op.lhs,
                    MINI_I64_SUB_RS_WR,
                    scratch_base,
                ]);
            }
            OpCode::BranchI32Le_Ss => {
                let op = decode::BranchI32Le_Ss::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I32_LE_SS, 0, lhs, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchI64Eq_Ss => {
                let op = decode::BranchI64Eq_Ss::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_EQ_SS, 0, lhs, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I32Mul_Rss => {
                let op = decode::I32Mul_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_MUL_SS_WR, lhs, rhs]);
            }
            // i32 multiply / xor / or / and against a folded constant (`acc OP imm`).
            // Save the accumulator to scratch, load the immediate into the accum,
            // then use the accumulator-slot op (1 scratch slot instead of 2).
            OpCode::I32Mul_Rri => {
                let op = decode::I32Mul_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    i64::from(op.rhs),
                    MINI_I32_MUL_RS_WR,
                    scratch_base,
                ]);
            }
            OpCode::I32BitXor_Rri => {
                let op = decode::I32BitXor_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    i64::from(op.rhs),
                    MINI_I32_XOR_RS_WR,
                    scratch_base,
                ]);
            }
            OpCode::I32BitOr_Rri => {
                let op = decode::I32BitOr_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    i64::from(op.rhs),
                    MINI_I32_OR_RS_WR,
                    scratch_base,
                ]);
            }
            OpCode::I32BitAnd_Rri => {
                let op = decode::I32BitAnd_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    i64::from(op.rhs),
                    MINI_I32_AND_RS_WR,
                    scratch_base,
                ]);
            }
            // i32 multiply / xor against a slot operand (`acc OP slots[rhs]`).
            // Use the dedicated accumulator-slot ops — no scratch copy needed.
            OpCode::I32Mul_Rrs => {
                let op = decode::I32Mul_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_MUL_RS_WR, rhs]);
            }
            OpCode::I32BitXor_Rrs => {
                let op = decode::I32BitXor_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_XOR_RS_WR, rhs]);
            }
            OpCode::I32Add_Rs_rs => {
                let op = decode::I32Add_Rs_rs::decode(&mut cursor).ok()?;
                let dst = s!(Slot::from(op.result));
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_ADD_RS_WB, dst, rhs]);
            }
            // i32 add with accumulator + immediate: pre-materialize the immediate
            // into a scratch slot, then use the accumulator+slot add.
            OpCode::I32Add_Rs_ri => {
                let op = decode::I32Add_Rs_ri::decode(&mut cursor).ok()?;
                let dst = i64::from(u16::from(Slot::from(op.result)));
                let imm = i64::from(op.rhs);
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_I32_ADD_RS_WB,
                    dst,
                    scratch_base,
                ]);
            }
            OpCode::BranchU32Le_Ss => {
                let op = decode::BranchU32Le_Ss::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_U32_LE_SS, 0, lhs, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Branch if ireg <=_u slot: spill ireg to scratch and use the
            // slot-slot unsigned LE branch.
            OpCode::BranchU32Le_Rs => {
                let op = decode::BranchU32Le_Rs::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let rhs = i64::from(u16::from(op.rhs));
                words.push(MINI_COPY_SR);
                words.push(scratch_base);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_U32_LE_SS, 0, scratch_base, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Branch if imm <_u slot (unsigned i64): materialize the immediate
            // into a scratch slot and use a slot-slot unsigned LT branch.
            OpCode::BranchU64Lt_Is => {
                let op = decode::BranchU64Lt_Is::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let imm = op.lhs as i64;
                let rhs = i64::from(u16::from(op.rhs));
                words.extend_from_slice(&[MINI_COPY_SI, scratch_base, imm]);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_U64_LT_SS, 0, scratch_base, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::Branch => {
                let op = decode::Branch::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_ALWAYS, 0]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchI64Le_Ss => {
                let op = decode::BranchI64Le_Ss::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_LE_SS, 0, lhs, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I64BitXor_Rss => {
                let op = decode::I64BitXor_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_XOR_SS_WR, lhs, rhs]);
            }
            OpCode::I64BitAnd_Rri => {
                let op = decode::I64BitAnd_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_I64_AND_RI_WR, op.rhs]);
            }
            OpCode::I64BitAnd_Rsi => {
                let op = decode::I64BitAnd_Rsi::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_I64_AND_SI_WR, lhs, op.rhs]);
            }
            // i64 multiply / or / xor against a folded constant (`acc OP imm`).
            // Save the accumulator to scratch, load the immediate into the accum,
            // then use the accumulator-slot op (1 scratch slot instead of 2).
            OpCode::I64Mul_Rri => {
                let op = decode::I64Mul_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    op.rhs,
                    MINI_I64_MUL_RS_WR,
                    scratch_base,
                ]);
            }
            OpCode::I64BitOr_Rri => {
                let op = decode::I64BitOr_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    op.rhs,
                    MINI_I64_OR_RS_WR,
                    scratch_base,
                ]);
            }
            OpCode::I64BitXor_Rri => {
                let op = decode::I64BitXor_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    op.rhs,
                    MINI_I64_XOR_RS_WR,
                    scratch_base,
                ]);
            }
            // i64 multiply / or / xor against a slot operand (`acc OP slots[rhs]`).
            // Use the dedicated accumulator-slot ops — no scratch copy needed.
            OpCode::I64Mul_Rrs => {
                let op = decode::I64Mul_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_MUL_RS_WR, rhs]);
            }
            OpCode::I64BitOr_Rrs => {
                let op = decode::I64BitOr_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_OR_RS_WR, rhs]);
            }
            OpCode::I64BitXor_Rrs => {
                let op = decode::I64BitXor_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_XOR_RS_WR, rhs]);
            }
            OpCode::I64Add_Rs_rs => {
                let op = decode::I64Add_Rs_rs::decode(&mut cursor).ok()?;
                let dst = s!(Slot::from(op.result));
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_ADD_RS_WB, dst, rhs]);
            }
            OpCode::I64Shl_Rsi => {
                let op = decode::I64Shl_Rsi::decode(&mut cursor).ok()?;
                let src = s!(op.lhs);
                let shift = i64::from(u32::from(u8::from(op.rhs)) & 63);
                words.extend_from_slice(&[MINI_I64_SHL_SI, src, shift]);
            }
            OpCode::I32Shl_Rsi => {
                let op = decode::I32Shl_Rsi::decode(&mut cursor).ok()?;
                let src = s!(op.lhs);
                // i32 shift amount is taken mod 32.
                let shift = i64::from(u32::from(u8::from(op.rhs)) & 31);
                words.extend_from_slice(&[MINI_I32_SHL_SI, src, shift]);
            }
            OpCode::I32Rotl_Rsi => {
                let op = decode::I32Rotl_Rsi::decode(&mut cursor).ok()?;
                let src = s!(op.lhs);
                // i32 rotate amount is taken mod 32.
                let k = u32::from(u8::from(op.rhs)) & 31;
                if k == 0 {
                    // Rotate by zero is the identity; reuse the i32 shift-by-zero
                    // arm rather than emit a rotate that would shift by 32.
                    words.extend_from_slice(&[MINI_I32_SHL_SI, src, 0]);
                } else {
                    words.extend_from_slice(&[MINI_I32_ROTL_SI, src, i64::from(k)]);
                }
            }
            OpCode::I32Rotr_Rsi => {
                let op = decode::I32Rotr_Rsi::decode(&mut cursor).ok()?;
                let src = s!(op.lhs);
                let k = u32::from(u8::from(op.rhs)) & 31;
                if k == 0 {
                    words.extend_from_slice(&[MINI_I32_SHL_SI, src, 0]);
                } else {
                    words.extend_from_slice(&[MINI_I32_ROTR_SI, src, i64::from(k)]);
                }
            }
            OpCode::U32Shr_Rri => {
                let op = decode::U32Shr_Rri::decode(&mut cursor).ok()?;
                let shift = i64::from(u32::from(u8::from(op.rhs)) & 31);
                words.extend_from_slice(&[MINI_U32_SHR_RI, shift]);
            }
            // Integer bit-count unary ops (clz / ctz / popcnt, never trap). The
            // slot-input form (`_Rs`) maps directly; the accumulator-input form
            // (`_Rr`) reads the integer accumulator, so it copies `ireg`
            // (`MINI_COPY_SR`) into a scratch slot first (the copy folds away).
            OpCode::I32Clz_Rs => {
                let op = decode::I32Clz_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I32_BITCOUNT_S, 0, src]);
            }
            OpCode::I32Clz_Rr => {
                let _op = decode::I32Clz_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I32_BITCOUNT_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::I32Ctz_Rs => {
                let op = decode::I32Ctz_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I32_BITCOUNT_S, 1, src]);
            }
            OpCode::I32Ctz_Rr => {
                let _op = decode::I32Ctz_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I32_BITCOUNT_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::I32Popcnt_Rs => {
                let op = decode::I32Popcnt_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I32_BITCOUNT_S, 2, src]);
            }
            OpCode::I32Popcnt_Rr => {
                let _op = decode::I32Popcnt_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I32_BITCOUNT_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::I64Clz_Rs => {
                let op = decode::I64Clz_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I64_BITCOUNT_S, 0, src]);
            }
            OpCode::I64Clz_Rr => {
                let _op = decode::I64Clz_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I64_BITCOUNT_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::I64Ctz_Rs => {
                let op = decode::I64Ctz_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I64_BITCOUNT_S, 1, src]);
            }
            OpCode::I64Ctz_Rr => {
                let _op = decode::I64Ctz_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I64_BITCOUNT_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::I64Popcnt_Rs => {
                let op = decode::I64Popcnt_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I64_BITCOUNT_S, 2, src]);
            }
            OpCode::I64Popcnt_Rr => {
                let _op = decode::I64Popcnt_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I64_BITCOUNT_S,
                    2,
                    scratch_base,
                ]);
            }
            // Integer division / remainder, all seven addressing forms per op.
            // The immediate divisor (`_Rsi`/`_Rri`) is a `NonZero`; the immediate
            // dividend (`_Ris`/`_Rir`) an ordinary integer. Both are extended into
            // an i64 word (the residual re-narrows to the op's width).
            OpCode::I32Div_Rss => {
                let op = decode::I32Div_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I32_DIV_S, s!(op.lhs), s!(op.rhs));
            }
            OpCode::I32Div_Rrs => {
                let op = decode::I32Div_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I32_DIV_S, s!(op.rhs));
            }
            OpCode::I32Div_Rsr => {
                let op = decode::I32Div_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I32_DIV_S, s!(op.lhs));
            }
            OpCode::I32Div_Rsi => {
                let op = decode::I32Div_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I32_DIV_S, s!(op.lhs), i64::from(op.rhs.get()));
            }
            OpCode::I32Div_Rri => {
                let op = decode::I32Div_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I32_DIV_S, i64::from(op.rhs.get()));
            }
            OpCode::I32Div_Ris => {
                let op = decode::I32Div_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I32_DIV_S, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::I32Div_Rir => {
                let op = decode::I32Div_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I32_DIV_S, i64::from(op.lhs));
            }
            OpCode::U32Div_Rss => {
                let op = decode::U32Div_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I32_DIV_U, s!(op.lhs), s!(op.rhs));
            }
            OpCode::U32Div_Rrs => {
                let op = decode::U32Div_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I32_DIV_U, s!(op.rhs));
            }
            OpCode::U32Div_Rsr => {
                let op = decode::U32Div_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I32_DIV_U, s!(op.lhs));
            }
            OpCode::U32Div_Rsi => {
                let op = decode::U32Div_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I32_DIV_U, s!(op.lhs), i64::from(op.rhs.get()));
            }
            OpCode::U32Div_Rri => {
                let op = decode::U32Div_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I32_DIV_U, i64::from(op.rhs.get()));
            }
            OpCode::U32Div_Ris => {
                let op = decode::U32Div_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I32_DIV_U, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::U32Div_Rir => {
                let op = decode::U32Div_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I32_DIV_U, i64::from(op.lhs));
            }
            OpCode::I32Rem_Rss => {
                let op = decode::I32Rem_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I32_REM_S, s!(op.lhs), s!(op.rhs));
            }
            OpCode::I32Rem_Rrs => {
                let op = decode::I32Rem_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I32_REM_S, s!(op.rhs));
            }
            OpCode::I32Rem_Rsr => {
                let op = decode::I32Rem_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I32_REM_S, s!(op.lhs));
            }
            OpCode::I32Rem_Rsi => {
                let op = decode::I32Rem_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I32_REM_S, s!(op.lhs), i64::from(op.rhs.get()));
            }
            OpCode::I32Rem_Rri => {
                let op = decode::I32Rem_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I32_REM_S, i64::from(op.rhs.get()));
            }
            OpCode::I32Rem_Ris => {
                let op = decode::I32Rem_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I32_REM_S, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::I32Rem_Rir => {
                let op = decode::I32Rem_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I32_REM_S, i64::from(op.lhs));
            }
            OpCode::U32Rem_Rss => {
                let op = decode::U32Rem_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I32_REM_U, s!(op.lhs), s!(op.rhs));
            }
            OpCode::U32Rem_Rrs => {
                let op = decode::U32Rem_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I32_REM_U, s!(op.rhs));
            }
            OpCode::U32Rem_Rsr => {
                let op = decode::U32Rem_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I32_REM_U, s!(op.lhs));
            }
            OpCode::U32Rem_Rsi => {
                let op = decode::U32Rem_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I32_REM_U, s!(op.lhs), i64::from(op.rhs.get()));
            }
            OpCode::U32Rem_Rri => {
                let op = decode::U32Rem_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I32_REM_U, i64::from(op.rhs.get()));
            }
            OpCode::U32Rem_Ris => {
                let op = decode::U32Rem_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I32_REM_U, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::U32Rem_Rir => {
                let op = decode::U32Rem_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I32_REM_U, i64::from(op.lhs));
            }
            OpCode::I64Div_Rss => {
                let op = decode::I64Div_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I64_DIV_S, s!(op.lhs), s!(op.rhs));
            }
            OpCode::I64Div_Rrs => {
                let op = decode::I64Div_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I64_DIV_S, s!(op.rhs));
            }
            OpCode::I64Div_Rsr => {
                let op = decode::I64Div_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I64_DIV_S, s!(op.lhs));
            }
            OpCode::I64Div_Rsi => {
                let op = decode::I64Div_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I64_DIV_S, s!(op.lhs), op.rhs.get());
            }
            OpCode::I64Div_Rri => {
                let op = decode::I64Div_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I64_DIV_S, op.rhs.get());
            }
            OpCode::I64Div_Ris => {
                let op = decode::I64Div_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I64_DIV_S, op.lhs, s!(op.rhs));
            }
            OpCode::I64Div_Rir => {
                let op = decode::I64Div_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I64_DIV_S, op.lhs);
            }
            OpCode::U64Div_Rss => {
                let op = decode::U64Div_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I64_DIV_U, s!(op.lhs), s!(op.rhs));
            }
            OpCode::U64Div_Rrs => {
                let op = decode::U64Div_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I64_DIV_U, s!(op.rhs));
            }
            OpCode::U64Div_Rsr => {
                let op = decode::U64Div_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I64_DIV_U, s!(op.lhs));
            }
            OpCode::U64Div_Rsi => {
                let op = decode::U64Div_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I64_DIV_U, s!(op.lhs), op.rhs.get() as i64);
            }
            OpCode::U64Div_Rri => {
                let op = decode::U64Div_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I64_DIV_U, op.rhs.get() as i64);
            }
            OpCode::U64Div_Ris => {
                let op = decode::U64Div_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I64_DIV_U, op.lhs as i64, s!(op.rhs));
            }
            OpCode::U64Div_Rir => {
                let op = decode::U64Div_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I64_DIV_U, op.lhs as i64);
            }
            OpCode::I64Rem_Rss => {
                let op = decode::I64Rem_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I64_REM_S, s!(op.lhs), s!(op.rhs));
            }
            OpCode::I64Rem_Rrs => {
                let op = decode::I64Rem_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I64_REM_S, s!(op.rhs));
            }
            OpCode::I64Rem_Rsr => {
                let op = decode::I64Rem_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I64_REM_S, s!(op.lhs));
            }
            OpCode::I64Rem_Rsi => {
                let op = decode::I64Rem_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I64_REM_S, s!(op.lhs), op.rhs.get());
            }
            OpCode::I64Rem_Rri => {
                let op = decode::I64Rem_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I64_REM_S, op.rhs.get());
            }
            OpCode::I64Rem_Ris => {
                let op = decode::I64Rem_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I64_REM_S, op.lhs, s!(op.rhs));
            }
            OpCode::I64Rem_Rir => {
                let op = decode::I64Rem_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I64_REM_S, op.lhs);
            }
            OpCode::U64Rem_Rss => {
                let op = decode::U64Rem_Rss::decode(&mut cursor).ok()?;
                dr_ss!(MINI_I64_REM_U, s!(op.lhs), s!(op.rhs));
            }
            OpCode::U64Rem_Rrs => {
                let op = decode::U64Rem_Rrs::decode(&mut cursor).ok()?;
                dr_rs!(MINI_I64_REM_U, s!(op.rhs));
            }
            OpCode::U64Rem_Rsr => {
                let op = decode::U64Rem_Rsr::decode(&mut cursor).ok()?;
                dr_sr!(MINI_I64_REM_U, s!(op.lhs));
            }
            OpCode::U64Rem_Rsi => {
                let op = decode::U64Rem_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I64_REM_U, s!(op.lhs), op.rhs.get() as i64);
            }
            OpCode::U64Rem_Rri => {
                let op = decode::U64Rem_Rri::decode(&mut cursor).ok()?;
                dr_ri!(MINI_I64_REM_U, op.rhs.get() as i64);
            }
            OpCode::U64Rem_Ris => {
                let op = decode::U64Rem_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I64_REM_U, op.lhs as i64, s!(op.rhs));
            }
            OpCode::U64Rem_Rir => {
                let op = decode::U64Rem_Rir::decode(&mut cursor).ok()?;
                dr_ir!(MINI_I64_REM_U, op.lhs as i64);
            }
            OpCode::I32Lt_Rsi => {
                let op = decode::I32Lt_Rsi::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                // i32 immediate, sign-extended into the i64 word.
                let imm = i64::from(op.rhs);
                words.extend_from_slice(&[MINI_I32_LT_SI_R, lhs, imm]);
            }
            OpCode::I64Lt_Ris => {
                let op = decode::I64Lt_Ris::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_LT_IS_R, op.lhs, rhs]);
            }
            OpCode::I64Lt_Rsi => {
                let op = decode::I64Lt_Rsi::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_I64_LT_SI_R, s!(op.lhs), op.rhs]);
            }
            // i64 compare-as-value with accumulator < immediate: spill ireg to
            // scratch, then use the slot-sourced lt.
            OpCode::I64Lt_Rri => {
                let op = decode::I64Lt_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I64_LT_SI_R,
                    scratch_base,
                    op.rhs,
                ]);
            }
            // i64 compare-as-value with accumulator < immediate: spill ireg to
            // scratch, then use the slot-sourced lt.
            OpCode::I64Lt_Rri => {
                let op = decode::I64Lt_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I64_LT_SI_R,
                    scratch_base,
                    op.rhs,
                ]);
            }
            OpCode::I32Eq_Rsi => {
                let op = decode::I32Eq_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I32_EQ_SS_R, s!(op.lhs), i64::from(op.rhs));
            }
            OpCode::I32NotEq_Rsi => {
                let op = decode::I32NotEq_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I32_NE_SS_R, s!(op.lhs), i64::from(op.rhs));
            }
            OpCode::I32Le_Rsi => {
                let op = decode::I32Le_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I32_LE_SS_R, s!(op.lhs), i64::from(op.rhs));
            }
            OpCode::U32Lt_Rsi => {
                let op = decode::U32Lt_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_U32_LT_SS_R, s!(op.lhs), i64::from(op.rhs));
            }
            OpCode::U32Le_Rsi => {
                let op = decode::U32Le_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_U32_LE_SS_R, s!(op.lhs), i64::from(op.rhs));
            }
            OpCode::I32Lt_Ris => {
                let op = decode::I32Lt_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I32_LT_SS_R, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::I32Le_Ris => {
                let op = decode::I32Le_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I32_LE_SS_R, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::U32Lt_Ris => {
                let op = decode::U32Lt_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_U32_LT_SS_R, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::U32Le_Ris => {
                let op = decode::U32Le_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_U32_LE_SS_R, i64::from(op.lhs), s!(op.rhs));
            }
            OpCode::I64Eq_Rsi => {
                let op = decode::I64Eq_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I64_EQ_SS_R, s!(op.lhs), op.rhs);
            }
            OpCode::I64NotEq_Rsi => {
                let op = decode::I64NotEq_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I64_NE_SS_R, s!(op.lhs), op.rhs);
            }
            OpCode::I64Le_Rsi => {
                let op = decode::I64Le_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_I64_LE_SS_R, s!(op.lhs), op.rhs);
            }
            OpCode::U64Lt_Rsi => {
                let op = decode::U64Lt_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_U64_LT_SS_R, s!(op.lhs), op.rhs as i64);
            }
            OpCode::U64Le_Rsi => {
                let op = decode::U64Le_Rsi::decode(&mut cursor).ok()?;
                dr_si!(MINI_U64_LE_SS_R, s!(op.lhs), op.rhs as i64);
            }
            OpCode::I64Le_Ris => {
                let op = decode::I64Le_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_I64_LE_SS_R, op.lhs, s!(op.rhs));
            }
            OpCode::U64Lt_Ris => {
                let op = decode::U64Lt_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_U64_LT_SS_R, op.lhs as i64, s!(op.rhs));
            }
            OpCode::U64Le_Ris => {
                let op = decode::U64Le_Ris::decode(&mut cursor).ok()?;
                dr_is!(MINI_U64_LE_SS_R, op.lhs as i64, s!(op.rhs));
            }
            OpCode::I32Lt_Rrs => {
                // i32.lt_s with the left operand in the accumulator, right in a slot.
                let op = decode::I32Lt_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_LT_RS_R, rhs]);
            }
            OpCode::I32Lt_Rsr => {
                // i32.lt_s with the left operand in a slot, right in the accumulator.
                let op = decode::I32Lt_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_I32_LT_SR_R, lhs]);
            }
            OpCode::U64Select_Rrss => {
                // select with the condition in the accumulator, both arms in slots.
                let op = decode::U64Select_Rrss::decode(&mut cursor).ok()?;
                let true_slot = s!(op.true_val);
                let false_slot = s!(op.false_val);
                words.extend_from_slice(&[MINI_SELECT, true_slot, false_slot]);
            }
            OpCode::U32Select_Rrsi => {
                // select with the condition in the accumulator, true arm in a slot, false arm a constant.
                let op = decode::U32Select_Rrsi::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    i64::from(op.false_val),
                    MINI_SELECT,
                    s!(op.true_val),
                    scratch_base,
                ]);
            }
            OpCode::U32Select_Rris => {
                // select with the condition in the accumulator, true arm a constant, false arm in a slot.
                let op = decode::U32Select_Rris::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    i64::from(op.true_val),
                    MINI_SELECT,
                    scratch_base,
                    s!(op.false_val),
                ]);
            }
            OpCode::U32Select_Rrii => {
                // select with the condition in the accumulator, both arms constants.
                let op = decode::U32Select_Rrii::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    i64::from(op.true_val),
                    MINI_COPY_SI,
                    scratch_base + 1,
                    i64::from(op.false_val),
                    MINI_SELECT,
                    scratch_base,
                    scratch_base + 1,
                ]);
            }
            OpCode::U64Select_Rrsi => {
                // select with the condition in the accumulator, true arm in a slot, false arm a constant.
                let op = decode::U64Select_Rrsi::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    op.false_val as i64,
                    MINI_SELECT,
                    s!(op.true_val),
                    scratch_base,
                ]);
            }
            OpCode::U64Select_Rris => {
                // select with the condition in the accumulator, true arm a constant, false arm in a slot.
                let op = decode::U64Select_Rris::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    op.true_val as i64,
                    MINI_SELECT,
                    scratch_base,
                    s!(op.false_val),
                ]);
            }
            OpCode::U64Select_Rrii => {
                // select with the condition in the accumulator, both arms constants.
                let op = decode::U64Select_Rrii::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    op.true_val as i64,
                    MINI_COPY_SI,
                    scratch_base + 1,
                    op.false_val as i64,
                    MINI_SELECT,
                    scratch_base,
                    scratch_base + 1,
                ]);
            }
            OpCode::I32Eq_Rrs => {
                // i32.eq with the left operand in the accumulator, right in a slot.
                let op = decode::I32Eq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_EQ_RS_R, rhs]);
            }
            OpCode::I32NotEq_Rrs => {
                // i32.ne with the left operand in the accumulator, right in a slot.
                let op = decode::I32NotEq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_NE_RS_R, rhs]);
            }
            OpCode::I64Eq_Rrs => {
                // i64.eq with the left operand in the accumulator, right in a slot.
                let op = decode::I64Eq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_EQ_RS_R, rhs]);
            }
            OpCode::I64NotEq_Rrs => {
                // i64.ne with the left operand in the accumulator, right in a slot.
                let op = decode::I64NotEq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_NE_RS_R, rhs]);
            }
            OpCode::I64Lt_Rrs => {
                // i64.lt_s with the left operand in the accumulator, right in a slot.
                let op = decode::I64Lt_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_LT_RS_R, rhs]);
            }
            OpCode::I32Le_Rrs => {
                // i32.le_s with the left operand in the accumulator, right in a slot.
                let op = decode::I32Le_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_LE_RS_R, rhs]);
            }
            OpCode::U32Lt_Rrs => {
                // i32.lt_u (unsigned) with the left operand in the accumulator.
                let op = decode::U32Lt_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_U32_LT_RS_R, rhs]);
            }
            OpCode::U32Le_Rrs => {
                // i32.le_u (unsigned) with the left operand in the accumulator.
                let op = decode::U32Le_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_U32_LE_RS_R, rhs]);
            }
            OpCode::I32Mul_Rsi => {
                // slot * imm -> reg, pre-materialized into a uniform slot*slot mul.
                let op = decode::I32Mul_Rsi::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let imm = i64::from(op.rhs);
                words.extend_from_slice(&[MINI_COPY_SI, scratch_base, imm]);
                words.extend_from_slice(&[MINI_I32_MUL_SS_WR, lhs, scratch_base]);
            }
            OpCode::I64Mul_Rsi => {
                // i64 slot * imm -> reg (immediate is a full i64), same recipe.
                let op = decode::I64Mul_Rsi::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_COPY_SI, scratch_base, op.rhs]);
                words.extend_from_slice(&[MINI_I64_MUL_SS_WR, lhs, scratch_base]);
            }
            OpCode::I32Add_Rrs => {
                let op = decode::I32Add_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_ADD_RS_WR, rhs]);
            }
            // Value-position `x + imm` (result in the accumulator, not a local):
            // materialize the immediate into a scratch slot, then reuse the
            // slot-and-reg add with its slot write routed to a throwaway scratch.
            // Add is commutative, so `slot + imm` and `imm + slot` are the same op.
            OpCode::I32Add_Rsi => {
                let op = decode::I32Add_Rsi::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let imm = i64::from(op.rhs);
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base + 1,
                    imm,
                    MINI_I32_ADD_SS_WB,
                    scratch_base,
                    lhs,
                    scratch_base + 1,
                ]);
            }
            OpCode::I32Add_Rri => {
                let op = decode::I32Add_Rri::decode(&mut cursor).ok()?;
                let imm = i64::from(op.rhs);
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base + 1,
                    imm,
                    MINI_I32_ADD_RS_WB,
                    scratch_base,
                    scratch_base + 1,
                ]);
            }
            OpCode::I64Add_Rsi => {
                let op = decode::I64Add_Rsi::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base + 1,
                    op.rhs,
                    MINI_I64_ADD_SS_WB,
                    scratch_base,
                    lhs,
                    scratch_base + 1,
                ]);
            }
            OpCode::I64Add_Rri => {
                let op = decode::I64Add_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base + 1,
                    op.rhs,
                    MINI_I64_ADD_RS_WB,
                    scratch_base,
                    scratch_base + 1,
                ]);
            }
            OpCode::I64Add_Rrs => {
                let op = decode::I64Add_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_ADD_RS_WR, rhs]);
            }
            OpCode::U64LoadMem0Offset16_Rr => {
                let op = decode::U64LoadMem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I64_LOAD_MEM0_OFF, offset]);
            }
            OpCode::F64LoadMem0Offset16_Rr => {
                // An f64 load is a bit-identical 8-byte read whose result lands in
                // the f64 accumulator (`freg64`); the address is the integer
                // accumulator. The bytes go through the same bounds-checked
                // residual as the i64 load.
                let op = decode::F64LoadMem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_F64_LOAD_MEM0_OFF, offset]);
            }
            // f64 load with pointer from a slot: copy slot to ireg, then load.
            OpCode::F64LoadMem0Offset16_Rs => {
                let op = decode::F64LoadMem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_F64_LOAD_MEM0_OFF, offset]);
            }
            OpCode::F64Add_Rsr => {
                // f64 add: left operand in a slot, right in the accumulator. Add is
                // commutative, so `slot + acc` maps to the `acc OP slot` selector arm.
                let op = decode::F64Add_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F64_ARITH_RS, 0, lhs]);
            }
            OpCode::F64Add_Rrs => {
                // f64 add: left operand in the accumulator, right in a slot (the
                // other operand order the translator emits; add is commutative).
                let op = decode::F64Add_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_ARITH_RS, 0, rhs]);
            }
            OpCode::F64Sub_Rrs => {
                // f64 sub: left operand in the accumulator, right in a slot.
                let op = decode::F64Sub_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_ARITH_RS, 1, rhs]);
            }
            OpCode::F64Mul_Rrs => {
                let op = decode::F64Mul_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_ARITH_RS, 2, rhs]);
            }
            OpCode::F64Div_Rrs => {
                // f64 div: left operand in the accumulator, right in a slot.
                let op = decode::F64Div_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_ARITH_RS, 3, rhs]);
            }
            // f32 memory + arithmetic, mirroring the f64 forms but routed to the
            // f32 accumulator (`freg32`). Loads/stores reuse the 4-byte i32
            // residuals; arithmetic goes through the `f32_*` bit-pattern residuals.
            OpCode::F32LoadMem0Offset16_Rr => {
                let op = decode::F32LoadMem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_F32_LOAD_MEM0_OFF, offset]);
            }
            // f32 load with pointer from a slot: copy slot to ireg, then load.
            OpCode::F32LoadMem0Offset16_Rs => {
                let op = decode::F32LoadMem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_F32_LOAD_MEM0_OFF, offset]);
            }
            OpCode::F32StoreMem0Offset16_Sr => {
                let op = decode::F32StoreMem0Offset16_Sr::decode(&mut cursor).ok()?;
                let ptr = s!(op.ptr);
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_F32_STORE_SR, ptr, offset]);
            }
            // f64 store with pointer in ireg, value in freg64: save ptr to scratch.
            OpCode::F64StoreMem0Offset16_Rr => {
                let op = decode::F64StoreMem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F64_STORE_SR,
                    scratch_base,
                    offset,
                ]);
            }
            // f32 store with pointer in ireg, value in freg32: save ptr to scratch.
            OpCode::F32StoreMem0Offset16_Rr => {
                let op = decode::F32StoreMem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F32_STORE_SR,
                    scratch_base,
                    offset,
                ]);
            }
            OpCode::F32Add_Rsr => {
                // f32 add: left operand in a slot, right in the accumulator. Add is
                // commutative, so `slot + acc` maps to the `acc OP slot` selector arm.
                let op = decode::F32Add_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F32_ARITH_RS, 0, lhs]);
            }
            OpCode::F32Add_Rrs => {
                // f32 add: left operand in the accumulator, right in a slot (the
                // other operand order the translator emits; add is commutative).
                let op = decode::F32Add_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_ARITH_RS, 0, rhs]);
            }
            OpCode::F32Sub_Rrs => {
                // f32 sub: left operand in the accumulator, right in a slot.
                let op = decode::F32Sub_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_ARITH_RS, 1, rhs]);
            }
            OpCode::F32Mul_Rrs => {
                let op = decode::F32Mul_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_ARITH_RS, 2, rhs]);
            }
            OpCode::F32Div_Rrs => {
                // f32 div: left operand in the accumulator, right in a slot.
                let op = decode::F32Div_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_ARITH_RS, 3, rhs]);
            }
            OpCode::F32Min_Rrs => {
                let op = decode::F32Min_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_MINMAX_RS, 0, rhs]);
            }
            OpCode::F32Min_Rsr => {
                // min is commutative, so the slot-then-accumulator form reuses the
                // `acc OP slot` selector arm with the slot as the right operand.
                let op = decode::F32Min_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F32_MINMAX_RS, 0, lhs]);
            }
            OpCode::F32Max_Rrs => {
                let op = decode::F32Max_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_MINMAX_RS, 1, rhs]);
            }
            OpCode::F32Max_Rsr => {
                let op = decode::F32Max_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F32_MINMAX_RS, 1, lhs]);
            }
            OpCode::F32Copysign_Rrs => {
                // f32.copysign(magnitude=accumulator, sign=slot): `sel` 2 keeps the
                // accumulator as the magnitude operand.
                let op = decode::F32Copysign_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_MINMAX_RS, 2, rhs]);
            }
            OpCode::F32Copysign_Rsr => {
                // f32.copysign(magnitude=slot, sign=accumulator): copysign is not
                // commutative, so `sel` 3 swaps operands to take the magnitude from
                // the slot and the sign from the accumulator.
                let op = decode::F32Copysign_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F32_MINMAX_RS, 3, lhs]);
            }
            // f32 arithmetic / min / max against a folded f32 constant (`_Rri`):
            // wasmi fuses the `f32.const` operand as an inline 4-byte immediate.
            // Pre-materialize the constant's bit pattern into a scratch slot, then
            // reuse the slot-form selector arm (the copy folds away in the trace).
            OpCode::F32Add_Rri => {
                let op = decode::F32Add_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F32_ARITH_RS,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F32Sub_Rri => {
                let op = decode::F32Sub_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F32_ARITH_RS,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F32Mul_Rri => {
                let op = decode::F32Mul_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F32_ARITH_RS,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::F32Div_Rri => {
                let op = decode::F32Div_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F32_ARITH_RS,
                    3,
                    scratch_base,
                ]);
            }
            OpCode::F32Min_Rri => {
                let op = decode::F32Min_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F32_MINMAX_RS,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F32Max_Rri => {
                let op = decode::F32Max_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F32_MINMAX_RS,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F32Lt_Rrs => {
                let op = decode::F32Lt_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_CMP_RS_R, 0, rhs]);
            }
            OpCode::F32Le_Rrs => {
                let op = decode::F32Le_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_CMP_RS_R, 1, rhs]);
            }
            // f32 < with the left operand in a slot, right in the f32 accumulator.
            // gt/ge arrive as swapped lt/le forms.
            OpCode::F32Lt_Rsr => {
                let op = decode::F32Lt_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F32_CMP_RS_R, 4, lhs]);
            }
            OpCode::F32Le_Rsr => {
                let op = decode::F32Le_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F32_CMP_RS_R, 5, lhs]);
            }
            OpCode::F32Eq_Rrs => {
                let op = decode::F32Eq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_CMP_RS_R, 2, rhs]);
            }
            OpCode::F32NotEq_Rrs => {
                let op = decode::F32NotEq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F32_CMP_RS_R, 3, rhs]);
            }
            // f32 unary ops (`freg32`). The slot-input form (`_Rs`) maps directly;
            // the accumulator-input form (`_Rr`) copies `freg32` into a scratch slot
            // first via `MINI_COPY_S_F32R` (the copy folds in the trace). sel: 0=abs,
            // 1=neg, 2=sqrt, 3=ceil, 4=floor, 5=trunc, 6=nearest.
            OpCode::F32Abs_Rs => {
                let op = decode::F32Abs_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_UNARY_S, 0, src]);
            }
            OpCode::F32Abs_Rr => {
                let _op = decode::F32Abs_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_UNARY_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F32Neg_Rs => {
                let op = decode::F32Neg_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_UNARY_S, 1, src]);
            }
            OpCode::F32Neg_Rr => {
                let _op = decode::F32Neg_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_UNARY_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F32Sqrt_Rs => {
                let op = decode::F32Sqrt_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_UNARY_S, 2, src]);
            }
            OpCode::F32Sqrt_Rr => {
                let _op = decode::F32Sqrt_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_UNARY_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::F32Ceil_Rs => {
                let op = decode::F32Ceil_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_UNARY_S, 3, src]);
            }
            OpCode::F32Ceil_Rr => {
                let _op = decode::F32Ceil_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_UNARY_S,
                    3,
                    scratch_base,
                ]);
            }
            OpCode::F32Floor_Rs => {
                let op = decode::F32Floor_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_UNARY_S, 4, src]);
            }
            OpCode::F32Floor_Rr => {
                let _op = decode::F32Floor_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_UNARY_S,
                    4,
                    scratch_base,
                ]);
            }
            OpCode::F32Trunc_Rs => {
                let op = decode::F32Trunc_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_UNARY_S, 5, src]);
            }
            OpCode::F32Trunc_Rr => {
                let _op = decode::F32Trunc_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_UNARY_S,
                    5,
                    scratch_base,
                ]);
            }
            OpCode::F32Nearest_Rs => {
                let op = decode::F32Nearest_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_UNARY_S, 6, src]);
            }
            OpCode::F32Nearest_Rr => {
                let _op = decode::F32Nearest_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_UNARY_S,
                    6,
                    scratch_base,
                ]);
            }
            // Integer→f32 conversions (never trap). Slot-input (`_Rs`) maps directly;
            // accumulator-input (`_Rr`) copies the INTEGER accumulator (`ireg`) into a
            // scratch slot via `MINI_COPY_SR`. sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
            OpCode::F32ConvertI32_Rs => {
                let op = decode::F32ConvertI32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_CVT_S, 0, src]);
            }
            OpCode::F32ConvertI32_Rr => {
                let _op = decode::F32ConvertI32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F32_CVT_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F32ConvertU32_Rs => {
                let op = decode::F32ConvertU32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_CVT_S, 1, src]);
            }
            OpCode::F32ConvertU32_Rr => {
                let _op = decode::F32ConvertU32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F32_CVT_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F32ConvertI64_Rs => {
                let op = decode::F32ConvertI64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_CVT_S, 2, src]);
            }
            OpCode::F32ConvertI64_Rr => {
                let _op = decode::F32ConvertI64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F32_CVT_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::F32ConvertU64_Rs => {
                let op = decode::F32ConvertU64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_CVT_S, 3, src]);
            }
            OpCode::F32ConvertU64_Rr => {
                let _op = decode::F32ConvertU64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F32_CVT_S,
                    3,
                    scratch_base,
                ]);
            }
            // f32↔f64 promote / demote. `_Rr` copies the source float accumulator
            // into a scratch slot first (freg32 via `MINI_COPY_S_F32R` for promote,
            // freg64 via `MINI_COPY_S_FR` for demote).
            OpCode::F64PromoteF32_Rs => {
                let op = decode::F64PromoteF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_PROMOTE_S, src]);
            }
            OpCode::F64PromoteF32_Rr => {
                let _op = decode::F64PromoteF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F64_PROMOTE_S,
                    scratch_base,
                ]);
            }
            OpCode::F32DemoteF64_Rs => {
                let op = decode::F32DemoteF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_DEMOTE_S, src]);
            }
            OpCode::F32DemoteF64_Rr => {
                let _op = decode::F32DemoteF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F32_DEMOTE_S,
                    scratch_base,
                ]);
            }
            // Bit-reinterpret between i32 and f32 (accumulator forms only). Pure bit
            // moves between `ireg` and `freg32`; no residual, no operand word.
            OpCode::I32ReinterpretF32_Rr => {
                let _op = decode::I32ReinterpretF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_I32_REINTERP_F32]);
            }
            OpCode::F32ReinterpretI32_Rr => {
                let _op = decode::F32ReinterpretI32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_F32_REINTERP_I32]);
            }
            // Bit-reinterpret between i64 and f64 (accumulator forms only). Pure bit
            // moves between `ireg` and `freg64`; no residual, no operand word.
            OpCode::I64ReinterpretF64_Rr => {
                let _op = decode::I64ReinterpretF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_I64_REINTERP_F64]);
            }
            OpCode::F64ReinterpretI64_Rr => {
                let _op = decode::F64ReinterpretI64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_F64_REINTERP_I64]);
            }
            // Saturating f32→integer truncations (never trap). Slot-input (`_Rs`)
            // maps directly; accumulator-input (`_Rr`) copies the f32 accumulator
            // (`freg32`) into a scratch slot via `MINI_COPY_S_F32R`. sel: 0=i32_s,
            // 1=u32, 2=i64_s, 3=u64.
            OpCode::I32TruncSatF32_Rs => {
                let op = decode::I32TruncSatF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_SAT_S, 0, src]);
            }
            OpCode::I32TruncSatF32_Rr => {
                let _op = decode::I32TruncSatF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_SAT_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::U32TruncSatF32_Rs => {
                let op = decode::U32TruncSatF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_SAT_S, 1, src]);
            }
            OpCode::U32TruncSatF32_Rr => {
                let _op = decode::U32TruncSatF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_SAT_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::I64TruncSatF32_Rs => {
                let op = decode::I64TruncSatF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_SAT_S, 2, src]);
            }
            OpCode::I64TruncSatF32_Rr => {
                let _op = decode::I64TruncSatF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_SAT_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::U64TruncSatF32_Rs => {
                let op = decode::U64TruncSatF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_SAT_S, 3, src]);
            }
            OpCode::U64TruncSatF32_Rr => {
                let _op = decode::U64TruncSatF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_SAT_S,
                    3,
                    scratch_base,
                ]);
            }
            // Trapping f32→integer truncations (NaN / out-of-range trap).
            OpCode::I32TruncF32_Rs => {
                let op = decode::I32TruncF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_S, 0, src]);
            }
            OpCode::I32TruncF32_Rr => {
                let _op = decode::I32TruncF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::U32TruncF32_Rs => {
                let op = decode::U32TruncF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_S, 1, src]);
            }
            OpCode::U32TruncF32_Rr => {
                let _op = decode::U32TruncF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::I64TruncF32_Rs => {
                let op = decode::I64TruncF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_S, 2, src]);
            }
            OpCode::I64TruncF32_Rr => {
                let _op = decode::I64TruncF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::U64TruncF32_Rs => {
                let op = decode::U64TruncF32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F32_TRUNC_S, 3, src]);
            }
            OpCode::U64TruncF32_Rr => {
                let _op = decode::U64TruncF32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_F32_TRUNC_S,
                    3,
                    scratch_base,
                ]);
            }
            OpCode::F64Lt_Rrs => {
                let op = decode::F64Lt_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_CMP_RS_R, 0, rhs]);
            }
            OpCode::F64Le_Rrs => {
                let op = decode::F64Le_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_CMP_RS_R, 1, rhs]);
            }
            // f64 < with the left operand in a slot, right in the f64 accumulator.
            // gt/ge arrive as swapped lt/le forms.
            OpCode::F64Lt_Rsr => {
                let op = decode::F64Lt_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F64_CMP_RS_R, 4, lhs]);
            }
            OpCode::F64Le_Rsr => {
                let op = decode::F64Le_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F64_CMP_RS_R, 5, lhs]);
            }
            OpCode::F64Eq_Rrs => {
                let op = decode::F64Eq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_CMP_RS_R, 2, rhs]);
            }
            OpCode::F64NotEq_Rrs => {
                let op = decode::F64NotEq_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_CMP_RS_R, 3, rhs]);
            }
            // f64 binops against a folded f64 constant: wasmi fuses the constant as
            // an inline 8-byte immediate (`_Rri`). Pre-materialize its bit pattern
            // into a scratch slot, then reuse the existing slot-form f64 op (the
            // copy folds away in the compiled trace).
            OpCode::F64Add_Rri => {
                let op = decode::F64Add_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F64_ARITH_RS,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F64Sub_Rri => {
                let op = decode::F64Sub_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F64_ARITH_RS,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F64Mul_Rri => {
                let op = decode::F64Mul_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F64_ARITH_RS,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::F64Div_Rri => {
                let op = decode::F64Div_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F64_ARITH_RS,
                    3,
                    scratch_base,
                ]);
            }
            // f64 unary ops. The slot-input form (`_Rs`) maps directly onto the
            // slot-form MINI op; the accumulator-input form (`_Rr`) reads the f64
            // accumulator, so it copies `freg64` (`MINI_COPY_S_FR`) into a scratch
            // slot first (the copy folds in the trace).
            OpCode::F64Abs_Rs => {
                let op = decode::F64Abs_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_UNARY_S, 0, src]);
            }
            OpCode::F64Abs_Rr => {
                let _op = decode::F64Abs_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_UNARY_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F64Neg_Rs => {
                let op = decode::F64Neg_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_UNARY_S, 1, src]);
            }
            OpCode::F64Neg_Rr => {
                let _op = decode::F64Neg_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_UNARY_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F64Sqrt_Rs => {
                let op = decode::F64Sqrt_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_UNARY_S, 2, src]);
            }
            OpCode::F64Sqrt_Rr => {
                let _op = decode::F64Sqrt_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_UNARY_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::F64Ceil_Rs => {
                let op = decode::F64Ceil_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_UNARY_S, 3, src]);
            }
            OpCode::F64Ceil_Rr => {
                let _op = decode::F64Ceil_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_UNARY_S,
                    3,
                    scratch_base,
                ]);
            }
            OpCode::F64Floor_Rs => {
                let op = decode::F64Floor_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_UNARY_S, 4, src]);
            }
            OpCode::F64Floor_Rr => {
                let _op = decode::F64Floor_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_UNARY_S,
                    4,
                    scratch_base,
                ]);
            }
            OpCode::F64Trunc_Rs => {
                let op = decode::F64Trunc_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_UNARY_S, 5, src]);
            }
            OpCode::F64Trunc_Rr => {
                let _op = decode::F64Trunc_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_UNARY_S,
                    5,
                    scratch_base,
                ]);
            }
            OpCode::F64Nearest_Rs => {
                let op = decode::F64Nearest_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_UNARY_S, 6, src]);
            }
            OpCode::F64Nearest_Rr => {
                let _op = decode::F64Nearest_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_UNARY_S,
                    6,
                    scratch_base,
                ]);
            }
            // f64 min/max. Commutative, so the accumulator-vs-slot (`_Rrs`) and
            // slot-vs-accumulator (`_Rsr`) forms both map onto the slot-form arm,
            // and the constant form (`_Rri`) pre-materializes the immediate.
            OpCode::F64Min_Rrs => {
                let op = decode::F64Min_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_MINMAX_RS, 0, rhs]);
            }
            OpCode::F64Min_Rsr => {
                let op = decode::F64Min_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F64_MINMAX_RS, 0, lhs]);
            }
            OpCode::F64Min_Rri => {
                let op = decode::F64Min_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F64_MINMAX_RS,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F64Max_Rrs => {
                let op = decode::F64Max_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_MINMAX_RS, 1, rhs]);
            }
            OpCode::F64Max_Rsr => {
                let op = decode::F64Max_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F64_MINMAX_RS, 1, lhs]);
            }
            OpCode::F64Max_Rri => {
                let op = decode::F64Max_Rri::decode(&mut cursor).ok()?;
                let imm = op.rhs.to_bits() as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_F64_MINMAX_RS,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F64Copysign_Rrs => {
                // f64.copysign(magnitude=accumulator, sign=slot): `sel` 2 keeps the
                // accumulator as the magnitude operand.
                let op = decode::F64Copysign_Rrs::decode(&mut cursor).ok()?;
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_F64_MINMAX_RS, 2, rhs]);
            }
            OpCode::F64Copysign_Rsr => {
                // f64.copysign(magnitude=slot, sign=accumulator): copysign is not
                // commutative, so `sel` 3 swaps operands to take the magnitude from
                // the slot and the sign from the accumulator.
                let op = decode::F64Copysign_Rsr::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                words.extend_from_slice(&[MINI_F64_MINMAX_RS, 3, lhs]);
            }
            // Widening integer-to-f64 conversions (never trap). The slot-input
            // form (`_Rs`) maps directly; the accumulator-input form (`_Rr`) copies
            // the accumulator into a scratch slot first.
            // form (`_Rs`) maps directly; the accumulator-input form (`_Rr`) copies
            // the INTEGER accumulator (`ireg`) into a scratch slot via `MINI_COPY_SR`.
            // sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
            OpCode::F64ConvertI32_Rs => {
                let op = decode::F64ConvertI32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_CVT_S, 0, src]);
            }
            OpCode::F64ConvertI32_Rr => {
                let _op = decode::F64ConvertI32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F64_CVT_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::F64ConvertU32_Rs => {
                let op = decode::F64ConvertU32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_CVT_S, 1, src]);
            }
            OpCode::F64ConvertU32_Rr => {
                let _op = decode::F64ConvertU32_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F64_CVT_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::F64ConvertI64_Rs => {
                let op = decode::F64ConvertI64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_CVT_S, 2, src]);
            }
            OpCode::F64ConvertI64_Rr => {
                let _op = decode::F64ConvertI64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F64_CVT_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::F64ConvertU64_Rs => {
                let op = decode::F64ConvertU64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_CVT_S, 3, src]);
            }
            OpCode::F64ConvertU64_Rr => {
                let _op = decode::F64ConvertU64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_F64_CVT_S,
                    3,
                    scratch_base,
                ]);
            }
            // Saturating f64→integer truncations (never trap). Slot-input (`_Rs`)
            // maps directly; accumulator-input (`_Rr`) copies the f64 accumulator
            // (`freg64`) into a scratch slot via `MINI_COPY_S_FR`. sel: 0=i32_s,
            // 1=u32, 2=i64_s, 3=u64.
            OpCode::I32TruncSatF64_Rs => {
                let op = decode::I32TruncSatF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_SAT_S, 0, src]);
            }
            OpCode::I32TruncSatF64_Rr => {
                let _op = decode::I32TruncSatF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_SAT_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::U32TruncSatF64_Rs => {
                let op = decode::U32TruncSatF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_SAT_S, 1, src]);
            }
            OpCode::U32TruncSatF64_Rr => {
                let _op = decode::U32TruncSatF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_SAT_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::I64TruncSatF64_Rs => {
                let op = decode::I64TruncSatF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_SAT_S, 2, src]);
            }
            OpCode::I64TruncSatF64_Rr => {
                let _op = decode::I64TruncSatF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_SAT_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::U64TruncSatF64_Rs => {
                let op = decode::U64TruncSatF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_SAT_S, 3, src]);
            }
            OpCode::U64TruncSatF64_Rr => {
                let _op = decode::U64TruncSatF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_SAT_S,
                    3,
                    scratch_base,
                ]);
            }
            // Trapping f64→integer truncations (NaN / out-of-range trap). Same
            // `_Rs` direct / `_Rr` scratch-copy lowering as the saturating forms;
            // the residual flags the trap and `run_jit` surfaces it.
            OpCode::I32TruncF64_Rs => {
                let op = decode::I32TruncF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_S, 0, src]);
            }
            OpCode::I32TruncF64_Rr => {
                let _op = decode::I32TruncF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_S,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::U32TruncF64_Rs => {
                let op = decode::U32TruncF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_S, 1, src]);
            }
            OpCode::U32TruncF64_Rr => {
                let _op = decode::U32TruncF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_S,
                    1,
                    scratch_base,
                ]);
            }
            OpCode::I64TruncF64_Rs => {
                let op = decode::I64TruncF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_S, 2, src]);
            }
            OpCode::I64TruncF64_Rr => {
                let _op = decode::I64TruncF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_S,
                    2,
                    scratch_base,
                ]);
            }
            OpCode::U64TruncF64_Rs => {
                let op = decode::U64TruncF64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_F64_TRUNC_S, 3, src]);
            }
            OpCode::U64TruncF64_Rr => {
                let _op = decode::U64TruncF64_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_F64_TRUNC_S,
                    3,
                    scratch_base,
                ]);
            }
            OpCode::U32LoadMem0Offset16_Rr => {
                let op = decode::U32LoadMem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I32_LOAD_MEM0_OFF, offset]);
            }
            OpCode::U32LoadExtend8Mem0Offset16_Rr => {
                let op = decode::U32LoadExtend8Mem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_U8_LOAD_MEM0_OFF, offset]);
            }
            OpCode::I32LoadExtend8Mem0Offset16_Rr => {
                let op = decode::I32LoadExtend8Mem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I8_LOAD_MEM0_OFF, offset]);
            }
            OpCode::U32LoadExtend16Mem0Offset16_Rr => {
                let op = decode::U32LoadExtend16Mem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_U16_LOAD_MEM0_OFF, offset]);
            }
            OpCode::I32LoadExtend16Mem0Offset16_Rr => {
                let op = decode::I32LoadExtend16Mem0Offset16_Rr::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I16_LOAD_MEM0_OFF, offset]);
            }
            // Sub-word loads with pointer from a slot: copy slot to ireg first.
            OpCode::U32LoadExtend8Mem0Offset16_Rs => {
                let op = decode::U32LoadExtend8Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_U8_LOAD_MEM0_OFF, offset]);
            }
            OpCode::I32LoadExtend8Mem0Offset16_Rs => {
                let op = decode::I32LoadExtend8Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_I8_LOAD_MEM0_OFF, offset]);
            }
            OpCode::U32LoadExtend16Mem0Offset16_Rs => {
                let op = decode::U32LoadExtend16Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_U16_LOAD_MEM0_OFF, offset]);
            }
            OpCode::I32LoadExtend16Mem0Offset16_Rs => {
                let op = decode::I32LoadExtend16Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_I16_LOAD_MEM0_OFF, offset]);
            }
            // Generic load with pre-computed absolute address (not Mem0Offset16).
            // Only memory 0 is supported by the kernel; bail otherwise.
            OpCode::U32Load_Ri => {
                let op = decode::U32Load_Ri::decode(&mut cursor).ok()?;
                if !op.memory.is_default() {
                    return None;
                }
                let addr = u64::from(op.address) as i64;
                words.extend_from_slice(&[MINI_COPY_RI, addr, MINI_I32_LOAD_MEM0_OFF, 0]);
            }
            OpCode::U64Load_Ri => {
                let op = decode::U64Load_Ri::decode(&mut cursor).ok()?;
                if !op.memory.is_default() {
                    return None;
                }
                let addr = u64::from(op.address) as i64;
                words.extend_from_slice(&[MINI_COPY_RI, addr, MINI_I64_LOAD_MEM0_OFF, 0]);
            }
            OpCode::U32StoreMem0Offset16_Sr => {
                // i32.store: pointer in a slot, value in the accumulator.
                let op = decode::U32StoreMem0Offset16_Sr::decode(&mut cursor).ok()?;
                let ptr = s!(op.ptr);
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I32_STORE_SR, ptr, offset]);
            }
            OpCode::U64StoreMem0Offset16_Sr => {
                // i64.store of a computed value: pointer in a slot, value in acc.
                let op = decode::U64StoreMem0Offset16_Sr::decode(&mut cursor).ok()?;
                let ptr = s!(op.ptr);
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I64_STORE_SR, ptr, offset]);
            }
            OpCode::F64StoreMem0Offset16_Sr => {
                // f64.store of a computed value: pointer in a slot, value in the
                // f64 accumulator (`freg64`) — a bit-identical 8-byte store.
                let op = decode::F64StoreMem0Offset16_Sr::decode(&mut cursor).ok()?;
                let ptr = s!(op.ptr);
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_F64_STORE_SR, ptr, offset]);
            }
            OpCode::I32StoreWrap8Mem0Offset16_Sr => {
                // i32.store8 of a computed value: pointer in a slot, value in acc.
                let op = decode::I32StoreWrap8Mem0Offset16_Sr::decode(&mut cursor).ok()?;
                let ptr = s!(op.ptr);
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I32_STORE8_SR, ptr, offset]);
            }
            OpCode::I32StoreWrap16Mem0Offset16_Sr => {
                // i32.store16 of a computed value: pointer in a slot, value in acc.
                let op = decode::I32StoreWrap16Mem0Offset16_Sr::decode(&mut cursor).ok()?;
                let ptr = s!(op.ptr);
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_I32_STORE16_SR, ptr, offset]);
            }
            OpCode::U32StoreMem0Offset16_Rs => {
                // i32.store of a local: pointer in the accumulator, value in a slot.
                let op = decode::U32StoreMem0Offset16_Rs::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let val = s!(op.value);
                words.extend_from_slice(&[MINI_I32_STORE_RS, offset, val]);
            }
            OpCode::U64StoreMem0Offset16_Rs => {
                // i64.store of a local: pointer in the accumulator, value in a slot.
                let op = decode::U64StoreMem0Offset16_Rs::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let val = s!(op.value);
                words.extend_from_slice(&[MINI_I64_STORE_RS, offset, val]);
            }
            OpCode::I32StoreWrap8Mem0Offset16_Rs => {
                // i32.store8: pointer in the accumulator, value (low byte) in a slot.
                let op = decode::I32StoreWrap8Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let val = s!(op.value);
                words.extend_from_slice(&[MINI_I32_STORE8_RS, offset, val]);
            }
            OpCode::I32StoreWrap16Mem0Offset16_Rs => {
                // i32.store16: pointer in the accumulator, value (low half) in a slot.
                let op = decode::I32StoreWrap16Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let val = s!(op.value);
                words.extend_from_slice(&[MINI_I32_STORE16_RS, offset, val]);
            }
            // i64.store32: truncate a 64-bit value to 32 bits and store. The
            // existing 4-byte store residuals handle this identically to i32.store.
            OpCode::I64StoreWrap32Mem0Offset16_Rs => {
                let op = decode::I64StoreWrap32Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let val = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_I32_STORE_RS, offset, val]);
            }
            // --- Narrow store with immediate value (ptr=slot, val=immediate) ---
            OpCode::I32StoreWrap8Mem0Offset16_Si => {
                let op = decode::I32StoreWrap8Mem0Offset16_Si::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let imm = i64::from(op.value as i8);
                words.extend_from_slice(&[MINI_COPY_RI, imm, MINI_I32_STORE8_SR, ptr, offset]);
            }
            OpCode::I32StoreWrap16Mem0Offset16_Si => {
                let op = decode::I32StoreWrap16Mem0Offset16_Si::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let imm = i64::from(op.value as i16);
                words.extend_from_slice(&[MINI_COPY_RI, imm, MINI_I32_STORE16_SR, ptr, offset]);
            }
            // --- Narrow store with both operands in slots ---
            OpCode::I32StoreWrap8Mem0Offset16_Ss => {
                let op = decode::I32StoreWrap8Mem0Offset16_Ss::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let val = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_COPY_RS, val, MINI_I32_STORE8_SR, ptr, offset]);
            }
            OpCode::I32StoreWrap16Mem0Offset16_Ss => {
                let op = decode::I32StoreWrap16Mem0Offset16_Ss::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let val = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_COPY_RS, val, MINI_I32_STORE16_SR, ptr, offset]);
            }
            // --- Narrow store with ptr in accumulator, val immediate ---
            OpCode::I32StoreWrap8Mem0Offset16_Ri => {
                let op = decode::I32StoreWrap8Mem0Offset16_Ri::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let imm = i64::from(op.value as i8);
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    imm,
                    MINI_I32_STORE8_SR,
                    scratch_base,
                    offset,
                ]);
            }
            OpCode::I32StoreWrap16Mem0Offset16_Ri => {
                let op = decode::I32StoreWrap16Mem0Offset16_Ri::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let imm = i64::from(op.value as i16);
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    imm,
                    MINI_I32_STORE16_SR,
                    scratch_base,
                    offset,
                ]);
            }
            // --- Store with immediate value (ptr=slot, val=immediate) ---
            // Decompose: load imm into ireg, then use the existing _SR store.
            OpCode::U64StoreMem0Offset16_Si => {
                let op = decode::U64StoreMem0Offset16_Si::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let imm = op.value as i64;
                words.extend_from_slice(&[MINI_COPY_RI, imm, MINI_I64_STORE_SR, ptr, offset]);
            }
            OpCode::U32StoreMem0Offset16_Si => {
                let op = decode::U32StoreMem0Offset16_Si::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let imm = i64::from(op.value);
                words.extend_from_slice(&[MINI_COPY_RI, imm, MINI_I32_STORE_SR, ptr, offset]);
            }
            // --- Store with both operands in slots (ptr=slot, val=slot) ---
            // Decompose: load val slot into ireg, then use the existing _SR store.
            OpCode::U64StoreMem0Offset16_Ss => {
                let op = decode::U64StoreMem0Offset16_Ss::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let val = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_COPY_RS, val, MINI_I64_STORE_SR, ptr, offset]);
            }
            OpCode::U32StoreMem0Offset16_Ss => {
                let op = decode::U32StoreMem0Offset16_Ss::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                let val = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_COPY_RS, val, MINI_I32_STORE_SR, ptr, offset]);
            }
            // --- Store with immediate value (ptr=accumulator, val=immediate) ---
            // Decompose: save ptr(ireg) to scratch, load imm into ireg, then _SR.
            OpCode::U64StoreMem0Offset16_Ri => {
                let op = decode::U64StoreMem0Offset16_Ri::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let imm = op.value as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    imm,
                    MINI_I64_STORE_SR,
                    scratch_base,
                    offset,
                ]);
            }
            OpCode::U32StoreMem0Offset16_Ri => {
                let op = decode::U32StoreMem0Offset16_Ri::decode(&mut cursor).ok()?;
                let offset = u64::from(op.offset) as i64;
                let imm = i64::from(op.value);
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    imm,
                    MINI_I32_STORE_SR,
                    scratch_base,
                    offset,
                ]);
            }
            // --- Load with pointer from a slot (ptr=slot, result=accumulator) ---
            // Decompose: copy slot to ireg, then use existing load (reads ptr from ireg).
            OpCode::U64LoadMem0Offset16_Rs => {
                let op = decode::U64LoadMem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_I64_LOAD_MEM0_OFF, offset]);
            }
            OpCode::U32LoadMem0Offset16_Rs => {
                let op = decode::U32LoadMem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_I32_LOAD_MEM0_OFF, offset]);
            }
            OpCode::I32Add_Rss => {
                // slot + slot -> reg (no slot result): reuse the slot-and-reg add
                // and route its slot write to a throwaway scratch slot.
                let op = decode::I32Add_Rss::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_ADD_SS_WB, scratch_base, lhs, rhs]);
            }
            OpCode::I64Sext32_Rr => {
                decode::I64Sext32_Rr::decode(&mut cursor).ok()?;
                words.push(MINI_I64_SEXT32);
            }
            OpCode::I64Sext32_Rs => {
                let op = decode::I64Sext32_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I64_SEXT32_S, src]);
            }
            // i32.wrap_i64: truncate i64 to i32. In the kernel's canonical i32
            // representation (sign-extended low 32 bits), this is equivalent to
            // i64.extend_i32_s — reuse the same MINI op.
            OpCode::I32WrapI64_Rr => {
                decode::I32WrapI64_Rr::decode(&mut cursor).ok()?;
                words.push(MINI_I64_SEXT32);
            }
            OpCode::I32WrapI64_Rs => {
                let op = decode::I32WrapI64_Rs::decode(&mut cursor).ok()?;
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_I64_SEXT32_S, src]);
            }
            // i32.wrap_i64: truncate i64 to i32. In the kernel's canonical i32
            // representation (sign-extended low 32 bits), this is equivalent to
            // i64.extend_i32_s — reuse the same MINI op.
            OpCode::I32WrapI64_Rr => {
                decode::I32WrapI64_Rr::decode(&mut cursor).ok()?;
                words.push(MINI_I64_SEXT32);
            }
            OpCode::I32WrapI64_Rs => {
                let op = decode::I32WrapI64_Rs::decode(&mut cursor).ok()?;
                let src = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_I64_SEXT32_S, src]);
            }
            OpCode::U64Shr_Rsi => {
                let op = decode::U64Shr_Rsi::decode(&mut cursor).ok()?;
                let lhs = s!(op.lhs);
                // wasm shift amount is taken mod 64.
                let n = u32::from(u8::from(op.rhs)) & 63;
                let shift = i64::from(n);
                // Mask off the `n` high bits the arithmetic `>>` would sign-extend,
                // reproducing the logical (zero-fill) shift.
                let mask = if n == 0 {
                    -1i64
                } else {
                    ((1u64 << (64 - n)) - 1) as i64
                };
                words.extend_from_slice(&[MINI_U64_SHR_SI, lhs, shift, mask]);
            }
            OpCode::I64Add_Rs_si => {
                let op = decode::I64Add_Rs_si::decode(&mut cursor).ok()?;
                let dst = s!(Slot::from(op.result));
                let lhs = s!(op.lhs);
                // Pre-materialize the immediate into a scratch slot, then reuse the
                // uniform two-slot add (`MINI_I64_ADD_SS_WB`). The copy folds away
                // in the compiled trace; correctness is identical to a dedicated
                // slot+imm arm.
                words.extend_from_slice(&[MINI_COPY_SI, scratch_base, op.rhs]);
                words.extend_from_slice(&[MINI_I64_ADD_SS_WB, dst, lhs, scratch_base]);
            }
            OpCode::I64Add_Rs_ss => {
                let op = decode::I64Add_Rs_ss::decode(&mut cursor).ok()?;
                let dst = s!(Slot::from(op.result));
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I64_ADD_SS_WB, dst, lhs, rhs]);
            }
            OpCode::BranchI64Le_Si => {
                let op = decode::BranchI64Le_Si::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = s!(op.lhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_LE_SI, 0, lhs, op.rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I32Add_Rs_ss => {
                let op = decode::I32Add_Rs_ss::decode(&mut cursor).ok()?;
                let dst = s!(Slot::from(op.result));
                let lhs = s!(op.lhs);
                let rhs = s!(op.rhs);
                words.extend_from_slice(&[MINI_I32_ADD_SS_WB, dst, lhs, rhs]);
            }
            OpCode::BranchI32Lt_Si => {
                let op = decode::BranchI32Lt_Si::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = s!(op.lhs);
                // i32 immediate: sign-extend to i64 so the kernel's sign-extended
                // slot compares against it in i32 signed order.
                let imm = i64::from(op.rhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I32_LT_SI, 0, lhs, imm]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchI64Lt_Ir => {
                let op = decode::BranchI64Lt_Ir::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_LT_IR, 0, op.lhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchI64Eq_Si => {
                let op = decode::BranchI64Eq_Si::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = s!(op.lhs);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_EQ_SI, 0, lhs, op.rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchI64NotEq_Ri => {
                let op = decode::BranchI64NotEq_Ri::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_NE_RI, 0, op.rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Branch if ireg == immediate: spill ireg to scratch and use the
            // slot-sourced equality branch.
            OpCode::BranchI64Eq_Ri => {
                let op = decode::BranchI64Eq_Ri::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                words.push(MINI_COPY_SR);
                words.push(scratch_base);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_EQ_SI, 0, scratch_base, op.rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            // Integer global read: `ireg = global_get(idx)`. Only `GlobalGetU64_R`
            // (i32/i64 globals) is lowered; f32/f64 globals stay ineligible.
            OpCode::GlobalGetU64_R => {
                let op = decode::GlobalGetU64_R::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[MINI_GLOBAL_GET_R, idx]);
                uses_globals = true;
            }
            // Integer global write from the accumulator: pre-materialize `ireg`
            // into a scratch slot, then the slot-sourced set arm.
            OpCode::GlobalSetU64_R => {
                let op = decode::GlobalSetU64_R::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_GLOBAL_SET_S,
                    idx,
                    scratch_base,
                ]);
                uses_globals = true;
            }
            // Integer global write from a slot: direct.
            OpCode::GlobalSetU64_S => {
                let op = decode::GlobalSetU64_S::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                let src = s!(op.value);
                words.extend_from_slice(&[MINI_GLOBAL_SET_S, idx, src]);
                uses_globals = true;
            }
            // Integer global write from an i64 immediate: materialize then set.
            OpCode::GlobalSetU64_I => {
                let op = decode::GlobalSetU64_I::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    op.value as i64,
                    MINI_GLOBAL_SET_S,
                    idx,
                    scratch_base,
                ]);
                uses_globals = true;
            }
            // Integer global write from a u32 immediate: zero-extend (matching the
            // stock `write_as::<u32>`) then set.
            OpCode::GlobalSetU32_I => {
                let op = decode::GlobalSetU32_I::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    i64::from(u32::from(op.value)),
                    MINI_GLOBAL_SET_S,
                    idx,
                    scratch_base,
                ]);
                uses_globals = true;
            }
            // f32 global read into the f32 accumulator.
            OpCode::GlobalGetF32_R => {
                let op = decode::GlobalGetF32_R::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[MINI_GLOBAL_GET_F32, idx]);
                uses_globals = true;
            }
            // f64 global read into the f64 accumulator.
            OpCode::GlobalGetF64_R => {
                let op = decode::GlobalGetF64_R::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[MINI_GLOBAL_GET_F64, idx]);
                uses_globals = true;
            }
            // f32 global write from the f32 accumulator: spill `freg32` into a
            // scratch slot, then the slot-sourced set arm.
            OpCode::GlobalSetF32_R => {
                let op = decode::GlobalSetF32_R::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[
                    MINI_COPY_S_F32R,
                    scratch_base,
                    MINI_GLOBAL_SET_S,
                    idx,
                    scratch_base,
                ]);
                uses_globals = true;
            }
            // f64 global write from the f64 accumulator: spill `freg64` then set.
            OpCode::GlobalSetF64_R => {
                let op = decode::GlobalSetF64_R::decode(&mut cursor).ok()?;
                let idx = i64::from(u32::from(op.global));
                words.extend_from_slice(&[
                    MINI_COPY_S_FR,
                    scratch_base,
                    MINI_GLOBAL_SET_S,
                    idx,
                    scratch_base,
                ]);
                uses_globals = true;
            }
            // Tail calls and internal calls: the kernel cannot handle these
            // directly. Emit MINI_RETURN_BAIL so the kernel runs the code
            // before the call/tail-call and then signals the caller to fall
            // back to the stock executor. This makes the function eligible
            // (the hot loop body before the call benefits from JIT) instead of
            // rejecting the entire function.
            OpCode::ReturnCallIndirect_R => {
                let _op = decode::ReturnCallIndirect_R::decode(&mut cursor).ok()?;
                words.push(MINI_RETURN_BAIL);
                has_yield_or_bail = true;
            }
            OpCode::ReturnCallIndirect_S => {
                let _op = decode::ReturnCallIndirect_S::decode(&mut cursor).ok()?;
                words.push(MINI_RETURN_BAIL);
                has_yield_or_bail = true;
            }
            OpCode::ReturnCallInternal => {
                let _op = decode::ReturnCallInternal::decode(&mut cursor).ok()?;
                words.push(MINI_RETURN_BAIL);
                has_yield_or_bail = true;
            }
            OpCode::CallInternal => {
                // Execute the call via a #[dont_look_inside] residual. The
                // kernel stages params from its slots, calls the residual which
                // executes the callee on a separate Stack, and stores the return
                // value in ireg. The kernel continues execution after the call.
                let op = decode::CallInternal::decode(&mut cursor).ok()?;
                let func_addr = usize::from(op.func) as i64;
                let params_start = i64::from(u16::from(op.params.span().head()));
                let params_len = i64::from(op.params.len());
                words.extend_from_slice(&[MINI_CALL_RESIDUAL, func_addr, params_start, params_len]);
            }
            OpCode::Trap => {
                // Unconditional trap (wasm `unreachable`). The kernel sets
                // the trap code and returns; run_jit surfaces it.
                let op = decode::Trap::decode(&mut cursor).ok()?;
                let code = u8::from(op.trap_code) as i64;
                words.extend_from_slice(&[MINI_TRAP, code]);
                has_yield_or_bail = true;
            }
            OpCode::MemorySize => {
                let op = decode::MemorySize::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                // Result goes into ireg (RegInt). MINI_MEMORY_SIZE is 1 word.
                words.push(MINI_MEMORY_SIZE);
            }
            OpCode::U32LoadExtend8_Rr => {
                // ptr+offset both dynamic (Reg operands). Yield to stock.
                let _op = decode::U32LoadExtend8_Rr::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_YIELD_STOCK, pos as i64, scratch_base]);
                has_yield_or_bail = true;
            }
            OpCode::U32LoadExtend16_Ri => {
                let op = decode::U32LoadExtend16_Ri::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                // _Ri = address is Immediate. Materialize → ireg, u16 load offset 0.
                let addr = u64::from(op.address) as i64;
                words.extend_from_slice(&[MINI_COPY_RI, addr, MINI_U16_LOAD_MEM0_OFF, 0]);
            }
            OpCode::U64Store_Is => {
                let op = decode::U64Store_Is::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                // _Is = address is Imm (Address), value in Slot.
                // Decompose: address → ireg, then i64 store with value slot.
                let addr = u64::from(op.address) as i64;
                let val = i64::from(u16::from(op.value));
                words.extend_from_slice(&[MINI_COPY_RI, addr, MINI_I64_STORE_RS, 0, val]);
            }
            OpCode::CallIndirect_S => {
                // Indirect call — residual through the kernel.
                let op = decode::CallIndirect_S::decode(&mut cursor).ok()?;
                let table = u32::from(op.table) as i64;
                let func_type = u32::from(op.func_type) as i64;
                let index_slot = i64::from(u16::from(op.index));
                let params_start = i64::from(u16::from(op.params.span().head()));
                let params_len = i64::from(op.params.len());
                words.extend_from_slice(&[
                    MINI_CALL_INDIRECT, table, func_type, index_slot,
                    params_start, params_len,
                ]);
                has_yield_or_bail = true;
            }
            // -- Bail ops that unblock runtime infrastructure functions --
            // These addressing forms are uncommon but appear in 788-1590 byte
            // runtime functions. Yield-to-stock keeps them eligible (the hot
            // loop body after the yield point benefits from JIT).
            OpCode::BranchU32Lt_Ir => {
                // if imm <u ireg goto target. Materialize both into scratch,
                // compute comparison, branch.
                let op = decode::BranchU32Lt_Ir::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let imm = i64::from(u32::from(op.lhs));
                let target_field = words.len() + 9;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm, // scratch = imm (lhs)
                    MINI_COPY_SR,
                    scratch_base + 1, // scratch+1 = ireg (rhs)
                    MINI_U32_LT_SS_R,
                    scratch_base,
                    scratch_base + 1,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I64Add_Rs_ri => {
                // slot[result] = ireg + imm (i64). Decompose: materialize
                // both operands into scratch slots, then slot-slot add.
                let op = decode::I64Add_Rs_ri::decode(&mut cursor).ok()?;
                let dst = i64::from(u16::from(Slot::from(op.result)));
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base, // scratch = ireg
                    MINI_COPY_SI,
                    scratch_base + 1,
                    op.rhs, // scratch+1 = imm
                    MINI_I64_ADD_SS_WB,
                    dst,
                    scratch_base,
                    scratch_base + 1,
                ]);
            }
            OpCode::U64LoadExtend32Mem0Offset16_Rs => {
                // i64.load32_u: load 4 bytes from mem[slot[ptr]+offset],
                // zero-extend to i64. Decompose: i32 load (sign-extends)
                // then mask to zero-extend.
                let op = decode::U64LoadExtend32Mem0Offset16_Rs::decode(&mut cursor).ok()?;
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[
                    MINI_COPY_RS,
                    ptr,
                    MINI_I32_LOAD_MEM0_OFF,
                    offset,
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I64_AND_SI_WR,
                    scratch_base,
                    0x_FFFF_FFFF_i64,
                ]);
            }
            OpCode::U32Store_Ir => {
                // i32.store at absolute address, value in ireg. Memory 0 only.
                let op = decode::U32Store_Ir::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                let addr = u64::from(op.address) as i64;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base, // scratch = ireg (value)
                    MINI_COPY_RI,
                    addr, // ireg = address (ptr)
                    MINI_I32_STORE_RS,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::U32Store_Ii => {
                // i32.store at absolute address, value is immediate. Memory 0 only.
                let op = decode::U32Store_Ii::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                let addr = u64::from(op.address) as i64;
                let val = i64::from(u32::from(op.value));
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    val, // scratch = value
                    MINI_COPY_RI,
                    addr, // ireg = address (ptr)
                    MINI_I32_STORE_RS,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::I64Lt_Rsr => {
                // ireg = (slot[lhs] < ireg) (signed i64 comparison, 0/1 value).
                // Spill ireg to scratch, then slot-slot comparison.
                let op = decode::I64Lt_Rsr::decode(&mut cursor).ok()?;
                let lhs = i64::from(u16::from(op.lhs));
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base, // scratch = ireg (rhs)
                    MINI_I64_LT_SS_R,
                    lhs,
                    scratch_base,
                ]);
            }
            OpCode::BranchU32Lt_Rs => {
                // if ireg <u slot[rhs] goto target. Spill ireg to scratch,
                // compute comparison, branch.
                let op = decode::BranchU32Lt_Rs::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let rhs = i64::from(u16::from(op.rhs));
                let target_field = words.len() + 6;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base, // scratch = ireg (lhs)
                    MINI_U32_LT_SS_R,
                    scratch_base,
                    rhs,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchU64Lt_Ir => {
                // if imm <u ireg (u64) goto target. Materialize both into
                // scratch, compute comparison, branch.
                let op = decode::BranchU64Lt_Ir::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let imm = op.lhs as i64;
                let target_field = words.len() + 9;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_COPY_SR,
                    scratch_base + 1,
                    MINI_U64_LT_SS_R,
                    scratch_base,
                    scratch_base + 1,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchU32Lt_Ss => {
                // if slot[lhs] <u slot[rhs] goto target. Decompose:
                // compute comparison into ireg, then branch on non-zero.
                let op = decode::BranchU32Lt_Ss::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = i64::from(u16::from(op.lhs));
                let rhs = i64::from(u16::from(op.rhs));
                let target_field = words.len() + 4;
                words.extend_from_slice(&[
                    MINI_U32_LT_SS_R,
                    lhs,
                    rhs,
                    MINI_BR_I32_NE_RI,
                    0,
                    0, // target patched by fixup
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchTable_R => {
                let _op = decode::BranchTable_R::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_YIELD_STOCK, pos as i64, scratch_base]);
                has_yield_or_bail = true;
            }
            OpCode::BranchI32Lt_Ri => {
                // if ireg <s imm (i32 signed) goto target. Materialize imm
                // into scratch, use I32_LT_RS_R (ireg < slot), branch.
                let op = decode::BranchI32Lt_Ri::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let imm = i64::from(i32::from(op.rhs));
                let target_field = words.len() + 6;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_I32_LT_RS_R,
                    scratch_base,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::F64NotLe_Rss => {
                // !(lhs <= rhs) for f64. Decompose:
                // 1. slot[lhs] → ireg → freg (bit copy via reinterpret)
                // 2. F64_CMP sel=1(le) with slot[rhs] → ireg = (freg <= slot[rhs]) ? 1 : 0
                // 3. Negate: ireg = (ireg == 0) ? 1 : 0
                let op = decode::F64NotLe_Rss::decode(&mut cursor).ok()?;
                let lhs = i64::from(u16::from(op.lhs));
                let rhs = i64::from(u16::from(op.rhs));
                words.extend_from_slice(&[
                    MINI_COPY_RS,
                    lhs,                   // ireg = slot[lhs] (f64 bits as i64)
                    MINI_F64_REINTERP_I64, // freg = ireg (reinterpret to f64)
                    MINI_F64_CMP_RS_R,
                    1,
                    rhs, // ireg = (freg <= slot[rhs]) ? 1 : 0
                    MINI_COPY_SI,
                    scratch_base,
                    0, // scratch = 0
                    MINI_I32_EQ_RS_R,
                    scratch_base, // ireg = (ireg == 0) ? 1 : 0 = !le
                ]);
            }
            OpCode::U32Select_Rsii => {
                // ireg = slot[condition] ? true_val : false_val (both u32 imm).
                // Materialize both into scratch, load condition into ireg, select.
                let op = decode::U32Select_Rsii::decode(&mut cursor).ok()?;
                let cond = i64::from(u16::from(op.condition));
                let t = i64::from(u32::from(op.true_val));
                let f = i64::from(u32::from(op.false_val));
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    t,
                    MINI_COPY_SI,
                    scratch_base + 1,
                    f,
                    MINI_COPY_RS,
                    cond,
                    MINI_SELECT,
                    scratch_base,
                    scratch_base + 1,
                ]);
            }
            OpCode::I32Eq_Rss => {
                // ireg = (slot[lhs] == slot[rhs]) (i32 comparison, 0/1 value).
                let op = decode::I32Eq_Rss::decode(&mut cursor).ok()?;
                let lhs = i64::from(u16::from(op.lhs));
                let rhs = i64::from(u16::from(op.rhs));
                words.extend_from_slice(&[MINI_I32_EQ_SS_R, lhs, rhs]);
            }
            OpCode::BranchU64Lt_Si => {
                // if slot[lhs] <u imm (u64) goto target.
                let op = decode::BranchU64Lt_Si::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = i64::from(u16::from(op.lhs));
                let imm = op.rhs as i64;
                let target_field = words.len() + 7;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_U64_LT_SS_R,
                    lhs,
                    scratch_base,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::U64Shr_Rir => {
                // ireg = (u64)imm >> ((u64)ireg & 63). Non-commutative:
                // lhs is immediate, rhs (shift amount) is ireg.
                // Decompose: materialize both into scratch, then U64_SHR_SS_WR.
                let op = decode::U64Shr_Rir::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    op.lhs as i64, // scratch = imm (lhs value)
                    MINI_COPY_SR,
                    scratch_base + 1, // scratch+1 = ireg (shift amt)
                    MINI_U64_SHR_SS_WR,
                    scratch_base,
                    scratch_base + 1,
                ]);
            }
            OpCode::I64Lt_Rss => {
                // ireg = (i64)slot[lhs] < (i64)slot[rhs] ? 1 : 0 (signed).
                let op = decode::I64Lt_Rss::decode(&mut cursor).ok()?;
                let lhs = i64::from(u16::from(op.lhs));
                let rhs = i64::from(u16::from(op.rhs));
                words.extend_from_slice(&[MINI_I64_LT_SS_R, lhs, rhs]);
            }
            OpCode::I32Shl_Rri => {
                // ireg = (i32)ireg << imm. Materialize ireg into scratch,
                // then I32_SHL_SI.
                let op = decode::I32Shl_Rri::decode(&mut cursor).ok()?;
                let shift = i64::from(u8::from(op.rhs));
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I32_SHL_SI,
                    scratch_base,
                    shift,
                ]);
            }
            OpCode::U64Select_Rrrs => {
                // ireg = ireg ? ireg : slot[false_val]. condition and true_val
                // share ireg; spill to scratch so SELECT reads true from a slot.
                let op = decode::U64Select_Rrrs::decode(&mut cursor).ok()?;
                let false_slot = i64::from(u16::from(op.false_val));
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base, // scratch = ireg (true_val)
                    MINI_SELECT,
                    scratch_base,
                    false_slot,
                ]);
            }
            OpCode::CallIndirect_R => {
                // Indirect call via Reg index — residual through the kernel.
                let op = decode::CallIndirect_R::decode(&mut cursor).ok()?;
                let table = u32::from(op.table) as i64;
                let func_type = u32::from(op.func_type) as i64;
                let params_start = i64::from(u16::from(op.params.span().head()));
                let params_len = i64::from(op.params.len());
                words.extend_from_slice(&[MINI_COPY_SR, scratch_base]);
                words.extend_from_slice(&[
                    MINI_CALL_INDIRECT, table, func_type, scratch_base,
                    params_start, params_len,
                ]);
                has_yield_or_bail = true;
            }
            OpCode::BranchU32Lt_Si => {
                // if slot[lhs] <u imm goto target. Materialize imm into
                // scratch, compute U32_LT_SS_R, then branch on non-zero.
                let op = decode::BranchU32Lt_Si::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = i64::from(u16::from(op.lhs));
                let imm = i64::from(u32::from(op.rhs));
                let target_field = words.len() + 7;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_U32_LT_SS_R,
                    lhs,
                    scratch_base,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::U32LoadExtend8_Ri => {
                // i32.load8_u at absolute address. Memory 0 only.
                // Decompose: address → ireg, then u8 load offset 0.
                let op = decode::U32LoadExtend8_Ri::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                let addr = u64::from(op.address) as i64;
                words.extend_from_slice(&[MINI_COPY_RI, addr, MINI_U8_LOAD_MEM0_OFF, 0]);
            }
            OpCode::I32BitAnd_Rsi => {
                // ireg = slot[lhs] & imm (i32). Materialize both into scratch
                // slots, then reuse the two-slot AND.
                let op = decode::I32BitAnd_Rsi::decode(&mut cursor).ok()?;
                let lhs = i64::from(u16::from(op.lhs));
                let imm = i64::from(op.rhs);
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm,
                    MINI_I32_AND_SS_WR,
                    lhs,
                    scratch_base,
                ]);
            }
            OpCode::U32Store_Is => {
                // i32.store at absolute address, value in slot. Memory 0 only.
                // Decompose: address → ireg, then i32 store with value slot.
                let op = decode::U32Store_Is::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                let addr = u64::from(op.address) as i64;
                let val = i64::from(u16::from(op.value));
                words.extend_from_slice(&[
                    MINI_COPY_RS,
                    val,
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_COPY_RI,
                    addr,
                    MINI_I32_STORE_RS,
                    0,
                    scratch_base,
                ]);
            }
            OpCode::BranchI32Eq_Rs => {
                // if ireg == slot[rhs] (i32) goto target. Spill ireg to scratch
                // and use the i64 slot-sourced equality branch.
                let op = decode::BranchI32Eq_Rs::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let rhs = i64::from(u16::from(op.rhs));
                words.push(MINI_COPY_SR);
                words.push(scratch_base);
                let target_field = words.len() + 1;
                words.extend_from_slice(&[MINI_BR_I64_EQ_SS, 0, scratch_base, rhs]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I64Eq_Rri => {
                // ireg = (ireg == imm) ? 1 : 0 (i64). Materialize imm into
                // scratch, then use I64_EQ_RS_R.
                let op = decode::I64Eq_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    op.rhs,
                    MINI_I64_EQ_RS_R,
                    scratch_base,
                ]);
            }
            OpCode::BranchI64Lt_Rs => {
                // if ireg <s slot[rhs] (i64 signed) goto target. Use
                // I64_LT_RS_R to compute the comparison, then branch.
                let op = decode::BranchI64Lt_Rs::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let rhs = i64::from(u16::from(op.rhs));
                let target_field = words.len() + 3;
                words.extend_from_slice(&[MINI_I64_LT_RS_R, rhs, MINI_BR_I32_NE_RI, 0, 0]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::CallImported => {
                // Imported function call — dispatch through the imported-call
                // runner which resolves the Func handle via the store, then
                // executes either the wasm callee or a host trampoline.
                let op = decode::CallImported::decode(&mut cursor).ok()?;
                let func_idx = u32::from(op.func) as i64;
                let params_start = i64::from(u16::from(op.params.span().head()));
                let params_len = i64::from(op.params.len());
                words.extend_from_slice(&[
                    MINI_CALL_IMPORTED,
                    func_idx,
                    params_start,
                    params_len,
                ]);
                has_yield_or_bail = true;
            }
            OpCode::MemoryCopy => {
                // memory.copy within the same linear memory — handled natively
                // by the mem_copy_within residual helper. Cross-memory copy
                // (dst_memory != src_memory) bails the function.
                let op = decode::MemoryCopy::decode(&mut cursor).ok()?;
                if op.dst_memory != op.src_memory {
                    return None;
                }
                let dst = i64::from(u16::from(op.dst));
                let src = i64::from(u16::from(op.src));
                let len = i64::from(u16::from(op.len));
                words.extend_from_slice(&[MINI_MEM_COPY_WITHIN, dst, src, len]);
            }
            OpCode::BranchI64Le_Rs => {
                // if ireg <=s slot[rhs] (i64 signed) goto target.
                let op = decode::BranchI64Le_Rs::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let rhs = i64::from(u16::from(op.rhs));
                // Spill ireg to scratch, then use I64_LE_SS_R (lhs=scratch, rhs=rhs).
                let target_field = words.len() + 6;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_I64_LE_SS_R,
                    scratch_base,
                    rhs,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::BranchU32Lt_Sr => {
                // if slot[lhs] <u ireg goto target. Spill ireg to scratch,
                // then compare slot vs scratch.
                let op = decode::BranchU32Lt_Sr::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = i64::from(u16::from(op.lhs));
                let target_field = words.len() + 6;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_U32_LT_SS_R,
                    lhs,
                    scratch_base,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I64Shr_Rsi => {
                // Arithmetic i64 right shift. No MINI_I64_SHR op exists yet;
                // yield to stock.
                let _op = decode::I64Shr_Rsi::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[MINI_YIELD_STOCK, pos as i64, scratch_base]);
                has_yield_or_bail = true;
            }
            OpCode::BranchU32Le_Sr => {
                // if slot[lhs] <=u ireg goto target. Spill ireg to scratch,
                // then compare slot vs scratch with U32_LE_SS_R.
                let op = decode::BranchU32Le_Sr::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let lhs = i64::from(u16::from(op.lhs));
                let target_field = words.len() + 6;
                words.extend_from_slice(&[
                    MINI_COPY_SR,
                    scratch_base,
                    MINI_U32_LE_SS_R,
                    lhs,
                    scratch_base,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I64BitOr_Rsi => {
                // ireg = slot[lhs] | sign_extend(imm16). Decompose as
                // COPY_RS(lhs) + I64_OR_RI_WR(imm).
                let op = decode::I64BitOr_Rsi::decode(&mut cursor).ok()?;
                let lhs = i64::from(u16::from(op.lhs));
                words.extend_from_slice(&[MINI_COPY_RS, lhs, MINI_I64_OR_RI_WR, op.rhs]);
            }
            OpCode::BranchU32Le_Ir => {
                // if imm <=u ireg goto target. Materialize both into scratch,
                // compute comparison, branch.
                let op = decode::BranchU32Le_Ir::decode(&mut cursor).ok()?;
                let offset = i32::from(op.offset) as isize;
                let target_byte = pos.checked_add_signed(offset)?;
                let imm = i64::from(u32::from(op.lhs));
                let target_field = words.len() + 9;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    imm, // scratch = imm (lhs)
                    MINI_COPY_SR,
                    scratch_base + 1, // scratch+1 = ireg (rhs)
                    MINI_U32_LE_SS_R,
                    scratch_base,
                    scratch_base + 1,
                    MINI_BR_I32_NE_RI,
                    0,
                    0,
                ]);
                fixups.push((target_field, target_byte, offset < 0));
            }
            OpCode::I32Eq_Rri => {
                // ireg = (ireg == imm) ? 1 : 0 (i32). Materialize imm into
                // scratch, then use I32_EQ_RS_R.
                let op = decode::I32Eq_Rri::decode(&mut cursor).ok()?;
                words.extend_from_slice(&[
                    MINI_COPY_SI,
                    scratch_base,
                    i64::from(op.rhs),
                    MINI_I32_EQ_RS_R,
                    scratch_base,
                ]);
            }
            OpCode::U64LoadExtend8_Rs => {
                // u8 load from slot ptr + offset, zero-extend to u64.
                // Memory 0 only.
                let op = decode::U64LoadExtend8_Rs::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                let ptr = i64::from(u16::from(op.ptr));
                let offset = u64::from(op.offset) as i64;
                words.extend_from_slice(&[MINI_COPY_RS, ptr, MINI_U8_LOAD_MEM0_OFF, offset]);
            }
            OpCode::U64LoadExtend32_Ri => {
                // u32 load from absolute address, zero-extend to u64.
                // Materialize the address into ireg, then i32 load, then mask
                // to u32 (the i32 load sign-extends, but we want zero-extend).
                let op = decode::U64LoadExtend32_Ri::decode(&mut cursor).ok()?;
                if u32::from(op.memory) != 0 {
                    return None;
                }
                let addr = u64::from(op.address) as i64;
                words.extend_from_slice(&[
                    MINI_COPY_RI,
                    addr,
                    MINI_I32_LOAD_MEM0_OFF,
                    0,
                    MINI_I64_AND_RI_WR,
                    0x0000_0000_FFFF_FFFFi64,
                ]);
            }
            // Any other op makes the function ineligible for the JIT tier.
            #[allow(unused_variables)]
            other => {
                #[cfg(feature = "std")]
                if std::env::var_os("WASMI_MAJIT_STATS").is_some() {
                    let pos = total - cursor.len();
                    std::eprintln!("[majit-prepass] bail at op {other:?} (byte ~{pos})");
                }
                return None;
            }
        }
    }

    let mut loop_header_word = None;
    for (target_field, target_byte, is_back) in &fixups {
        // A branch into the middle of an op (not an op boundary) is malformed.
        let target_word = *byte_to_word.get(target_byte)?;
        words[*target_field] = target_word as i64;
        if *is_back {
            loop_header_word = Some(target_word);
        }
    }

    Some(MiniProgram {
        words,
        num_slots: dense_count,
        slot_map,
        loop_header_word,
        writes_result,
        uses_globals,
        has_yield_or_bail,
    })
}

/// Observe a function's op stream by decoding a known subset, printing each op
/// (and stopping at the first op outside the subset, dumping its raw bytes).
///
/// A full generic disassembler is not viable: zero-operand control ops such as
/// `Return` have no `decode::*` struct, so a `for_each_op!`-driven match fails
/// to compile. The prepass only needs the subset anyway, so the walker decodes
/// the subset and reports anything else.
#[cfg(test)]
pub(crate) fn disasm_observe(ops: &[u8]) {
    use crate::ir::Decode;

    let total = ops.len();
    let mut cursor: &[u8] = ops;
    macro_rules! dec {
        ($ty:ident) => {{
            crate::ir::decode::$ty::decode(&mut cursor).expect("operand decode");
        }};
    }
    while !cursor.is_empty() {
        let pos = total - cursor.len();
        let code = OpCode::decode(&mut cursor).expect("opcode decode");
        match code {
            OpCode::ConsumeFuel => dec!(ConsumeFuel),
            OpCode::Branch => dec!(Branch),
            OpCode::Trap => dec!(Trap),
            OpCode::Return => { /* zero operands */ }
            OpCode::I32Sub_Rrs => dec!(I32Sub_Rrs),
            OpCode::I32Sub_Rsr => dec!(I32Sub_Rsr),
            OpCode::I32Sub_Rss => dec!(I32Sub_Rss),
            OpCode::I32Sub_Rir => dec!(I32Sub_Rir),
            OpCode::I32Sub_Ris => dec!(I32Sub_Ris),
            OpCode::I32Add_Rrs => dec!(I32Add_Rrs),
            OpCode::I32Add_Rri => dec!(I32Add_Rri),
            OpCode::I32Add_Rss => dec!(I32Add_Rss),
            OpCode::I32Add_Rsi => dec!(I32Add_Rsi),
            OpCode::I32Add_Rs_rs => dec!(I32Add_Rs_rs),
            OpCode::I32Add_Rs_ri => dec!(I32Add_Rs_ri),
            OpCode::I32Add_Rs_ss => dec!(I32Add_Rs_ss),
            OpCode::I32Add_Rs_si => dec!(I32Add_Rs_si),
            OpCode::U32Copy_Ri => dec!(U32Copy_Ri),
            OpCode::U32Copy_Si => dec!(U32Copy_Si),
            OpCode::U64Copy_Rs => dec!(U64Copy_Rs),
            OpCode::U64Copy_Ri => dec!(U64Copy_Ri),
            OpCode::U64Copy_Sr => dec!(U64Copy_Sr),
            OpCode::U64Copy_Ss => dec!(U64Copy_Ss),
            OpCode::U64Copy_Si => dec!(U64Copy_Si),
            OpCode::U64Copy_S0r => dec!(U64Copy_S0r),
            OpCode::U64Copy_S1r => dec!(U64Copy_S1r),
            OpCode::U64Copy_S2r => dec!(U64Copy_S2r),
            OpCode::U64Copy_S3r => dec!(U64Copy_S3r),
            OpCode::U64Copy_S4r => dec!(U64Copy_S4r),
            OpCode::U64Copy_S5r => dec!(U64Copy_S5r),
            OpCode::U64Copy_S6r => dec!(U64Copy_S6r),
            OpCode::U64Copy_S7r => dec!(U64Copy_S7r),
            OpCode::U64Copy_S8r => dec!(U64Copy_S8r),
            OpCode::U64Copy_S9r => dec!(U64Copy_S9r),
            OpCode::F64Copy_S0r => dec!(F64Copy_S0r),
            OpCode::F64Copy_S1r => dec!(F64Copy_S1r),
            OpCode::F64Copy_S2r => dec!(F64Copy_S2r),
            OpCode::F64Copy_S3r => dec!(F64Copy_S3r),
            OpCode::F64Copy_S4r => dec!(F64Copy_S4r),
            OpCode::F64Copy_S5r => dec!(F64Copy_S5r),
            OpCode::F64Copy_S6r => dec!(F64Copy_S6r),
            OpCode::F64Copy_S7r => dec!(F64Copy_S7r),
            OpCode::F64Copy_S8r => dec!(F64Copy_S8r),
            OpCode::F64Copy_S9r => dec!(F64Copy_S9r),
            OpCode::F64Copy_Sr => dec!(F64Copy_Sr),
            OpCode::I64Mul_Rss => dec!(I64Mul_Rss),
            OpCode::I32Mul_Rss => dec!(I32Mul_Rss),
            OpCode::I32Mul_Rri => dec!(I32Mul_Rri),
            OpCode::I32BitXor_Rri => dec!(I32BitXor_Rri),
            OpCode::I32BitOr_Rri => dec!(I32BitOr_Rri),
            OpCode::I32BitAnd_Rri => dec!(I32BitAnd_Rri),
            OpCode::I32Mul_Rrs => dec!(I32Mul_Rrs),
            OpCode::I32BitXor_Rrs => dec!(I32BitXor_Rrs),
            OpCode::BranchU32Le_Ss => dec!(BranchU32Le_Ss),
            OpCode::I64BitAnd_Rsi => dec!(I64BitAnd_Rsi),
            OpCode::I64BitXor_Rss => dec!(I64BitXor_Rss),
            OpCode::I64BitAnd_Rri => dec!(I64BitAnd_Rri),
            OpCode::I64Mul_Rri => dec!(I64Mul_Rri),
            OpCode::I64BitOr_Rri => dec!(I64BitOr_Rri),
            OpCode::I64BitXor_Rri => dec!(I64BitXor_Rri),
            OpCode::I64Mul_Rrs => dec!(I64Mul_Rrs),
            OpCode::I64BitOr_Rrs => dec!(I64BitOr_Rrs),
            OpCode::I64BitXor_Rrs => dec!(I64BitXor_Rrs),
            OpCode::BranchI64Le_Ss => dec!(BranchI64Le_Ss),
            OpCode::U64Shr_Rsi => dec!(U64Shr_Rsi),
            OpCode::I64Shl_Rsi => dec!(I64Shl_Rsi),
            OpCode::I32Shl_Rsi => dec!(I32Shl_Rsi),
            OpCode::U32Shr_Rri => dec!(U32Shr_Rri),
            OpCode::I32Lt_Rsi => dec!(I32Lt_Rsi),
            OpCode::I64Lt_Ris => dec!(I64Lt_Ris),
            OpCode::I64Lt_Rsi => dec!(I64Lt_Rsi),
            OpCode::I32Eq_Rsi => dec!(I32Eq_Rsi),
            OpCode::I32NotEq_Rsi => dec!(I32NotEq_Rsi),
            OpCode::I32Le_Rsi => dec!(I32Le_Rsi),
            OpCode::U32Lt_Rsi => dec!(U32Lt_Rsi),
            OpCode::U32Le_Rsi => dec!(U32Le_Rsi),
            OpCode::I32Lt_Ris => dec!(I32Lt_Ris),
            OpCode::I32Le_Ris => dec!(I32Le_Ris),
            OpCode::U32Lt_Ris => dec!(U32Lt_Ris),
            OpCode::U32Le_Ris => dec!(U32Le_Ris),
            OpCode::I64Eq_Rsi => dec!(I64Eq_Rsi),
            OpCode::I64NotEq_Rsi => dec!(I64NotEq_Rsi),
            OpCode::I64Le_Rsi => dec!(I64Le_Rsi),
            OpCode::U64Lt_Rsi => dec!(U64Lt_Rsi),
            OpCode::U64Le_Rsi => dec!(U64Le_Rsi),
            OpCode::I64Le_Ris => dec!(I64Le_Ris),
            OpCode::U64Lt_Ris => dec!(U64Lt_Ris),
            OpCode::U64Le_Ris => dec!(U64Le_Ris),
            OpCode::I32Lt_Rrs => dec!(I32Lt_Rrs),
            OpCode::I32Lt_Rsr => dec!(I32Lt_Rsr),
            OpCode::U64Select_Rrss => dec!(U64Select_Rrss),
            OpCode::U32Select_Rrsi => dec!(U32Select_Rrsi),
            OpCode::U32Select_Rris => dec!(U32Select_Rris),
            OpCode::U32Select_Rrii => dec!(U32Select_Rrii),
            OpCode::U64Select_Rrsi => dec!(U64Select_Rrsi),
            OpCode::U64Select_Rris => dec!(U64Select_Rris),
            OpCode::U64Select_Rrii => dec!(U64Select_Rrii),
            OpCode::I32Eq_Rrs => dec!(I32Eq_Rrs),
            OpCode::I32NotEq_Rrs => dec!(I32NotEq_Rrs),
            OpCode::I64Eq_Rrs => dec!(I64Eq_Rrs),
            OpCode::I64NotEq_Rrs => dec!(I64NotEq_Rrs),
            OpCode::I64Lt_Rrs => dec!(I64Lt_Rrs),
            OpCode::I32Le_Rrs => dec!(I32Le_Rrs),
            OpCode::U32Lt_Rrs => dec!(U32Lt_Rrs),
            OpCode::U32Le_Rrs => dec!(U32Le_Rrs),
            OpCode::I32Mul_Rsi => dec!(I32Mul_Rsi),
            OpCode::U64LoadMem0Offset16_Rr => dec!(U64LoadMem0Offset16_Rr),
            OpCode::F64LoadMem0Offset16_Rr => dec!(F64LoadMem0Offset16_Rr),
            OpCode::F64Add_Rsr => dec!(F64Add_Rsr),
            OpCode::F64Sub_Rrs => dec!(F64Sub_Rrs),
            OpCode::F64Mul_Rrs => dec!(F64Mul_Rrs),
            OpCode::F64Div_Rrs => dec!(F64Div_Rrs),
            OpCode::F64Lt_Rrs => dec!(F64Lt_Rrs),
            OpCode::F64Le_Rrs => dec!(F64Le_Rrs),
            OpCode::F64Lt_Rsr => dec!(F64Lt_Rsr),
            OpCode::F64Le_Rsr => dec!(F64Le_Rsr),
            OpCode::F64Eq_Rrs => dec!(F64Eq_Rrs),
            OpCode::F64NotEq_Rrs => dec!(F64NotEq_Rrs),
            OpCode::F32Lt_Rrs => dec!(F32Lt_Rrs),
            OpCode::F32Le_Rrs => dec!(F32Le_Rrs),
            OpCode::F32Lt_Rsr => dec!(F32Lt_Rsr),
            OpCode::F32Le_Rsr => dec!(F32Le_Rsr),
            OpCode::F32Eq_Rrs => dec!(F32Eq_Rrs),
            OpCode::F32NotEq_Rrs => dec!(F32NotEq_Rrs),
            OpCode::F64Add_Rri => dec!(F64Add_Rri),
            OpCode::F64Sub_Rri => dec!(F64Sub_Rri),
            OpCode::F64Mul_Rri => dec!(F64Mul_Rri),
            OpCode::F64Div_Rri => dec!(F64Div_Rri),
            OpCode::F64Abs_Rs => dec!(F64Abs_Rs),
            OpCode::F64Abs_Rr => dec!(F64Abs_Rr),
            OpCode::F64Neg_Rs => dec!(F64Neg_Rs),
            OpCode::F64Neg_Rr => dec!(F64Neg_Rr),
            OpCode::F64Sqrt_Rs => dec!(F64Sqrt_Rs),
            OpCode::F64Sqrt_Rr => dec!(F64Sqrt_Rr),
            OpCode::F64Ceil_Rs => dec!(F64Ceil_Rs),
            OpCode::F64Ceil_Rr => dec!(F64Ceil_Rr),
            OpCode::F64Floor_Rs => dec!(F64Floor_Rs),
            OpCode::F64Floor_Rr => dec!(F64Floor_Rr),
            OpCode::F64Trunc_Rs => dec!(F64Trunc_Rs),
            OpCode::F64Trunc_Rr => dec!(F64Trunc_Rr),
            OpCode::F64Nearest_Rs => dec!(F64Nearest_Rs),
            OpCode::F64Nearest_Rr => dec!(F64Nearest_Rr),
            OpCode::F64Min_Rrs => dec!(F64Min_Rrs),
            OpCode::F64Min_Rsr => dec!(F64Min_Rsr),
            OpCode::F64Min_Rri => dec!(F64Min_Rri),
            OpCode::F64Max_Rrs => dec!(F64Max_Rrs),
            OpCode::F64Max_Rsr => dec!(F64Max_Rsr),
            OpCode::F64Max_Rri => dec!(F64Max_Rri),
            OpCode::F64Copysign_Rrs => dec!(F64Copysign_Rrs),
            OpCode::F64Copysign_Rsr => dec!(F64Copysign_Rsr),
            OpCode::F64ConvertI32_Rs => dec!(F64ConvertI32_Rs),
            OpCode::F64ConvertI32_Rr => dec!(F64ConvertI32_Rr),
            OpCode::F64ConvertU32_Rs => dec!(F64ConvertU32_Rs),
            OpCode::F64ConvertU32_Rr => dec!(F64ConvertU32_Rr),
            OpCode::F64ConvertI64_Rs => dec!(F64ConvertI64_Rs),
            OpCode::F64ConvertI64_Rr => dec!(F64ConvertI64_Rr),
            OpCode::F64ConvertU64_Rs => dec!(F64ConvertU64_Rs),
            OpCode::F64ConvertU64_Rr => dec!(F64ConvertU64_Rr),
            OpCode::I32TruncSatF64_Rs => dec!(I32TruncSatF64_Rs),
            OpCode::I32TruncSatF64_Rr => dec!(I32TruncSatF64_Rr),
            OpCode::U32TruncSatF64_Rs => dec!(U32TruncSatF64_Rs),
            OpCode::U32TruncSatF64_Rr => dec!(U32TruncSatF64_Rr),
            OpCode::I64TruncSatF64_Rs => dec!(I64TruncSatF64_Rs),
            OpCode::I64TruncSatF64_Rr => dec!(I64TruncSatF64_Rr),
            OpCode::U64TruncSatF64_Rs => dec!(U64TruncSatF64_Rs),
            OpCode::U64TruncSatF64_Rr => dec!(U64TruncSatF64_Rr),
            OpCode::I32TruncF64_Rs => dec!(I32TruncF64_Rs),
            OpCode::I32TruncF64_Rr => dec!(I32TruncF64_Rr),
            OpCode::U32TruncF64_Rs => dec!(U32TruncF64_Rs),
            OpCode::U32TruncF64_Rr => dec!(U32TruncF64_Rr),
            OpCode::I64TruncF64_Rs => dec!(I64TruncF64_Rs),
            OpCode::I64TruncF64_Rr => dec!(I64TruncF64_Rr),
            OpCode::U64TruncF64_Rs => dec!(U64TruncF64_Rs),
            OpCode::U64TruncF64_Rr => dec!(U64TruncF64_Rr),
            OpCode::U32LoadMem0Offset16_Rr => dec!(U32LoadMem0Offset16_Rr),
            OpCode::U32LoadExtend8Mem0Offset16_Rr => dec!(U32LoadExtend8Mem0Offset16_Rr),
            OpCode::I32LoadExtend8Mem0Offset16_Rr => dec!(I32LoadExtend8Mem0Offset16_Rr),
            OpCode::U32LoadExtend16Mem0Offset16_Rr => dec!(U32LoadExtend16Mem0Offset16_Rr),
            OpCode::I32LoadExtend16Mem0Offset16_Rr => dec!(I32LoadExtend16Mem0Offset16_Rr),
            OpCode::U32StoreMem0Offset16_Sr => dec!(U32StoreMem0Offset16_Sr),
            OpCode::U64StoreMem0Offset16_Sr => dec!(U64StoreMem0Offset16_Sr),
            OpCode::F64StoreMem0Offset16_Sr => dec!(F64StoreMem0Offset16_Sr),
            OpCode::I32StoreWrap8Mem0Offset16_Sr => dec!(I32StoreWrap8Mem0Offset16_Sr),
            OpCode::I32StoreWrap16Mem0Offset16_Sr => dec!(I32StoreWrap16Mem0Offset16_Sr),
            OpCode::U32StoreMem0Offset16_Rs => dec!(U32StoreMem0Offset16_Rs),
            OpCode::U64StoreMem0Offset16_Rs => dec!(U64StoreMem0Offset16_Rs),
            OpCode::I32StoreWrap8Mem0Offset16_Rs => dec!(I32StoreWrap8Mem0Offset16_Rs),
            OpCode::I32StoreWrap16Mem0Offset16_Rs => dec!(I32StoreWrap16Mem0Offset16_Rs),
            OpCode::U64StoreMem0Offset16_Si => dec!(U64StoreMem0Offset16_Si),
            OpCode::U32StoreMem0Offset16_Si => dec!(U32StoreMem0Offset16_Si),
            OpCode::U64StoreMem0Offset16_Ss => dec!(U64StoreMem0Offset16_Ss),
            OpCode::U32StoreMem0Offset16_Ss => dec!(U32StoreMem0Offset16_Ss),
            OpCode::U64StoreMem0Offset16_Ri => dec!(U64StoreMem0Offset16_Ri),
            OpCode::U32StoreMem0Offset16_Ri => dec!(U32StoreMem0Offset16_Ri),
            OpCode::U64LoadMem0Offset16_Rs => dec!(U64LoadMem0Offset16_Rs),
            OpCode::U32LoadMem0Offset16_Rs => dec!(U32LoadMem0Offset16_Rs),
            OpCode::I32WrapI64_Rr => dec!(I32WrapI64_Rr),
            OpCode::I32WrapI64_Rs => dec!(I32WrapI64_Rs),
            OpCode::F64LoadMem0Offset16_Rs => dec!(F64LoadMem0Offset16_Rs),
            OpCode::F32LoadMem0Offset16_Rs => dec!(F32LoadMem0Offset16_Rs),
            OpCode::F64StoreMem0Offset16_Rr => dec!(F64StoreMem0Offset16_Rr),
            OpCode::F32StoreMem0Offset16_Rr => dec!(F32StoreMem0Offset16_Rr),
            OpCode::U32LoadExtend8Mem0Offset16_Rs => dec!(U32LoadExtend8Mem0Offset16_Rs),
            OpCode::I32LoadExtend8Mem0Offset16_Rs => dec!(I32LoadExtend8Mem0Offset16_Rs),
            OpCode::U32LoadExtend16Mem0Offset16_Rs => dec!(U32LoadExtend16Mem0Offset16_Rs),
            OpCode::I32LoadExtend16Mem0Offset16_Rs => dec!(I32LoadExtend16Mem0Offset16_Rs),
            OpCode::I32StoreWrap8Mem0Offset16_Si => dec!(I32StoreWrap8Mem0Offset16_Si),
            OpCode::I32StoreWrap16Mem0Offset16_Si => dec!(I32StoreWrap16Mem0Offset16_Si),
            OpCode::I32StoreWrap8Mem0Offset16_Ss => dec!(I32StoreWrap8Mem0Offset16_Ss),
            OpCode::I32StoreWrap16Mem0Offset16_Ss => dec!(I32StoreWrap16Mem0Offset16_Ss),
            OpCode::I32StoreWrap8Mem0Offset16_Ri => dec!(I32StoreWrap8Mem0Offset16_Ri),
            OpCode::I32StoreWrap16Mem0Offset16_Ri => dec!(I32StoreWrap16Mem0Offset16_Ri),
            OpCode::I32Add_Rs_ri => dec!(I32Add_Rs_ri),
            OpCode::BranchI32NotEq_Si => dec!(BranchI32NotEq_Si),
            OpCode::BranchI32Eq_Si => dec!(BranchI32Eq_Si),
            OpCode::BranchI32Eq_Ri => dec!(BranchI32Eq_Ri),
            OpCode::BranchI64Eq_Ri => dec!(BranchI64Eq_Ri),
            OpCode::BranchU32Le_Rs => dec!(BranchU32Le_Rs),
            OpCode::I64Lt_Rri => dec!(I64Lt_Rri),
            OpCode::I64StoreWrap32Mem0Offset16_Rs => dec!(I64StoreWrap32Mem0Offset16_Rs),
            OpCode::I64Sext32_Rr => dec!(I64Sext32_Rr),
            OpCode::I64Sext32_Rs => dec!(I64Sext32_Rs),
            OpCode::I64Add_Rs_rs => dec!(I64Add_Rs_rs),
            OpCode::BranchI64Le_Si => dec!(BranchI64Le_Si),
            OpCode::BranchI32Lt_Si => dec!(BranchI32Lt_Si),
            OpCode::I32Add_Rs_ss => dec!(I32Add_Rs_ss),
            OpCode::BranchI64Lt_Ir => dec!(BranchI64Lt_Ir),
            OpCode::I64BitOr_Rss => dec!(I64BitOr_Rss),
            OpCode::BranchI32Le_Ss => dec!(BranchI32Le_Ss),
            OpCode::I32BitXor_Rss => dec!(I32BitXor_Rss),
            OpCode::I32BitAnd_Rss => dec!(I32BitAnd_Rss),
            OpCode::I32BitOr_Rss => dec!(I32BitOr_Rss),
            OpCode::I32BitXor_Rrs => dec!(I32BitXor_Rrs),
            OpCode::I32BitAnd_Rrs => dec!(I32BitAnd_Rrs),
            OpCode::I32BitOr_Rrs => dec!(I32BitOr_Rrs),
            OpCode::I32Sub_Rrs => dec!(I32Sub_Rrs),
            OpCode::I32Sub_Rsr => dec!(I32Sub_Rsr),
            OpCode::BranchI32Eq_Rs => dec!(BranchI32Eq_Rs),
            OpCode::BranchI32Eq_Ri => dec!(BranchI32Eq_Ri),
            OpCode::BranchI32Eq_Ss => dec!(BranchI32Eq_Ss),
            OpCode::BranchI32Eq_Si => dec!(BranchI32Eq_Si),
            OpCode::BranchI32NotEq_Rs => dec!(BranchI32NotEq_Rs),
            OpCode::BranchI32NotEq_Ri => dec!(BranchI32NotEq_Ri),
            OpCode::BranchI32NotEq_Ss => dec!(BranchI32NotEq_Ss),
            OpCode::BranchI32NotEq_Si => dec!(BranchI32NotEq_Si),
            OpCode::I64Add_Rrs => dec!(I64Add_Rrs),
            OpCode::I64Add_Rri => dec!(I64Add_Rri),
            OpCode::I64Add_Rss => dec!(I64Add_Rss),
            OpCode::I64Add_Rsi => dec!(I64Add_Rsi),
            OpCode::I64Add_Rs_rs => dec!(I64Add_Rs_rs),
            OpCode::I64Add_Rs_ri => dec!(I64Add_Rs_ri),
            OpCode::I64Add_Rs_ss => dec!(I64Add_Rs_ss),
            OpCode::I64Add_Rs_si => dec!(I64Add_Rs_si),
            OpCode::I64Sub_Rrs => dec!(I64Sub_Rrs),
            OpCode::I64Sub_Rsr => dec!(I64Sub_Rsr),
            OpCode::I64Sub_Rss => dec!(I64Sub_Rss),
            OpCode::I64Sub_Rir => dec!(I64Sub_Rir),
            OpCode::I64Sub_Ris => dec!(I64Sub_Ris),
            OpCode::BranchI64Eq_Rs => dec!(BranchI64Eq_Rs),
            OpCode::BranchI64Eq_Ri => dec!(BranchI64Eq_Ri),
            OpCode::BranchI64Eq_Ss => dec!(BranchI64Eq_Ss),
            OpCode::BranchI64Eq_Si => dec!(BranchI64Eq_Si),
            OpCode::BranchI64NotEq_Rs => dec!(BranchI64NotEq_Rs),
            OpCode::BranchI64NotEq_Ri => dec!(BranchI64NotEq_Ri),
            OpCode::BranchI64NotEq_Ss => dec!(BranchI64NotEq_Ss),
            OpCode::BranchI64NotEq_Si => dec!(BranchI64NotEq_Si),
            OpCode::I64ReinterpretF64_Rr => dec!(I64ReinterpretF64_Rr),
            OpCode::F64ReinterpretI64_Rr => dec!(F64ReinterpretI64_Rr),
            OpCode::U32Load_Ri => dec!(U32Load_Ri),
            OpCode::U64Load_Ri => dec!(U64Load_Ri),
            OpCode::BranchU64Lt_Is => dec!(BranchU64Lt_Is),
            OpCode::CallInternal => dec!(CallInternal),
            OpCode::Trap => dec!(Trap),
            OpCode::MemorySize => dec!(MemorySize),
            OpCode::U32LoadExtend8_Rr => dec!(U32LoadExtend8_Rr),
            OpCode::U32LoadExtend16_Ri => dec!(U32LoadExtend16_Ri),
            OpCode::U64Store_Is => dec!(U64Store_Is),
            OpCode::CallIndirect_S => dec!(CallIndirect_S),
            OpCode::BranchU32Lt_Ir => dec!(BranchU32Lt_Ir),
            OpCode::I64Add_Rs_ri => dec!(I64Add_Rs_ri),
            OpCode::U64LoadExtend32Mem0Offset16_Rs => dec!(U64LoadExtend32Mem0Offset16_Rs),
            OpCode::U32Store_Ir => dec!(U32Store_Ir),
            OpCode::U32Store_Ii => dec!(U32Store_Ii),
            OpCode::I64Lt_Rsr => dec!(I64Lt_Rsr),
            OpCode::BranchU32Lt_Rs => dec!(BranchU32Lt_Rs),
            OpCode::BranchU64Lt_Ir => dec!(BranchU64Lt_Ir),
            OpCode::BranchU32Lt_Ss => dec!(BranchU32Lt_Ss),
            OpCode::BranchTable_R => dec!(BranchTable_R),
            OpCode::BranchI32Lt_Ri => dec!(BranchI32Lt_Ri),
            OpCode::F64NotLe_Rss => dec!(F64NotLe_Rss),
            OpCode::U32Select_Rsii => dec!(U32Select_Rsii),
            OpCode::I32Eq_Rss => dec!(I32Eq_Rss),
            OpCode::BranchU64Lt_Si => dec!(BranchU64Lt_Si),
            OpCode::U64Shr_Rir => dec!(U64Shr_Rir),
            OpCode::I64Lt_Rss => dec!(I64Lt_Rss),
            OpCode::I32Shl_Rri => dec!(I32Shl_Rri),
            OpCode::U64Select_Rrrs => dec!(U64Select_Rrrs),
            OpCode::CallIndirect_R => dec!(CallIndirect_R),
            OpCode::ReturnCallIndirect_R => dec!(ReturnCallIndirect_R),
            OpCode::ReturnCallIndirect_S => dec!(ReturnCallIndirect_S),
            OpCode::ReturnCallInternal => dec!(ReturnCallInternal),
            OpCode::BranchU32Lt_Si => dec!(BranchU32Lt_Si),
            OpCode::U32LoadExtend8_Ri => dec!(U32LoadExtend8_Ri),
            OpCode::I32BitAnd_Rsi => dec!(I32BitAnd_Rsi),
            OpCode::U32Store_Is => dec!(U32Store_Is),
            OpCode::I64Eq_Rri => dec!(I64Eq_Rri),
            OpCode::BranchI64Lt_Rs => dec!(BranchI64Lt_Rs),
            OpCode::CallImported => dec!(CallImported),
            OpCode::MemoryCopy => dec!(MemoryCopy),
            OpCode::BranchI64Le_Rs => dec!(BranchI64Le_Rs),
            OpCode::BranchU32Lt_Sr => dec!(BranchU32Lt_Sr),
            OpCode::I64Shr_Rsi => dec!(I64Shr_Rsi),
            OpCode::BranchU32Le_Sr => dec!(BranchU32Le_Sr),
            OpCode::BranchU32Le_Ir => dec!(BranchU32Le_Ir),
            OpCode::I64BitOr_Rsi => dec!(I64BitOr_Rsi),
            OpCode::I32Eq_Rri => dec!(I32Eq_Rri),
            OpCode::U64LoadExtend8_Rs => dec!(U64LoadExtend8_Rs),
            OpCode::U64LoadExtend32_Ri => dec!(U64LoadExtend32_Ri),
            other => {
                let rest = &cursor[..cursor.len().min(16)];
                std::eprintln!("  @{pos:>3}: UNKNOWN {other:?}  next_bytes={rest:02x?}");
                std::eprintln!("[prepass-observe] STOP at first op outside subset");
                return;
            }
        }
        std::eprintln!("  @{pos:>3}: {code:?}");
    }
    std::eprintln!("[prepass-observe] end of stream (all ops in subset)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Engine, Module};

    const COUNTER_WAT: &str = r#"
        (module
            (func (export "count-via-locals") (param $n i32) (result i32)
                (loop $continue
                    (br_if
                        $continue
                        (local.tee $n
                            (i32.sub (local.get $n) (i32.const 1)))))
                (return (local.get $n))))
    "#;

    /// Compile a single-function `.wat` and return its [`MiniProgram`], or panic.
    fn compile_and_prepass(wat: &str) -> MiniProgram {
        let wasm = wat::parse_str(wat).expect("wat parse");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm[..]).expect("module");
        let ef = module
            .engine_func_by_index(0)
            .expect("engine func for defined function 0");
        engine
            .with_compiled_ops(ef, |ops, len_local_slots, len_stack_slots| {
                prepass(ops, len_local_slots, len_stack_slots)
            })
            .expect("function compiled")
            .expect("function is JIT-eligible")
    }

    /// A loop chaining i32 multiply then xor against local (slot) operands is
    /// JIT-eligible: the prepass lowers `I32Mul_Rrs` and `I32BitXor_Rrs` onto the
    /// existing two-slot i32 ops via the accumulator scratch-copy (and/or `_Rrs`
    /// already had dedicated arms).
    #[test]
    fn i32_slot_alu_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32)
                                   (param $b i32) (param $c i32) (result i32)
                    (local $i i32) (local $sum i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i32.add (local.get $sum)
                                    (i32.xor
                                        (i32.mul
                                            (i32.load
                                                (i32.add (local.get $ptr)
                                                         (i32.mul (local.get $i) (i32.const 4))))
                                            (local.get $b))
                                        (local.get $c))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_MUL_RS_WR),
            "must lower i32 mul-by-slot"
        );
        assert!(
            mp.words.contains(&MINI_I32_XOR_RS_WR),
            "must lower i32 xor-slot"
        );
    }

    /// A loop chaining i32 multiply / xor / or / and against folded constants over
    /// a loaded value is JIT-eligible: the prepass lowers `I32Mul_Rri`,
    /// `I32BitXor_Rri`, `I32BitOr_Rri`, `I32BitAnd_Rri` onto the existing two-slot
    /// i32 ops via operand pre-materialization (no dedicated kernel arm).
    #[test]
    fn i32_const_alu_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i32)
                    (local $i i32) (local $sum i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i32.add (local.get $sum)
                                    (i32.and
                                        (i32.or
                                            (i32.xor
                                                (i32.mul
                                                    (i32.load
                                                        (i32.add (local.get $ptr)
                                                                 (i32.mul (local.get $i) (i32.const 4))))
                                                    (i32.const 3))
                                                (i32.const 5))
                                            (i32.const 6))
                                        (i32.const 12))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_MUL_RS_WR),
            "must lower i32 mul-by-const"
        );
        assert!(
            mp.words.contains(&MINI_I32_XOR_RS_WR),
            "must lower i32 xor-const"
        );
        assert!(
            mp.words.contains(&MINI_I32_OR_RS_WR),
            "must lower i32 or-const"
        );
        assert!(
            mp.words.contains(&MINI_I32_AND_RS_WR),
            "must lower i32 and-const"
        );
    }

    /// A loop chaining i64 multiply / or / xor against local (slot) operands is
    /// JIT-eligible: the prepass lowers `I64Mul_Rrs`, `I64BitOr_Rrs`,
    /// `I64BitXor_Rrs` onto the existing two-slot ops by copying the accumulator
    /// into a scratch slot first (no dedicated kernel arm).
    #[test]
    fn i64_slot_alu_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32)
                                   (param $b i64) (param $c i64) (param $d i64) (result i64)
                    (local $i i32) (local $sum i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.xor
                                        (i64.or
                                            (i64.mul
                                                (i64.load
                                                    (i32.add (local.get $ptr)
                                                             (i32.mul (local.get $i) (i32.const 8))))
                                                (local.get $b))
                                            (local.get $c))
                                        (local.get $d))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_MUL_RS_WR),
            "must lower i64 mul-by-slot"
        );
        assert!(
            mp.words.contains(&MINI_I64_OR_RS_WR),
            "must lower i64 or-slot"
        );
        assert!(
            mp.words.contains(&MINI_I64_XOR_RS_WR),
            "must lower i64 xor-slot"
        );
    }

    /// A loop chaining i64 multiply / or / xor against folded constants over a
    /// loaded value is JIT-eligible: the prepass lowers `I64Mul_Rri`,
    /// `I64BitOr_Rri`, `I64BitXor_Rri` onto the existing two-slot ops via operand
    /// pre-materialization (no dedicated kernel arm).
    #[test]
    fn i64_const_alu_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $i i32) (local $sum i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.xor
                                        (i64.or
                                            (i64.mul
                                                (i64.load
                                                    (i32.add (local.get $ptr)
                                                             (i32.mul (local.get $i) (i32.const 8))))
                                                (i64.const 3))
                                            (i64.const 5))
                                        (i64.const 7))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_MUL_RS_WR),
            "must lower i64 mul-by-const"
        );
        assert!(
            mp.words.contains(&MINI_I64_OR_RS_WR),
            "must lower i64 or-const"
        );
        assert!(
            mp.words.contains(&MINI_I64_XOR_RS_WR),
            "must lower i64 xor-const"
        );
    }

    /// A loop summing the unsigned bytes of a buffer from linear memory is
    /// JIT-eligible: the prepass lowers the byte load
    /// (`U32LoadExtend8Mem0Offset16_Rr`) and the two-slot address add
    /// (`I32Add_Rss`).
    #[test]
    fn byte_sum_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $sum i64) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.extend_i32_u
                                        (i32.load8_u
                                            (i32.add (local.get $ptr) (local.get $i))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_U8_LOAD_MEM0_OFF),
            "must lower the unsigned byte load"
        );
    }

    /// A loop summing the signed bytes of a buffer from linear memory is
    /// JIT-eligible: the prepass lowers the signed byte load
    /// (`I32LoadExtend8Mem0Offset16_Rr`).
    #[test]
    fn signed_byte_sum_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $sum i64) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.extend_i32_s
                                        (i32.load8_s
                                            (i32.add (local.get $ptr) (local.get $i))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I8_LOAD_MEM0_OFF),
            "must lower the signed byte load"
        );
    }

    /// Loops summing 16-bit elements from linear memory are JIT-eligible: the
    /// prepass lowers both the unsigned (`U32LoadExtend16Mem0Offset16_Rr`) and
    /// signed (`I32LoadExtend16Mem0Offset16_Rr`) 16-bit loads.
    #[test]
    fn u16_loads_are_eligible() {
        let u16_wat = |load: &str, ext: &str| {
            alloc::format!(
                r#"
                (module
                    (memory 1)
                    (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                        (local $sum i64) (local $i i32)
                        (block $break
                            (loop $continue
                                (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                                (local.set $sum
                                    (i64.add (local.get $sum)
                                        ({ext}
                                            ({load}
                                                (i32.add (local.get $ptr)
                                                         (i32.mul (local.get $i) (i32.const 2)))))))
                                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                                (br $continue)))
                        (local.get $sum)))
            "#
            )
        };
        let mp_u = compile_and_prepass(&u16_wat("i32.load16_u", "i64.extend_i32_u"));
        assert!(
            mp_u.words.contains(&MINI_U16_LOAD_MEM0_OFF),
            "must lower the u16 load"
        );
        let mp_s = compile_and_prepass(&u16_wat("i32.load16_s", "i64.extend_i32_s"));
        assert!(
            mp_s.words.contains(&MINI_I16_LOAD_MEM0_OFF),
            "must lower the i16 load"
        );
    }

    /// Observe (does not assert) the real op stream of `count-via-locals`.
    #[test]
    /// A loop writing an i32 to linear memory each iteration (a buffer fill) is
    /// JIT-eligible: the prepass lowers the 32-bit store
    /// (`U32StoreMem0Offset16_Sr`, pointer in a slot, value in the accumulator).
    #[test]
    fn i32_store_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 4)))
                                (i32.mul (local.get $i) (local.get $i)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_STORE_SR),
            "must lower the i32 store",
        );
    }

    /// A loop storing a local (not a computed accumulator value) to linear
    /// memory is JIT-eligible: the address stays in the accumulator and the
    /// prepass lowers the slot-value 32-bit store (`U32StoreMem0Offset16_Rs`).
    #[test]
    fn i32_store_local_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 4)))
                                (local.get $i))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_STORE_RS),
            "must lower the slot-value i32 store",
        );
    }

    /// A loop storing a 64-bit local to linear memory is JIT-eligible: the
    /// address stays in the accumulator and the prepass lowers the slot-value
    /// 64-bit store (`U64StoreMem0Offset16_Rs`).
    #[test]
    fn i64_store_local_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32)
                    (local $i i32) (local $acc i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i64.add (local.get $acc) (i64.const 1)))
                            (i64.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 8)))
                                (local.get $acc))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_STORE_RS),
            "must lower the slot-value i64 store",
        );
    }

    /// Loops writing narrow (8- and 16-bit, wrapping) locals to linear memory are
    /// JIT-eligible: the prepass lowers `I32StoreWrap8Mem0Offset16_Rs` and
    /// `I32StoreWrap16Mem0Offset16_Rs`.
    #[test]
    fn narrow_stores_are_eligible() {
        let narrow_wat = |store: &str, stride: i32| {
            alloc::format!(
                r#"
                (module
                    (memory 1)
                    (func (export "f") (param $ptr i32) (param $n i32)
                        (local $i i32)
                        (block $break
                            (loop $continue
                                (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                                ({store}
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const {stride})))
                                    (local.get $i))
                                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                                (br $continue)))))
            "#
            )
        };
        let mp8 = compile_and_prepass(&narrow_wat("i32.store8", 1));
        assert!(
            mp8.words.contains(&MINI_I32_STORE8_RS),
            "must lower i32.store8"
        );
        let mp16 = compile_and_prepass(&narrow_wat("i32.store16", 2));
        assert!(
            mp16.words.contains(&MINI_I32_STORE16_RS),
            "must lower i32.store16",
        );
    }

    /// Storing a COMPUTED value (the value lands in the accumulator, forcing the
    /// address into a slot) is JIT-eligible for i64 / 8-bit / 16-bit stores: the
    /// prepass lowers the `_Sr` store forms.
    #[test]
    fn computed_value_stores_are_eligible() {
        // i64.store of acc+acc (computed i64 -> accumulator).
        const I64_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32)
                    (local $i i32) (local $acc i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i64.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 8)))
                                (i64.add (local.get $acc) (local.get $acc)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;
        let mp = compile_and_prepass(I64_WAT);
        assert!(
            mp.words.contains(&MINI_I64_STORE_SR),
            "must lower the computed-value i64 store",
        );

        // Narrow stores of i*i (computed i32 -> accumulator).
        let narrow_wat = |store: &str, stride: i32| {
            alloc::format!(
                r#"
                (module
                    (memory 1)
                    (func (export "f") (param $ptr i32) (param $n i32)
                        (local $i i32)
                        (block $break
                            (loop $continue
                                (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                                ({store}
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const {stride})))
                                    (i32.mul (local.get $i) (local.get $i)))
                                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                                (br $continue)))))
            "#
            )
        };
        let mp8 = compile_and_prepass(&narrow_wat("i32.store8", 1));
        assert!(
            mp8.words.contains(&MINI_I32_STORE8_SR),
            "must lower the computed-value i32.store8",
        );
        let mp16 = compile_and_prepass(&narrow_wat("i32.store16", 2));
        assert!(
            mp16.words.contains(&MINI_I32_STORE16_SR),
            "must lower the computed-value i32.store16",
        );
    }

    /// A loop counting array elements below a threshold (a signed i32 compare
    /// result used as a 0/1 value, not a branch) is JIT-eligible: the prepass
    /// lowers the accumulator-vs-slot compare-value op (`I32Lt_Rrs`).
    #[test]
    fn compare_value_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t i32) (result i32)
                    (local $i i32) (local $count i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (i32.lt_s
                                        (i32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4))))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_LT_RS_R),
            "must lower the compare-value op",
        );
    }

    /// A loop counting array elements ABOVE a threshold (`mem[i] > t`, which wasmi
    /// lowers to `t < mem[i]` with the loaded value in the accumulator) is
    /// JIT-eligible: the prepass lowers the slot-vs-accumulator compare-value op
    /// (`I32Lt_Rsr`).
    #[test]
    fn compare_value_swapped_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t i32) (result i32)
                    (local $i i32) (local $count i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (i32.gt_s
                                        (i32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4))))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_LT_SR_R),
            "must lower the swapped compare-value op",
        );
    }

    /// A running-max loop using `select` (branchless `cond ? a : b`) is
    /// JIT-eligible: the prepass lowers the select op.
    #[test]
    fn select_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i32)
                    (local $i i32) (local $max i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $max
                                (select
                                    (i32.load
                                        (i32.add (local.get $ptr)
                                                 (i32.mul (local.get $i) (i32.const 4))))
                                    (local.get $max)
                                    (i32.gt_s
                                        (i32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4))))
                                        (local.get $max))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $max)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(mp.words.contains(&MINI_SELECT), "must lower the select op");
    }

    /// Selects with constant arms are JIT-eligible: the prepass materializes
    /// constants into scratch slots, then uses the normal branchless select.
    #[test]
    fn select_const_arms_are_eligible() {
        const I32_FALSE_CONST: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $i i32) (local $acc i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (select
                                        (local.get $i)
                                        (i32.const 7)
                                        (i32.lt_s (local.get $i) (i32.const 5)))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        const I32_TRUE_CONST: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $i i32) (local $acc i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (select
                                        (i32.const 7)
                                        (local.get $i)
                                        (i32.lt_s (local.get $i) (i32.const 5)))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        const I32_BOTH_CONST: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $i i32) (local $acc i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (select
                                        (i32.const 11)
                                        (i32.const 7)
                                        (i32.lt_s (local.get $i) (i32.const 5)))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        const I64_FALSE_CONST: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $i i64) (local $acc i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (select
                                        (local.get $i)
                                        (i64.const 7)
                                        (i64.lt_s (local.get $i) (i64.const 5)))))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        for wat in [I32_FALSE_CONST, I32_TRUE_CONST, I32_BOTH_CONST] {
            let mp = compile_and_prepass(wat);
            assert!(
                mp.words.contains(&MINI_SELECT),
                "must lower const-arm i32 select"
            );
        }

        let mp = compile_and_prepass(I64_FALSE_CONST);
        assert!(
            mp.words.contains(&MINI_SELECT),
            "must lower const-arm i64 select"
        );
        assert!(
            mp.words.contains(&MINI_I64_LT_SI_R),
            "must lower i64.lt_s(slot, const)"
        );
    }

    /// Slot-vs-const compare VALUE forms across i32/i64 compare operators are
    /// JIT-eligible; wasmi may rewrite gt/ge to swapped lt/le forms.
    #[test]
    fn cmp_value_const_forms_are_eligible() {
        for ty in ["i32", "i64"] {
            for cmp in [
                "eq", "ne", "lt_s", "le_s", "gt_s", "ge_s", "lt_u", "le_u", "gt_u", "ge_u",
            ] {
                let cmp_expr = std::format!("({ty}.{cmp} (local.get $i) ({ty}.const 5))");
                let cmp_expr = if ty == "i64" {
                    std::format!("(i64.extend_i32_u {cmp_expr})")
                } else {
                    cmp_expr
                };
                let wat = std::format!(
                    r#"
            (module
                (func (export "f") (param $n {ty}) (result {ty})
                    (local $i {ty}) (local $acc {ty})
                    (block $break
                        (loop $continue
                            (br_if $break ({ty}.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                ({ty}.add (local.get $acc) {cmp_expr}))
                            (local.set $i ({ty}.add (local.get $i) ({ty}.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#,
                );
                compile_and_prepass(&wat);
            }
        }
    }

    /// A running f64 max using `f64.gt` is JIT-eligible: wasmi rewrites gt into a
    /// swapped `F64Lt_Rsr`, which lowers through the existing compare selector arm.
    #[test]
    fn f64_swapped_cmp_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $scratch i32) (result f64)
                    (local $i i32) (local $v f64) (local $max f64) (local $v_bits i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $max
                                (f64.load (i32.add (local.get $scratch) (i32.const 0))))
                            (local.set $v
                                (f64.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 8)))))
                            (local.set $v_bits
                                (i64.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 8)))))
                            (i64.store
                                (i32.add (local.get $scratch) (i32.const 0))
                                (select
                                    (local.get $v_bits)
                                    (i64.load (i32.add (local.get $scratch) (i32.const 0)))
                                    (f64.gt (local.get $v) (local.get $max))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (f64.load (i32.add (local.get $scratch) (i32.const 0)))))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_F64_CMP_RS_R),
            "must lower f64.gt via the f64 compare selector arm"
        );
        assert!(
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_CMP_RS_R && w[1] == 4),
            "must use the f64 swapped lt selector"
        );
    }

    /// Running f32 max loops using `f32.gt` / `f32.ge` are JIT-eligible: wasmi
    /// rewrites gt/ge into swapped `F32Lt_Rsr` / `F32Le_Rsr`.
    #[test]
    fn f32_swapped_cmp_is_eligible() {
        const GT_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $scratch i32) (result f32)
                    (local $i i32) (local $v f32) (local $max f32) (local $v_bits i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $max
                                (f32.load (i32.add (local.get $scratch) (i32.const 0))))
                            (local.set $v
                                (f32.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 4)))))
                            (local.set $v_bits
                                (i32.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 4)))))
                            (i32.store
                                (i32.add (local.get $scratch) (i32.const 0))
                                (select
                                    (local.get $v_bits)
                                    (i32.load (i32.add (local.get $scratch) (i32.const 0)))
                                    (f32.gt (local.get $v) (local.get $max))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (f32.load (i32.add (local.get $scratch) (i32.const 0)))))
        "#;
        const GE_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $scratch i32) (result f32)
                    (local $i i32) (local $v f32) (local $max f32) (local $v_bits i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $max
                                (f32.load (i32.add (local.get $scratch) (i32.const 0))))
                            (local.set $v
                                (f32.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 4)))))
                            (local.set $v_bits
                                (i32.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 4)))))
                            (i32.store
                                (i32.add (local.get $scratch) (i32.const 0))
                                (select
                                    (local.get $v_bits)
                                    (i32.load (i32.add (local.get $scratch) (i32.const 0)))
                                    (f32.ge (local.get $v) (local.get $max))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (f32.load (i32.add (local.get $scratch) (i32.const 0)))))
        "#;
        let has_cmp = |mp: &MiniProgram, sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F32_CMP_RS_R && w[1] == sel)
        };
        let gt = compile_and_prepass(GT_WAT);
        assert!(
            gt.words.contains(&MINI_F32_CMP_RS_R),
            "must lower f32.gt via the f32 compare selector arm"
        );
        assert!(has_cmp(&gt, 4), "must use the f32 swapped lt selector");
        let ge = compile_and_prepass(GE_WAT);
        assert!(
            ge.words.contains(&MINI_F32_CMP_RS_R),
            "must lower f32.ge via the f32 compare selector arm"
        );
        assert!(has_cmp(&ge, 5), "must use the f32 swapped le selector");
    }

    /// Loops using i32 `==` / `!=` as a 0/1 VALUE (counting occurrences /
    /// non-matches) are JIT-eligible: the prepass lowers the equality and
    /// inequality compare-value ops (`I32Eq_Rrs` / `I32NotEq_Rrs`).
    #[test]
    fn equality_value_is_eligible() {
        // `mem[i] == t` counted as a value.
        const EQ_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t i32) (result i32)
                    (local $i i32) (local $count i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (i32.eq
                                        (i32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4))))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let eq = compile_and_prepass(EQ_WAT);
        assert!(
            eq.words.contains(&MINI_I32_EQ_RS_R),
            "must lower i32.eq value"
        );

        // Same loop with `!=`.
        let ne_wat = EQ_WAT.replace("(i32.eq", "(i32.ne");
        let ne = compile_and_prepass(&ne_wat);
        assert!(
            ne.words.contains(&MINI_I32_NE_RS_R),
            "must lower i32.ne value"
        );
    }

    /// Loops using i64 `==` / `!=` / `<` as a 0/1 VALUE over an i64 array are
    /// JIT-eligible: the prepass lowers the full-i64 compare-value ops
    /// (`I64Eq_Rrs` / `I64NotEq_Rrs` / `I64Lt_Rrs`).
    #[test]
    fn i64_compare_value_is_eligible() {
        const EQ_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t i64) (result i32)
                    (local $i i32) (local $count i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (i64.eq
                                        (i64.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 8))))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let eq = compile_and_prepass(EQ_WAT);
        assert!(
            eq.words.contains(&MINI_I64_EQ_RS_R),
            "must lower i64.eq value"
        );

        let ne = compile_and_prepass(&EQ_WAT.replace("(i64.eq", "(i64.ne"));
        assert!(
            ne.words.contains(&MINI_I64_NE_RS_R),
            "must lower i64.ne value"
        );

        let lt = compile_and_prepass(&EQ_WAT.replace("(i64.eq", "(i64.lt_s"));
        assert!(
            lt.words.contains(&MINI_I64_LT_RS_R),
            "must lower i64.lt value"
        );
    }

    /// Loops summing ONE saturating f64→int truncation of the loaded f64 are
    /// JIT-eligible (the trunc reads the load result directly, in the
    /// accumulator). The prepass lowers `I64TruncSatF64_Rr` / `U64TruncSatF64_Rr`
    /// and the i32 variants `I32TruncSatF64_Rr` / `U32TruncSatF64_Rr`.
    #[test]
    fn f64_trunc_sat_is_eligible() {
        const I64S_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $i i32) (local $sum i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.trunc_sat_f64_s
                                        (f64.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 8)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let sat = |mp: &MiniProgram, sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_TRUNC_SAT_S && w[1] == sel)
        };
        let i64s = compile_and_prepass(I64S_WAT);
        assert!(sat(&i64s, 2), "must lower trunc_sat_i64_s (selector 2)");
        let i64u =
            compile_and_prepass(&I64S_WAT.replace("i64.trunc_sat_f64_s", "i64.trunc_sat_f64_u"));
        assert!(sat(&i64u, 3), "must lower trunc_sat_i64_u (selector 3)");

        // The i32 variants accumulate into an i32 sum.
        let i32s_wat = I64S_WAT
            .replace("(local $sum i64)", "(local $sum i32)")
            .replace("(result i64)", "(result i32)")
            .replace("i64.add", "i32.add")
            .replace("i64.trunc_sat_f64_s", "i32.trunc_sat_f64_s");
        let i32s = compile_and_prepass(&i32s_wat);
        assert!(sat(&i32s, 0), "must lower trunc_sat_i32_s (selector 0)");
        let i32u =
            compile_and_prepass(&i32s_wat.replace("i32.trunc_sat_f64_s", "i32.trunc_sat_f64_u"));
        assert!(sat(&i32u, 1), "must lower trunc_sat_i32_u (selector 1)");
    }

    /// Loops summing ONE trapping f64→int truncation of the loaded f64 are
    /// JIT-eligible (the trunc reads the load result in the accumulator). The
    /// prepass lowers `I64TruncF64_Rr` / `U64TruncF64_Rr` and the i32 variants
    /// `I32TruncF64_Rr` / `U32TruncF64_Rr`.
    #[test]
    fn f64_trunc_is_eligible() {
        const I64S_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $i i32) (local $sum i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.trunc_f64_s
                                        (f64.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 8)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let trunc = |mp: &MiniProgram, sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_TRUNC_S && w[1] == sel)
        };
        let i64s = compile_and_prepass(I64S_WAT);
        assert!(trunc(&i64s, 2), "must lower trunc_i64_s (selector 2)");
        let i64u = compile_and_prepass(&I64S_WAT.replace("i64.trunc_f64_s", "i64.trunc_f64_u"));
        assert!(trunc(&i64u, 3), "must lower trunc_i64_u (selector 3)");

        let i32s_wat = I64S_WAT
            .replace("(local $sum i64)", "(local $sum i32)")
            .replace("(result i64)", "(result i32)")
            .replace("i64.add", "i32.add")
            .replace("i64.trunc_f64_s", "i32.trunc_f64_s");
        let i32s = compile_and_prepass(&i32s_wat);
        assert!(trunc(&i32s, 0), "must lower trunc_i32_s (selector 0)");
        let i32u = compile_and_prepass(&i32s_wat.replace("i32.trunc_f64_s", "i32.trunc_f64_u"));
        assert!(trunc(&i32u, 1), "must lower trunc_i32_u (selector 1)");
    }

    /// A loop summing widening int→f64 conversions of an i32 local and an i64
    /// local is JIT-eligible: the prepass lowers `F64ConvertI32_Rs` /
    /// `F64ConvertU32_Rs` / `F64ConvertI64_Rs` / `F64ConvertU64_Rs`.
    #[test]
    fn f64_convert_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (param $j i64) (result f64)
                    (local $i i32) (local $sum f64) (local $x i32) (local $y i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $x (i32.sub (local.get $i) (i32.const 500)))
                            (local.set $y (i64.add (local.get $j)
                                                   (i64.extend_i32_s (local.get $i))))
                            (local.set $sum
                                (f64.add (local.get $sum)
                                    (f64.add
                                        (f64.convert_i32_s (local.get $x))
                                        (f64.add
                                            (f64.convert_i32_u (local.get $x))
                                            (f64.add
                                                (f64.convert_i64_s (local.get $y))
                                                (f64.convert_i64_u (local.get $y)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        let cvt = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_CVT_S && w[1] == sel)
        };
        assert!(cvt(0), "must lower convert_i32_s (selector 0)");
        assert!(cvt(1), "must lower convert_i32_u (selector 1)");
        assert!(cvt(2), "must lower convert_i64_s (selector 2)");
        assert!(cvt(3), "must lower convert_i64_u (selector 3)");
    }

    /// An f64 array-sum loop (float load via the i64-load residual + float add via
    /// the `f64_arith` residual + 64-bit copy + 64-bit return) is JIT-eligible: the
    /// prepass lowers `F64LoadMem0Offset16_Rr`, `F64Add_Rsr`, `F64Copy_S{N}r`.
    #[test]
    fn f64_sum_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result f64)
                    (local $i i32) (local $sum f64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f64.add (local.get $sum)
                                    (f64.load
                                        (i32.add (local.get $ptr)
                                                 (i32.mul (local.get $i) (i32.const 8))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_ARITH_RS && w[1] == 0),
            "must lower the f64 add (arith selector 0)"
        );
    }

    /// A clamp loop `min(max(a[i], lo_local), hi_const)` is JIT-eligible: the
    /// prepass lowers `F64Max_Rrs` (slot form) and `F64Min_Rri` (constant form)
    /// through the `MINI_F64_MINMAX_RS` selector arm (sel 1=max, 0=min).
    #[test]
    fn f64_minmax_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $lo f64) (result f64)
                    (local $i i32) (local $sum f64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f64.add (local.get $sum)
                                    (f64.min
                                        (f64.max
                                            (f64.load
                                                (i32.add (local.get $ptr)
                                                         (i32.mul (local.get $i) (i32.const 8))))
                                            (local.get $lo))
                                        (f64.const 100.0))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        let has = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_MINMAX_RS && w[1] == sel)
        };
        assert!(has(1), "must lower f64.max (slot) as minmax selector 1");
        assert!(has(0), "must lower f64.min (const) as minmax selector 0");
    }

    /// An f64 loop using the unary ops `sqrt(abs(a[i]))` plus `floor`/`ceil` is
    /// JIT-eligible: the prepass lowers `F64{Abs,Sqrt,Floor,Ceil}_R{r,s}` through the
    /// `MINI_F64_UNARY_S` selector arm (sel 0=abs, 2=sqrt, 3=ceil, 4=floor).
    #[test]
    fn f64_unary_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result f64)
                    (local $i i32) (local $sum f64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f64.add (local.get $sum)
                                    (f64.floor
                                        (f64.ceil
                                            (f64.sqrt
                                                (f64.abs
                                                    (f64.load
                                                        (i32.add (local.get $ptr)
                                                                 (i32.mul (local.get $i) (i32.const 8))))))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        let has = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_UNARY_S && w[1] == sel)
        };
        assert!(has(0), "must lower f64.abs (unary selector 0)");
        assert!(has(2), "must lower f64.sqrt (unary selector 2)");
        assert!(has(3), "must lower f64.ceil (unary selector 3)");
        assert!(has(4), "must lower f64.floor (unary selector 4)");
    }

    /// An i32 loop summing `clz(i) + ctz(i) + popcnt(i)` of a local is
    /// JIT-eligible: the prepass lowers `I32{Clz,Ctz,Popcnt}_Rs` onto the
    /// slot-form bit-count MINI ops.
    #[test]
    fn i32_bitcount_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.add (local.get $acc) (i32.clz (local.get $i))))
                            (local.set $acc (i32.add (local.get $acc) (i32.ctz (local.get $i))))
                            (local.set $acc (i32.add (local.get $acc) (i32.popcnt (local.get $i))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        let has = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_I32_BITCOUNT_S && w[1] == sel)
        };
        assert!(has(0), "must lower i32.clz (sel 0)");
        assert!(has(1), "must lower i32.ctz (sel 1)");
        assert!(has(2), "must lower i32.popcnt (sel 2)");
    }

    /// An i64 loop summing `clz/ctz/popcnt` of a computed value (`i * 3`) is
    /// JIT-eligible: the accumulator-input `I64{Clz,Ctz,Popcnt}_Rr` forms lower
    /// via the `MINI_COPY_SR` scratch-copy onto the slot-form bit-count MINI ops.
    #[test]
    fn i64_bitcount_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $acc i64) (local $i i64)
                    (local.set $i (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $i)))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.clz (i64.mul (local.get $i) (local.get $i)))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.ctz (i64.mul (local.get $i) (local.get $i)))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.popcnt (i64.mul (local.get $i) (local.get $i)))))
                            (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        let has = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_I64_BITCOUNT_S && w[1] == sel)
        };
        assert!(has(0), "must lower i64.clz (sel 0)");
        assert!(has(1), "must lower i64.ctz (sel 1)");
        assert!(has(2), "must lower i64.popcnt (sel 2)");
    }

    /// An i32 loop that divides a running value by a runtime divisor with all four
    /// operators (`div_s`, `div_u`, `rem_s`, `rem_u`) is JIT-eligible: the prepass
    /// lowers each onto its two-slot div/rem MINI op.
    #[test]
    fn i32_divrem_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (param $d i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.div_s (local.get $i) (local.get $d))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.div_u (local.get $i) (local.get $d))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.rem_s (local.get $i) (local.get $d))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.rem_u (local.get $i) (local.get $d))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(mp.words.contains(&MINI_I32_DIV_S), "must lower i32.div_s");
        assert!(mp.words.contains(&MINI_I32_DIV_U), "must lower i32.div_u");
        assert!(mp.words.contains(&MINI_I32_REM_S), "must lower i32.rem_s");
        assert!(mp.words.contains(&MINI_I32_REM_U), "must lower i32.rem_u");
    }

    /// The i64 counterpart: a countdown loop dividing by a runtime divisor with
    /// all four 64-bit operators is JIT-eligible.
    #[test]
    fn i64_divrem_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (param $d i64) (result i64)
                    (local $acc i64) (local $i i64)
                    (local.set $i (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $i)))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.div_s (local.get $i) (local.get $d))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.div_u (local.get $i) (local.get $d))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.rem_s (local.get $i) (local.get $d))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.rem_u (local.get $i) (local.get $d))))
                            (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(mp.words.contains(&MINI_I64_DIV_S), "must lower i64.div_s");
        assert!(mp.words.contains(&MINI_I64_DIV_U), "must lower i64.div_u");
        assert!(mp.words.contains(&MINI_I64_REM_S), "must lower i64.rem_s");
        assert!(mp.words.contains(&MINI_I64_REM_U), "must lower i64.rem_u");
    }

    /// Division by an immediate divisor (`x / const`, a `NonZero` `_Rsi`/`_Rri`
    /// operand) and by an immediate dividend (`const / x`, a `_Ris`/`_Rir`
    /// operand) are both JIT-eligible: the prepass pre-materializes the immediate
    /// into a scratch slot before the two-slot div/rem MINI op.
    #[test]
    fn divrem_immediate_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.div_s (local.get $i) (i32.const 3))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.rem_u (local.get $i) (i32.const 7))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.div_s (i32.const 1000) (local.get $n))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_DIV_S),
            "must lower i32.div_s (immediate)"
        );
        assert!(
            mp.words.contains(&MINI_I32_REM_U),
            "must lower i32.rem_u (immediate)"
        );
        // The immediate operands are pre-materialized via slot-immediate copies.
        assert!(
            mp.words.contains(&MINI_COPY_SI),
            "must pre-materialize the immediate"
        );
    }

    /// A loop using `x + const` in value position (the result feeds another op
    /// rather than a `local.set`) is JIT-eligible: the prepass lowers the
    /// `I32/I64 Add_Rsi` (slot + imm) and `_Rri` (accumulator + imm) forms by
    /// pre-materializing the immediate and reusing the two-operand add MINI ops.
    /// Without those arms `compile_and_prepass` would panic (function ineligible).
    #[test]
    fn add_immediate_value_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i64)
                    (local $acc i64) (local $i i32) (local $j i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            ;; (i + 3) in value position -> I32Add_Rsi
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.extend_i32_s
                                    (i32.mul (i32.add (local.get $i) (i32.const 3))
                                             (local.get $i)))))
                            ;; (i*i) + 5 in value position -> I32Add_Rri
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.extend_i32_s
                                    (i32.add (i32.mul (local.get $i) (local.get $i))
                                             (i32.const 5)))))
                            ;; (j + 7) in value position -> I64Add_Rsi
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.mul (i64.add (local.get $j) (i64.const 7))
                                         (local.get $j))))
                            ;; (j*j) + 11 in value position -> I64Add_Rri
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.add (i64.mul (local.get $j) (local.get $j))
                                         (i64.const 11))))
                            (local.set $j (i64.add (local.get $j) (i64.const 1)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        // The `.expect("JIT-eligible")` inside is the assertion: any unlowered
        // add form would make the whole function bail to the stock executor.
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_ADD_SS_WB) && mp.words.contains(&MINI_I64_ADD_SS_WB),
            "the value-position adds lower onto the two-slot add MINI ops"
        );
    }

    /// Value-position `sub` in its remaining operand shapes — `a - b` (slot-slot),
    /// `a - <computed>` (slot - accumulator), `const - x` (imm - slot / imm -
    /// accumulator) — is JIT-eligible: each pre-materializes accumulator/immediate
    /// operands into scratch slots (order-preserving) and reuses the two-slot sub.
    #[test]
    fn sub_value_forms_are_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i64)
                    (local $acc i64) (local $i i32) (local $j i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            ;; n - i (slot - slot) -> I32Sub_Rss
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.extend_i32_s (i32.sub (local.get $n) (local.get $i)))))
                            ;; i - (i*i) (slot - acc) -> I32Sub_Rsr
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.extend_i32_s
                                    (i32.sub (local.get $i) (i32.mul (local.get $i) (local.get $i))))))
                            ;; 1000 - i (imm - slot) -> I32Sub_Ris
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.extend_i32_s (i32.sub (i32.const 1000) (local.get $i)))))
                            ;; 500 - (i*i) (imm - acc) -> I32Sub_Rir
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.extend_i32_s
                                    (i32.sub (i32.const 500) (i32.mul (local.get $i) (local.get $i))))))
                            ;; (j*j) - j (acc - slot) -> I64Sub_Rrs
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.sub (i64.mul (local.get $j) (local.get $j)) (local.get $j))))
                            ;; 900 - j (imm - slot) -> I64Sub_Ris
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.sub (i64.const 900) (local.get $j))))
                            (local.set $j (i64.add (local.get $j) (i64.const 1)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_SUB_SS_WR) && mp.words.contains(&MINI_I64_SUB_SS_WR),
            "the value-position subs lower onto the two-slot sub MINI ops"
        );
    }

    /// An i64 `j * const` (`I64Mul_Rsi`, a full i64 immediate) is JIT-eligible,
    /// mirroring the already-supported `I32Mul_Rsi` — pre-materialize the
    /// immediate into a scratch slot, then the two-slot i64 multiply.
    #[test]
    fn i64_mul_immediate_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $acc i64) (local $j i64)
                    (local.set $j (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $j)))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.mul (local.get $j) (i64.const 1000003))))
                            (local.set $j (i64.sub (local.get $j) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_MUL_SS_WR),
            "must lower i64 slot * immediate"
        );
    }

    /// An f64 loop combining the four arithmetic ops against folded constants
    /// (`((a[i] * 2.0) - 1.0) / 4.0 + 0.5`) is JIT-eligible: the prepass lowers the
    /// fused-immediate forms `F64{Mul,Sub,Div,Add}_Rri` via pre-materialization.
    #[test]
    fn f64_const_arith_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result f64)
                    (local $i i32) (local $sum f64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f64.add (local.get $sum)
                                    (f64.add
                                        (f64.div
                                            (f64.sub
                                                (f64.mul
                                                    (f64.load
                                                        (i32.add (local.get $ptr)
                                                                 (i32.mul (local.get $i) (i32.const 8))))
                                                    (f64.const 2.0))
                                                (f64.const 1.0))
                                            (f64.const 4.0))
                                        (f64.const 0.5))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        // The mul/sub/div/add-immediate forms all pre-materialize into a slot then
        // reuse the arith selector arm, so `MINI_F64_ARITH_RS` with each selector
        // (0=add, 1=sub, 2=mul, 3=div) must be present.
        let has = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_ARITH_RS && w[1] == sel)
        };
        assert!(has(2), "must lower f64 mul-const (selector 2)");
        assert!(has(1), "must lower f64 sub-const (selector 1)");
        assert!(has(3), "must lower f64 div-const (selector 3)");
        assert!(has(0), "must lower f64 add-const (selector 0)");
    }

    /// Loops using f64 `<` / `<=` / `==` / `!=` as 0/1 VALUES (counting elements
    /// matching a float predicate) are JIT-eligible: the prepass lowers
    /// `F64Lt_Rrs`, `F64Le_Rrs`, `F64Eq_Rrs`, `F64NotEq_Rrs`.
    #[test]
    fn f64_compare_value_is_eligible() {
        const LT_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t f64) (result i32)
                    (local $i i32) (local $count i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (f64.lt
                                        (f64.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 8))))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let has_cmp = |mp: &MiniProgram, sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_CMP_RS_R && w[1] == sel)
        };
        let lt = compile_and_prepass(LT_WAT);
        assert!(has_cmp(&lt, 0), "must lower f64.lt value (selector 0)");
        let le = compile_and_prepass(&LT_WAT.replace("(f64.lt", "(f64.le"));
        assert!(has_cmp(&le, 1), "must lower f64.le value (selector 1)");
        let eq = compile_and_prepass(&LT_WAT.replace("(f64.lt", "(f64.eq"));
        assert!(has_cmp(&eq, 2), "must lower f64.eq value (selector 2)");
        let ne = compile_and_prepass(&LT_WAT.replace("(f64.lt", "(f64.ne"));
        assert!(has_cmp(&ne, 3), "must lower f64.ne value (selector 3)");
    }

    /// An f64 transform loop `out[i] = a[i] * scale` (f64 load + mul + f64 STORE)
    /// is JIT-eligible: the prepass lowers `F64StoreMem0Offset16_Sr` onto the
    /// bit-identical i64 store.
    #[test]
    fn f64_store_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $src i32) (param $dst i32) (param $n i32) (param $s f64)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (f64.store
                                (i32.add (local.get $dst)
                                         (i32.mul (local.get $i) (i32.const 8)))
                                (f64.mul
                                    (f64.load
                                        (i32.add (local.get $src)
                                                 (i32.mul (local.get $i) (i32.const 8))))
                                    (local.get $s)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_F64_LOAD_MEM0_OFF),
            "must lower the f64 load into the f64 accumulator"
        );
        assert!(
            mp.words.contains(&MINI_F64_STORE_SR),
            "must lower the f64 store from the f64 accumulator"
        );
    }

    /// An f64 loop combining sub / mul / div is JIT-eligible: the prepass lowers
    /// `F64Sub_Rrs`, `F64Mul_Rrs`, `F64Div_Rrs`.
    #[test]
    fn f64_arith_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t f64) (result f64)
                    (local $i i32) (local $acc f64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (f64.add (local.get $acc)
                                    (f64.div
                                        (f64.mul
                                            (f64.sub
                                                (f64.load
                                                    (i32.add (local.get $ptr)
                                                             (i32.mul (local.get $i) (i32.const 8))))
                                                (local.get $t))
                                            (local.get $t))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        let has = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F64_ARITH_RS && w[1] == sel)
        };
        assert!(has(1), "must lower f64 sub (selector 1)");
        assert!(has(2), "must lower f64 mul (selector 2)");
        assert!(has(3), "must lower f64 div (selector 3)");
    }

    /// An f32 store transform `out[i] = a[i] * scale` is JIT-eligible: the prepass
    /// lowers `F32LoadMem0Offset16_Rr` and `F32StoreMem0Offset16_Sr`.
    #[test]
    fn f32_store_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $src i32) (param $dst i32) (param $n i32) (param $s f32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (f32.store
                                (i32.add (local.get $dst)
                                         (i32.mul (local.get $i) (i32.const 4)))
                                (f32.mul
                                    (f32.load
                                        (i32.add (local.get $src)
                                                 (i32.mul (local.get $i) (i32.const 4))))
                                    (local.get $s)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_F32_LOAD_MEM0_OFF),
            "must lower the f32 load into the f32 accumulator"
        );
        assert!(
            mp.words.contains(&MINI_F32_STORE_SR),
            "must lower the f32 store from the f32 accumulator"
        );
    }

    /// An f32 loop combining add / sub / mul / div is JIT-eligible: the prepass
    /// lowers `F32Add_Rsr`, `F32Sub_Rrs`, `F32Mul_Rrs`, `F32Div_Rrs` — all through
    /// the selector-dispatched `MINI_F32_ARITH_RS` arm.
    #[test]
    fn f32_arith_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t f32) (result f32)
                    (local $i i32) (local $acc f32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (f32.add (local.get $acc)
                                    (f32.div
                                        (f32.mul
                                            (f32.sub
                                                (f32.load
                                                    (i32.add (local.get $ptr)
                                                             (i32.mul (local.get $i) (i32.const 4))))
                                                (local.get $t))
                                            (local.get $t))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        // Every f32 binary op folds into the one selector arm; the four selectors
        // 0/1/2/3 (add/sub/mul/div) all appear as the word after the opcode.
        assert!(
            mp.words.contains(&MINI_F32_ARITH_RS),
            "must lower f32 arithmetic through the selector arm"
        );
        for sel in 0..=3i64 {
            let matches = mp
                .words
                .windows(2)
                .any(|w| w[0] == MINI_F32_ARITH_RS && w[1] == sel);
            assert!(matches, "must emit f32 arith selector {sel}");
        }
    }

    /// Loops using f32 `<` / `<=` / `==` / `!=` as 0/1 VALUES (counting elements
    /// matching a float predicate) are JIT-eligible: the prepass lowers `F32Lt_Rrs`,
    /// `F32Le_Rrs`, `F32Eq_Rrs`, `F32NotEq_Rrs` through the `MINI_F32_CMP_RS_R`
    /// selector arm.
    #[test]
    fn f32_compare_value_is_eligible() {
        const LT_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t f32) (result i32)
                    (local $i i32) (local $count i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (f32.lt
                                        (f32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4))))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let has_cmp = |mp: &MiniProgram, sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F32_CMP_RS_R && w[1] == sel)
        };
        let lt = compile_and_prepass(LT_WAT);
        assert!(has_cmp(&lt, 0), "must lower f32.lt value (selector 0)");
        let le = compile_and_prepass(&LT_WAT.replace("(f32.lt", "(f32.le"));
        assert!(has_cmp(&le, 1), "must lower f32.le value (selector 1)");
        let eq = compile_and_prepass(&LT_WAT.replace("(f32.lt", "(f32.eq"));
        assert!(has_cmp(&eq, 2), "must lower f32.eq value (selector 2)");
        let ne = compile_and_prepass(&LT_WAT.replace("(f32.lt", "(f32.ne"));
        assert!(has_cmp(&ne, 3), "must lower f32.ne value (selector 3)");
    }

    /// An f32 loop using the unary ops `sqrt(abs(a[i]))` plus `floor`/`ceil` is
    /// JIT-eligible: the prepass lowers `F32{Abs,Sqrt,Floor,Ceil}_R{r,s}` through the
    /// `MINI_F32_UNARY_S` selector arm (sel 0=abs, 2=sqrt, 3=ceil, 4=floor).
    #[test]
    fn f32_unary_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result f32)
                    (local $i i32) (local $sum f32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f32.add (local.get $sum)
                                    (f32.floor
                                        (f32.ceil
                                            (f32.sqrt
                                                (f32.abs
                                                    (f32.load
                                                        (i32.add (local.get $ptr)
                                                                 (i32.mul (local.get $i) (i32.const 4))))))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        let has = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F32_UNARY_S && w[1] == sel)
        };
        assert!(has(0), "must lower f32.abs (unary selector 0)");
        assert!(has(2), "must lower f32.sqrt (unary selector 2)");
        assert!(has(3), "must lower f32.ceil (unary selector 3)");
        assert!(has(4), "must lower f32.floor (unary selector 4)");
    }

    /// An f32 loop mixing integer→f32 conversions (all four selectors) with a
    /// promote/demote round-trip is JIT-eligible: the prepass lowers
    /// `F32Convert{I,U}{32,64}` (`MINI_F32_CVT_S` sel 0/1/2/3), `F64PromoteF32`, and
    /// `F32DemoteF64`.
    #[test]
    fn f32_convert_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (param $q i64) (result f32)
                    (local $i i32) (local $sum f32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f32.add (local.get $sum)
                                    (f32.add (f32.convert_i32_s (local.get $i))
                                        (f32.add (f32.convert_i32_u (local.get $i))
                                            (f32.add (f32.convert_i64_s (local.get $q))
                                                (f32.demote_f64
                                                    (f64.promote_f32
                                                        (f32.convert_i64_u (local.get $q)))))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        let cvt = |sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F32_CVT_S && w[1] == sel)
        };
        assert!(cvt(0), "must lower f32.convert_i32_s (selector 0)");
        assert!(cvt(1), "must lower f32.convert_i32_u (selector 1)");
        assert!(cvt(2), "must lower f32.convert_i64_s (selector 2)");
        assert!(cvt(3), "must lower f32.convert_i64_u (selector 3)");
        assert!(
            mp.words.contains(&MINI_F64_PROMOTE_S),
            "must lower f64.promote_f32"
        );
        assert!(
            mp.words.contains(&MINI_F32_DEMOTE_S),
            "must lower f32.demote_f64"
        );
    }

    /// An f32 loop round-tripping bits through `i32.reinterpret_f32` and
    /// `f32.reinterpret_i32` (separated by a local so wasmi keeps both) is
    /// JIT-eligible: the prepass lowers `MINI_I32_REINTERP_F32`, `MINI_F32_REINTERP_I32`.
    #[test]
    fn f32_reinterpret_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result f32)
                    (local $i i32) (local $sum f32) (local $x f32) (local $ib i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $x
                                (f32.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 4)))))
                            (local.set $ib (i32.reinterpret_f32 (local.get $x)))
                            (local.set $sum
                                (f32.add (local.get $sum)
                                    (f32.reinterpret_i32 (local.get $ib))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_REINTERP_F32),
            "must lower i32.reinterpret_f32"
        );
        assert!(
            mp.words.contains(&MINI_F32_REINTERP_I32),
            "must lower f32.reinterpret_i32"
        );
    }

    /// Saturating and trapping f32→integer truncations are JIT-eligible: the
    /// prepass lowers `{I,U}{32,64}TruncSatF32` through `MINI_F32_TRUNC_SAT_S` and
    /// `{I,U}{32,64}TruncF32` through `MINI_F32_TRUNC_S` (both selector 0/1/2/3).
    #[test]
    fn f32_trunc_is_eligible() {
        const I64S_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $i i32) (local $sum i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.trunc_sat_f32_s
                                        (f32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let i32s_wat = I64S_WAT
            .replace("(local $sum i64)", "(local $sum i32)")
            .replace("(result i64)", "(result i32)")
            .replace("i64.add", "i32.add")
            .replace("i64.trunc_sat_f32_s", "i32.trunc_sat_f32_s");

        let sat = |mp: &MiniProgram, sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F32_TRUNC_SAT_S && w[1] == sel)
        };
        assert!(
            sat(&compile_and_prepass(I64S_WAT), 2),
            "trunc_sat i64_s (sel 2)"
        );
        assert!(
            sat(
                &compile_and_prepass(
                    &I64S_WAT.replace("i64.trunc_sat_f32_s", "i64.trunc_sat_f32_u")
                ),
                3
            ),
            "trunc_sat i64_u (sel 3)"
        );
        assert!(
            sat(&compile_and_prepass(&i32s_wat), 0),
            "trunc_sat i32_s (sel 0)"
        );
        assert!(
            sat(
                &compile_and_prepass(
                    &i32s_wat.replace("i32.trunc_sat_f32_s", "i32.trunc_sat_f32_u")
                ),
                1
            ),
            "trunc_sat i32_u (sel 1)"
        );

        // The trapping forms drop the `_sat` infix.
        let trunc = |mp: &MiniProgram, sel: i64| {
            mp.words
                .windows(2)
                .any(|w| w[0] == MINI_F32_TRUNC_S && w[1] == sel)
        };
        assert!(
            trunc(
                &compile_and_prepass(&I64S_WAT.replace("i64.trunc_sat_f32_s", "i64.trunc_f32_s")),
                2
            ),
            "trunc i64_s (sel 2)"
        );
        assert!(
            trunc(
                &compile_and_prepass(&i32s_wat.replace("i32.trunc_sat_f32_s", "i32.trunc_f32_u")),
                1
            ),
            "trunc i32_u (sel 1)"
        );
    }

    /// Integer `global.get`/`global.set` are JIT-eligible: the prepass lowers
    /// `GlobalGetU64_R` to `MINI_GLOBAL_GET_R` and every `GlobalSet{U64_R,U64_S,
    /// U64_I,U32_I}` form onto the single slot-sourced `MINI_GLOBAL_SET_S` arm
    /// (register / immediate sources pre-materialized), and sets `uses_globals`.
    #[test]
    fn globals_are_eligible() {
        const WAT: &str = r#"
            (module
                (global $g (mut i64) (i64.const 0))
                (func (export "run") (param $n i32) (result i64)
                    (local $i i32)
                    (global.set $g (i64.const 0))
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (global.set $g
                                (i64.add (global.get $g)
                                         (i64.extend_i32_s (local.get $i))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (global.get $g)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.uses_globals,
            "global-referencing function sets uses_globals"
        );
        assert!(
            mp.words.contains(&MINI_GLOBAL_GET_R),
            "global.get must lower to MINI_GLOBAL_GET_R"
        );
        assert!(
            mp.words.contains(&MINI_GLOBAL_SET_S),
            "global.set must lower onto MINI_GLOBAL_SET_S"
        );

        // A function that touches no global leaves `uses_globals` false.
        const NO_GLOBAL_WAT: &str = r#"
            (module
                (func (export "id") (param $n i64) (result i64)
                    (local.get $n)))
        "#;
        assert!(
            !compile_and_prepass(NO_GLOBAL_WAT).uses_globals,
            "a globals-free function must not set uses_globals"
        );

        // Immediate-sourced set (`GlobalSetU64_I`) also lowers via MINI_GLOBAL_SET_S.
        const IMM_WAT: &str = r#"
            (module
                (global $g (mut i64) (i64.const 0))
                (func (export "run") (param $n i64) (result i64)
                    (local $i i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (global.set $g (i64.const 7))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (global.get $g)))
        "#;
        let imp = compile_and_prepass(IMM_WAT);
        assert!(imp.uses_globals && imp.words.contains(&MINI_GLOBAL_SET_S));

        // Float globals lower to the dedicated get arms; the float set spills the
        // accumulator into a scratch slot and reuses MINI_GLOBAL_SET_S.
        const F64_WAT: &str = r#"
            (module
                (global $g (mut f64) (f64.const 0))
                (func (export "run") (param $n i32) (result f64)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (global.set $g
                                (f64.add (global.get $g)
                                         (f64.convert_i32_s (local.get $i))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (global.get $g)))
        "#;
        let f64m = compile_and_prepass(F64_WAT);
        assert!(
            f64m.words.contains(&MINI_GLOBAL_GET_F64) && f64m.words.contains(&MINI_GLOBAL_SET_S),
            "f64 global.get/set must lower"
        );
        let f32_wat = F64_WAT
            .replace("mut f64", "mut f32")
            .replace("(result f64)", "(result f32)")
            .replace("f64.const 0", "f32.const 0")
            .replace("f64.add", "f32.add")
            .replace("f64.convert_i32_s", "f32.convert_i32_s");
        let f32m = compile_and_prepass(&f32_wat);
        assert!(
            f32m.words.contains(&MINI_GLOBAL_GET_F32) && f32m.words.contains(&MINI_GLOBAL_SET_S),
            "f32 global.get/set must lower"
        );
    }

    /// Loops using signed `<=` and unsigned `<` / `<=` as 0/1 VALUES are
    /// JIT-eligible: the prepass lowers `I32Le_Rrs`, `U32Lt_Rrs`, `U32Le_Rrs`.
    #[test]
    fn le_and_unsigned_value_is_eligible() {
        const BASE_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (param $t i32) (result i32)
                    (local $i i32) (local $count i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (i32.le_s
                                        (i32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4))))
                                        (local.get $t))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let les = compile_and_prepass(BASE_WAT);
        assert!(
            les.words.contains(&MINI_I32_LE_RS_R),
            "must lower i32.le_s value"
        );

        let ltu = compile_and_prepass(&BASE_WAT.replace("(i32.le_s", "(i32.lt_u"));
        assert!(
            ltu.words.contains(&MINI_U32_LT_RS_R),
            "must lower i32.lt_u value"
        );

        let leu = compile_and_prepass(&BASE_WAT.replace("(i32.le_s", "(i32.le_u"));
        assert!(
            leu.words.contains(&MINI_U32_LE_RS_R),
            "must lower i32.le_u value"
        );
    }

    fn observe_count_via_locals_ops() {
        let wasm = wat::parse_str(COUNTER_WAT).expect("wat parse");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm[..]).expect("module");
        let ef = module.engine_func_by_index(0).expect("engine func 0");
        engine
            .with_compiled_ops(ef, |ops, l, s| {
                std::eprintln!(
                    "[prepass-observe] ops.len()={} len_local_slots={l} len_stack_slots={s}",
                    ops.len(),
                );
                disasm_observe(ops);
            })
            .expect("function compiled");
    }

    /// A loop chaining i32 `xor`/`and`/`or`/`sub` (the accumulator-OP-slot forms
    /// pre-materialized into a scratch slot) behind a signed i32 `<=` loop guard
    /// is JIT-eligible — one slot-slot kernel arm per op covers every form.
    #[test]
    fn i32_bitwise_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.xor (local.get $acc) (local.get $i)))
                            (local.set $acc (i32.and (local.get $acc) (local.get $n)))
                            (local.set $acc (i32.or (local.get $acc) (local.get $i)))
                            (local.set $acc (i32.sub (local.get $acc) (local.get $i)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        for (ss, rs) in [
            (MINI_I32_XOR_SS_WR, MINI_I32_XOR_RS_WR),
            (MINI_I32_AND_SS_WR, MINI_I32_AND_RS_WR),
            (MINI_I32_OR_SS_WR, MINI_I32_OR_RS_WR),
            (MINI_I32_SUB_SS_WR, MINI_I32_SUB_RS_WR),
        ] {
            assert!(
                mp.words.contains(&ss) || mp.words.contains(&rs),
                "must lower op {ss} or {rs}"
            );
        }
        assert!(mp.words.contains(&MINI_BR_I32_LE_SS), "must lower branch");
    }

    /// A loop summing an i32 array out of linear memory (sign-extending each
    /// element to i64) is JIT-eligible: the prepass lowers the i32 load
    /// (`U32LoadMem0Offset16_Rr`) and the sign-extend (`I64Sext32_Rr`).
    #[test]
    fn i32_array_sum_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $sum i64) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.extend_i32_s
                                        (i32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_LOAD_MEM0_OFF),
            "must lower the i32 linear-memory load"
        );
        assert!(
            mp.words.contains(&MINI_I64_SEXT32),
            "must lower the sign-extend"
        );
    }

    /// A loop summing an i64 array out of linear memory is JIT-eligible: the
    /// prepass lowers the load (`U64LoadMem0Offset16_Rr`) and the address
    /// arithmetic (`I32Mul_Rsi`, `I32Add_Rrs`) into the supported subset.
    #[test]
    fn array_sum_is_eligible() {
        const WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $sum i64) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.load
                                        (i32.add (local.get $ptr)
                                                 (i32.mul (local.get $i) (i32.const 8))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_LOAD_MEM0_OFF),
            "must lower the i64 linear-memory load"
        );
    }

    /// A running-max loop with an `if` (a forward conditional branch over the
    /// then-body, joining back into the loop) is JIT-eligible with no new op —
    /// the existing `BranchI64Le_Ss` arm handles a forward target (no back-edge).
    #[test]
    fn runningmax_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $max i64) (local $i i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (if (i64.gt_s (local.get $i) (local.get $max))
                                (then (local.set $max (local.get $i))))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $max)))
        "#;
        let mp = compile_and_prepass(WAT);
        // Two forward `BranchI64Le_Ss`: the loop-exit guard and the `if` skip.
        let le_ss = mp.words.iter().filter(|&&w| w == MINI_BR_I64_LE_SS).count();
        assert!(
            le_ss >= 2,
            "expected the loop guard and the `if` skip branch"
        );
    }

    /// An accumulation loop with a two-variable `i64.sub` (`a - i`) is
    /// JIT-eligible: the prepass lowers `I64Sub_Rss`.
    #[test]
    fn sub_accum_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $a i64) (param $b i64) (result i64)
                    (local $acc i64) (local $i i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $b)))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.sub (local.get $a) (local.get $i))))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_SUB_SS_WR),
            "must lower the two-variable i64 sub"
        );
    }

    /// An i64 bitwise-OR accumulation loop is JIT-eligible: the prepass lowers
    /// `I64BitOr_Rss`, and the `i + 1` step reuses the pre-materialized
    /// `I64Add_Rs_si` path (a scratch copy + the uniform two-slot add).
    #[test]
    fn or_accum_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $acc i64) (local $i i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i64.or (local.get $acc) (local.get $i)))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(mp.words.contains(&MINI_I64_OR_SS_WR), "must lower i64 or");
        // The `i + 1` step pre-materializes its immediate into the scratch slot.
        assert!(
            mp.words.contains(&MINI_COPY_SI),
            "the pre-materialized `i + 1` emits a scratch copy"
        );
    }

    /// A loop whose exit guard is an i64 equality compare of two slots
    /// (`i == n`) is JIT-eligible: the prepass lowers `BranchI64Eq_Ss`.
    #[test]
    fn eq_guard_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $count i64) (local $i i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eq (local.get $i) (local.get $n)))
                            (local.set $count (i64.add (local.get $count) (i64.const 1)))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_BR_I64_EQ_SS),
            "must lower the i64 equality guard"
        );
    }

    /// GCD by subtraction is JIT-eligible with no new op — it exercises an
    /// `if`/`else` (two forward branches plus the unconditional `Branch` that
    /// skips the else arm), all already in the subset.
    #[test]
    fn gcd_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $a i64) (param $b i64) (result i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eq (local.get $a) (local.get $b)))
                            (if (i64.gt_s (local.get $a) (local.get $b))
                                (then (local.set $a (i64.sub (local.get $a) (local.get $b))))
                                (else (local.set $b (i64.sub (local.get $b) (local.get $a)))))
                            (br $continue)))
                    (local.get $a)))
        "#;
        let mp = compile_and_prepass(WAT);
        // Two compare-branches (the `a == b` exit and the `if` skip) plus two
        // unconditional branches (the else-skip and the back-edge).
        let branches = mp.words.iter().filter(|&&w| w == MINI_BR_ALWAYS).count();
        assert!(
            branches >= 2,
            "if/else emits an else-skip and a back-edge branch"
        );
    }

    /// A loop adding an i64-operand comparison used as a 0/1 value
    /// (`count += i > 10`) is JIT-eligible: the prepass lowers `I64Lt_Ris`
    /// (`i64.gt_s i 10` rewritten to `10 < i`).
    #[test]
    fn i64_cmp_value_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i32)
                    (local $count i32) (local $i i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (i64.gt_s (local.get $i) (i64.const 10))))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_LT_IS_R),
            "must lower the i64 compare-as-value"
        );
    }

    /// A loop adding an i32 comparison used as a 0/1 VALUE (`count += i < 100`)
    /// is JIT-eligible: the prepass lowers `I32Lt_Rsi`.
    #[test]
    fn cmp_value_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $count i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $count
                                (i32.add (local.get $count)
                                    (i32.lt_s (local.get $i) (i32.const 100))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_LT_SI_R),
            "must lower the i32 compare-as-value"
        );
    }

    /// A loop accumulating `i << 2` (i64 left-shift by a constant) is
    /// JIT-eligible: the prepass lowers `I64Shl_Rsi`.
    #[test]
    fn shl_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $acc i64) (local $i i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.shl (local.get $i) (i64.const 2))))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_SHL_SI),
            "must lower the left shift"
        );
    }

    /// A loop with i32 left-shift (`I32Shl_Rsi`) and i32 logical right-shift of
    /// the accumulator (`U32Shr_Rri`) is JIT-eligible.
    #[test]
    fn i32_shift_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.shr_u (i32.shl (local.get $i) (i32.const 3))
                                               (i32.const 1))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I32_SHL_SI),
            "must lower the i32 left shift"
        );
        assert!(
            mp.words.contains(&MINI_U32_SHR_RI),
            "must lower the i32 logical right shift"
        );
    }

    /// A pure-i64 multiply loop (factorial) is JIT-eligible: the prepass lowers
    /// `I64Mul_Rss` and every other op into the supported subset.
    #[test]
    fn factorial_is_eligible() {
        const FACT_WAT: &str = r#"
            (module
                (func (export "fact") (param $n i64) (result i64)
                    (local $acc i64)
                    (local.set $acc (i64.const 1))
                    (block $break
                        (br_if $break (i64.eqz (local.get $n)))
                        (loop $continue
                            (local.set $acc (i64.mul (local.get $acc) (local.get $n)))
                            (local.set $n (i64.sub (local.get $n) (i64.const 1)))
                            (br_if $continue (i64.ne (local.get $n) (i64.const 0)))))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(FACT_WAT);
        assert!(
            mp.words.contains(&MINI_I64_MUL_SS_WR),
            "factorial must lower an i64 multiply"
        );
    }

    /// An i32 loop mixing `i32.mul`, an accumulator-plus-slot `i32.add`, an
    /// unsigned loop-exit compare (`i32.ge_u` → `BranchU32Le_Ss`), and an
    /// unconditional back-edge (`br` → `Branch`) is JIT-eligible.
    #[test]
    fn i32_sumsq_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $i i32) (local $acc i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_u (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.mul (local.get $i) (local.get $i))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(mp.words.contains(&MINI_I32_MUL_SS_WR), "must lower i32 mul");
        assert!(
            mp.words.contains(&MINI_BR_U32_LE_SS),
            "must lower the unsigned loop-exit compare"
        );
        assert!(
            mp.words.contains(&MINI_BR_ALWAYS),
            "must lower the unconditional back-edge"
        );
        assert_eq!(
            mp.loop_header_word,
            Some(0),
            "the back-edge targets the loop header at word 0"
        );
    }

    /// A bottom-tested loop — whose back-edge is a conditional `i64.gt_s`
    /// rewritten by wasmi to `BranchI64Lt_Ir` (`imm < ireg`) — is JIT-eligible.
    #[test]
    fn bottom_tested_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $sum i64) (local $i i64)
                    (local.set $i (local.get $n))
                    (block $break
                        (br_if $break (i64.le_s (local.get $i) (i64.const 0)))
                        (loop $continue
                            (local.set $sum (i64.add (local.get $sum) (local.get $i)))
                            (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                            (br_if $continue (i64.gt_s (local.get $i) (i64.const 0)))))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_BR_I64_LT_IR),
            "must lower the `imm < ireg` conditional back-edge"
        );
    }

    /// An i32 countup-sum loop — a signed i32 `< imm` loop guard
    /// (`BranchI32Lt_Si`) and a two-slot i32 add (`I32Add_Rs_ss`, with i32
    /// wraparound) — is JIT-eligible.
    #[test]
    fn i32_signed_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $sum i32) (local $i i32)
                    (local.set $i (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i32.lt_s (local.get $i) (i32.const 1)))
                            (local.set $sum (i32.add (local.get $sum) (local.get $i)))
                            (local.set $i (i32.sub (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_BR_I32_LT_SI),
            "must lower the signed i32 `< imm` loop guard"
        );
        assert!(
            mp.words.contains(&MINI_I32_ADD_SS_WB),
            "must lower the two-slot i32 add"
        );
    }

    /// A countdown-sum loop — a signed `<= imm` loop-exit guard
    /// (`BranchI64Le_Si`), a slot-and-reg add of two slots (`I64Add_Rs_ss`), and
    /// the `i - 1` decrement (which wasmi lowers to `I64Add_Rs_si` with a
    /// negated immediate) — is JIT-eligible.
    #[test]
    fn countdown_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $sum i64) (local $i i64)
                    (local.set $i (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.le_s (local.get $i) (i64.const 0)))
                            (local.set $sum (i64.add (local.get $sum) (local.get $i)))
                            (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_BR_I64_LE_SI),
            "must lower the signed `<= imm` loop guard"
        );
        assert!(
            mp.words.contains(&MINI_I64_ADD_SS_WB),
            "must lower the slot-and-reg two-slot add"
        );
    }

    /// A popcount loop — `i64.and` with an immediate against a slot
    /// (`I64BitAnd_Rsi`), an accumulator-plus-slot `i64.add` (`I64Add_Rs_rs`),
    /// and a logical shift-right by an immediate (`U64Shr_Rsi`) — is JIT-eligible.
    #[test]
    fn popcount_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $x i64) (result i64)
                    (local $count i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $x)))
                            (local.set $count
                                (i64.add (local.get $count)
                                    (i64.and (local.get $x) (i64.const 1))))
                            (local.set $x (i64.shr_u (local.get $x) (i64.const 1)))
                            (br $continue)))
                    (local.get $count)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(
            mp.words.contains(&MINI_I64_AND_SI_WR),
            "must lower i64 and-imm"
        );
        assert!(
            mp.words.contains(&MINI_U64_SHR_SI),
            "must lower the logical shift-right"
        );
    }

    /// An i64 loop mixing `i64.xor` (slot-slot), `i64.and` with an immediate
    /// against the accumulator (`I64BitAnd_Rri`), and a signed loop-exit compare
    /// (`i64.ge_s` → `BranchI64Le_Ss`) is JIT-eligible.
    #[test]
    fn mix_is_eligible() {
        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $h i64) (local $i i64)
                    (local.set $h (i64.const -1))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (local.set $h (i64.xor (local.get $h) (local.get $i)))
                            (local.set $h (i64.and (local.get $h) (i64.const 1023)))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $h)))
        "#;
        let mp = compile_and_prepass(WAT);
        assert!(mp.words.contains(&MINI_I64_XOR_SS_WR), "must lower i64 xor");
        assert!(
            mp.words.contains(&MINI_I64_AND_RI_WR),
            "must lower the accumulator-and-immediate"
        );
        assert!(
            mp.words.contains(&MINI_BR_I64_LE_SS),
            "must lower the signed loop-exit compare"
        );
    }

    /// Scope-finding observation for M4 (fibonacci_iter, a pure i64 loop).
    #[test]
    fn observe_fibonacci_iter_ops() {
        const FIB_WAT: &str = r#"
            (module
                (func (export "fibonacci_iter") (param $n i64) (result i64)
                    (local $a i64) (local $b i64) (local $i i64)
                    (local.set $a (i64.const 0))
                    (local.set $b (i64.const 1))
                    (local.set $i (local.get $n))
                    (block $break
                        (br_if $break (i64.eqz (local.get $i)))
                        (loop $continue
                            (i64.add (local.get $a) (local.get $b))
                            (local.set $a (local.get $b))
                            (local.set $b)
                            (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                            (br_if $continue (i64.ne (local.get $i) (i64.const 0)))))
                    (local.get $a)))
        "#;
        let wasm = wat::parse_str(FIB_WAT).expect("wat parse");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm[..]).expect("module");
        let ef = module.engine_func_by_index(0).expect("engine func 0");
        engine
            .with_compiled_ops(ef, |ops, l, s| {
                std::eprintln!(
                    "[prepass-observe] fib ops.len()={} len_local_slots={l} len_stack_slots={s}",
                    ops.len(),
                );
                disasm_observe(ops);
            })
            .expect("compiled");
    }

    /// M2 deliverable: the prepass turns `count-via-locals` into a well-formed
    /// MiniProgram (no execution — that is M3).
    #[test]
    fn prepass_count_via_locals_is_wellformed() {
        let mp = compile_and_prepass(COUNTER_WAT);

        // The return copies the accumulator into slot 0 (the caller's result
        // location) and then returns `slots[0]`.
        assert_eq!(
            mp.words,
            alloc::vec![
                MINI_I32_ADD_SI_WB,
                0,  // dst slot ($n)
                0,  // lhs slot ($n)
                -1, // imm (x - 1 == x + (-1))
                MINI_BR_I32_NE_RI,
                0, // target word -> loop header
                0, // imm (compare against 0)
                MINI_COPY_SR,
                0, // dst slot 0 <- accumulator
                MINI_RETURN_S,
                0, // return slots[0]
            ],
            "lowered MiniProgram word stream mismatch",
        );

        // The back-edge is recognised and points at the loop header (word 0).
        assert_eq!(mp.loop_header_word, Some(0));

        // The branch's resolved target is an op boundary that holds an opcode
        // (the loop header op), not a mid-instruction word.
        let header = mp.loop_header_word.unwrap();
        assert_eq!(mp.words[header], MINI_I32_ADD_SI_WB);
        assert_eq!(i64::try_from(mp.words[5]).unwrap(), header as i64);

        // Only slot 0 is referenced (1 live slot); the full frame is 5 slots
        // (1 local + 4 stack) but the compaction pass shrinks to the live range.
        assert_eq!(mp.num_slots, 1);

        // The function returns `slots[0]`.
        assert_eq!(*mp.words.last().unwrap(), 0);
        assert_eq!(mp.words[mp.words.len() - 2], MINI_RETURN_S);
    }

    /// A function outside the supported subset is rejected (`None`), so the
    /// caller will fall back to the stock executor.
    #[test]
    fn prepass_rejects_unsupported_ops() {
        // `memory.grow` is not in the prepass subset — the function must be
        // rejected as ineligible.
        const INDIRECT_WAT: &str = r#"
            (module
                (memory 1)
                (func (export "f") (param $p i32) (result i32)
                    (memory.grow (local.get $p))))
        "#;
        let wasm = wat::parse_str(INDIRECT_WAT).expect("wat parse");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm[..]).expect("module");
        let ef = module.engine_func_by_index(0).expect("engine func 0");
        let result = engine
            .with_compiled_ops(ef, |ops, l, s| prepass(ops, l, s))
            .expect("function compiled");
        assert!(
            result.is_none(),
            "memory.grow op must make the function ineligible"
        );
    }
}
