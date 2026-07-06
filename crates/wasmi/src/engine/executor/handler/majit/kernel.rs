//! M3 kernel: the majit-traced mainloop that interprets a [`MiniProgram`] (see
//! [`super::prepass`]). This is the loop majit meta-traces and compiles; it
//! replaces wasmi's handler-threaded dispatch for an eligible function.
//!
//! State (reds): the wasm frame slots are a virtualizable `[int; virt]` cell
//! array (`slots`); wasmi's implicit accumulator registers are a three-element
//! virtualizable array (`accum`): `accum[0]` is the integer accumulator
//! (`Reg<i64>`, `ireg`), `accum[1]` the f64 accumulator (`Reg<f64>`, `freg64`,
//! raw bits), and `accum[2]` the f32 accumulator (`Reg<f32>`, `freg32`, raw
//! bits in the low 32). They live on the vable rather than scalar reds so a
//! per-op handler jitcode can mutate them through the vable reference (the macro
//! has no virtualizable scalar). `greens = [pc, program]` lets the operand reads
//! (`program[pc + N]`) constant-fold so the loop traces and compiles.
//!
//! [`MiniProgram`]: super::prepass::MiniProgram

// M3 scaffolding mirrors `smoke`: the `stacksize` local is template boilerplate
// the macro expects, and the kernel is currently exercised only from tests.
#![allow(dead_code, unused_variables, unused_mut)]

// The `#[jit_interp]`-generated code names `Box`/`Vec`/`eprintln!`/`ToString`
// unqualified; this `#![no_std]` crate must bring them into scope (`majit-jit`
// always implies `std`).
use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::eprintln;

use super::prepass::{
    MINI_BR_ALWAYS, MINI_BR_I32_LE_SS, MINI_BR_I32_LT_SI, MINI_BR_I32_NE_RI, MINI_BR_I64_EQ_SI,
    MINI_BR_I64_EQ_SS, MINI_BR_I64_LE_SI, MINI_BR_I64_LE_SS, MINI_BR_I64_LT_IR, MINI_BR_I64_NE_RI,
    MINI_BR_I64_NE_SS, MINI_BR_U32_LE_SS, MINI_BR_U64_LT_SS, MINI_CALL_RESIDUAL, MINI_COPY_RI,
    MINI_COPY_RS, MINI_COPY_S_F32R, MINI_COPY_S_FR, MINI_COPY_SI, MINI_COPY_SR, MINI_COPY_SS,
    MINI_F32_ARITH_RS, MINI_F32_CMP_RS_R, MINI_F32_CVT_S, MINI_F32_DEMOTE_S,
    MINI_F32_LOAD_MEM0_OFF, MINI_F32_MINMAX_RS, MINI_F32_REINTERP_I32, MINI_F32_STORE_SR,
    MINI_F32_TRUNC_S, MINI_F32_TRUNC_SAT_S, MINI_F32_UNARY_S, MINI_F64_ARITH_RS, MINI_F64_CMP_RS_R,
    MINI_F64_CVT_S, MINI_F64_LOAD_MEM0_OFF, MINI_F64_MINMAX_RS, MINI_F64_PROMOTE_S,
    MINI_F64_REINTERP_I64, MINI_F64_STORE_SR, MINI_F64_TRUNC_S, MINI_F64_TRUNC_SAT_S,
    MINI_F64_UNARY_S, MINI_GLOBAL_GET_F32, MINI_GLOBAL_GET_F64, MINI_GLOBAL_GET_R,
    MINI_GLOBAL_SET_S, MINI_I8_LOAD_MEM0_OFF, MINI_I16_LOAD_MEM0_OFF, MINI_I32_ADD_RS_WB,
    MINI_I32_ADD_SI_WB, MINI_I32_ADD_SS_WB, MINI_I32_AND_SS_WR, MINI_I32_BITCOUNT_S,
    MINI_I32_DIV_S, MINI_I32_DIV_U, MINI_I32_EQ_RS_R, MINI_I32_EQ_SS_R, MINI_I32_LE_RS_R,
    MINI_I32_LE_SS_R, MINI_I32_LOAD_MEM0_OFF, MINI_I32_LT_RS_R, MINI_I32_LT_SI_R, MINI_I32_LT_SR_R,
    MINI_I32_LT_SS_R, MINI_I32_MUL_SS_WR, MINI_I32_NE_RS_R, MINI_I32_NE_SS_R, MINI_I32_OR_SS_WR,
    MINI_I32_REINTERP_F32, MINI_I32_REM_S, MINI_I32_REM_U, MINI_I32_ROTL_SI, MINI_I32_ROTR_SI,
    MINI_I32_SHL_SI, MINI_I32_STORE_RS, MINI_I32_STORE_SR, MINI_I32_STORE8_RS, MINI_I32_STORE8_SR,
    MINI_I32_STORE16_RS, MINI_I32_STORE16_SR, MINI_I32_SUB_SS_WR, MINI_I32_XOR_SS_WR,
    MINI_I64_ADD_RS_WB, MINI_I64_ADD_SS_WB, MINI_I64_ADD_SS_WR, MINI_I64_AND_RI_WR,
    MINI_I64_AND_SI_WR, MINI_I64_BITCOUNT_S, MINI_I64_DIV_S, MINI_I64_DIV_U, MINI_I64_EQ_RS_R,
    MINI_I64_EQ_SS_R, MINI_I64_LE_SS_R, MINI_I64_LOAD_MEM0_OFF, MINI_I64_LT_IS_R, MINI_I64_LT_RS_R,
    MINI_I64_LT_SI_R, MINI_I64_LT_SS_R, MINI_I64_MUL_SS_WR, MINI_I64_NE_RS_R, MINI_I64_NE_SS_R,
    MINI_I64_OR_SS_WR, MINI_I64_REINTERP_F64, MINI_I64_REM_S, MINI_I64_REM_U, MINI_I64_SEXT32,
    MINI_I64_SEXT32_S, MINI_I64_SHL_SI, MINI_I64_STORE_RS, MINI_I64_STORE_SR, MINI_I64_SUB_SS_WR,
    MINI_I64_XOR_SS_WR, MINI_MEMORY_SIZE, MINI_RETURN_BAIL, MINI_RETURN_F_R, MINI_RETURN_F32_R,
    MINI_RETURN_R, MINI_RETURN_S, MINI_RETURN_VOID, MINI_SELECT, MINI_TRAP, MINI_U8_LOAD_MEM0_OFF,
    MINI_U16_LOAD_MEM0_OFF, MINI_U32_LE_RS_R, MINI_U32_LE_SS_R, MINI_U32_LT_RS_R, MINI_U32_LT_SS_R,
    MINI_U32_SHR_RI, MINI_U64_LE_SS_R, MINI_U64_LT_SS_R, MINI_U64_SHR_SI, MINI_U64_SHR_SS_WR,
    MINI_CALL_IMPORTED, MINI_CALL_INDIRECT, MINI_I64_OR_RI_WR, MINI_MEM_COPY_WITHIN,
    MINI_YIELD_STOCK, MiniCode,
};

/// Counts hot loops majit compiled in the kernel — evidence the JIT tier traced
/// and compiled the wasm loop rather than only interpreting it.
pub static KERNEL_COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Counts guard-failure deopts: distinguishes "compiled trace runs the loop"
/// (≈1 deopt at the loop-exit side exit) from "bails to the interpreter every
/// iteration" (≈N deopts).
pub static KERNEL_GUARD_FAILS: AtomicUsize = AtomicUsize::new(0);

const U64_ORDER_FLIP: i64 = i64::MIN;

const _: [i64; 12] = [
    MINI_I32_EQ_SS_R,
    MINI_I32_NE_SS_R,
    MINI_I32_LT_SS_R,
    MINI_I32_LE_SS_R,
    MINI_U32_LT_SS_R,
    MINI_U32_LE_SS_R,
    MINI_I64_EQ_SS_R,
    MINI_I64_NE_SS_R,
    MINI_I64_LT_SS_R,
    MINI_I64_LE_SS_R,
    MINI_U64_LT_SS_R,
    MINI_U64_LE_SS_R,
];

std::thread_local! {
    /// The default linear memory's `(base, len)` for the current kernel run, set
    /// by [`run_persistent`]/`run_kernel` before each entry. The residual memory
    /// helpers read it rather than receiving the base/len as JIT reds — a
    /// loop-invariant read-only red would break the trace's snapshot liveness,
    /// and keeping the vable schema unchanged (slots + accum) leaves the proven
    /// integer-kernel snapshot machinery untouched. Re-set each entry, so a
    /// `memory.grow` relocation between calls is reflected.
    static MEM_CTX: core::cell::Cell<(i64, i64)> = const { core::cell::Cell::new((0, 0)) };
    /// Set by a residual memory access when its effective address is out of
    /// bounds. The kernel cannot trap directly (it returns a plain `i64`), so the
    /// access returns a dummy value (and a store applies nothing), the loop runs
    /// to completion, and [`run_jit`] then surfaces the trap.
    ///
    /// Recovery depends on whether a store was committed this run ([`MEM_DID_STORE`]):
    /// - no store committed (pure-load run, or a store function that trapped
    ///   before its first in-bounds store): the discarded JIT run left memory
    ///   untouched, so re-running the stock executor reproduces the faithful wasm
    ///   bounds-check trap.
    /// - a store committed: re-running stock would double-apply the stores the JIT
    ///   already performed, so [`run_jit`] raises the out-of-bounds trap directly.
    ///   This is sound because the store helper applies stores in program order
    ///   and stops at the first out-of-bounds access, so memory matches what stock
    ///   would have left at the trap point.
    ///
    /// [`run_jit`]: super::super::func
    static MEM_TRAP: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// Set true when a residual store actually wrote to memory this run. Selects
    /// the [`MEM_TRAP`] recovery strategy (stock re-run vs. direct trap).
    static MEM_DID_STORE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// The trap code to raise when [`MEM_TRAP`] fires after a committed store.
    /// Defaults to `MemoryOutOfBounds` (the only code the memory residuals raise);
    /// a trapping f64→int conversion overrides it with the exact core trap code
    /// (`BadConversionToInteger` / `IntegerOverflow`). First trap in program order
    /// wins (the residuals latch on [`MEM_TRAP`]), so this matches what the stock
    /// executor would have trapped on at the same point.
    static TRAP_CODE: core::cell::Cell<crate::TrapCode> =
        const { core::cell::Cell::new(crate::TrapCode::MemoryOutOfBounds) };
    /// Per-run table of raw pointers to each instance global's `RawVal` storage
    /// (`(table_ptr, count)`), set by [`set_globals_ctx`] before a run whose
    /// function references globals. The residual global helpers index it by wasm
    /// global index. Like [`MEM_CTX`], it is read-only in the trace (globals are a
    /// loop-invariant environment, not a red) and re-set each entry so a store
    /// relocation between calls is reflected. `table_ptr` points at a `Vec<*mut u64>`
    /// owned by the caller for the duration of the run; `count` is its length.
    static GLOBALS_CTX: core::cell::Cell<(*const *mut u64, usize)> =
        const { core::cell::Cell::new((core::ptr::null(), 0)) };
    /// Set by [`MINI_RETURN_BAIL`] when the kernel hits an instruction that
    /// requires falling back to the stock executor (e.g. a tail call). Cleared
    /// at each run entry.
    static BAIL_TO_STOCK: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// Set by [`MINI_YIELD_STOCK`] when the kernel hits a CallInternal. Unlike
    /// [`BAIL_TO_STOCK`], the yield carries the byte offset and a slot snapshot
    /// so the caller can resume the stock executor at the exact instruction
    /// rather than re-running from the start. Cleared at each run entry.
    static YIELD_TO_STOCK: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// The byte offset of the CallInternal instruction in the op stream,
    /// recorded by [`MINI_YIELD_STOCK`]. Valid only when [`YIELD_TO_STOCK`] is
    /// true.
    static YIELD_BYTE_OFFSET: core::cell::Cell<i64> = const { core::cell::Cell::new(0) };
}

std::thread_local! {
    /// Snapshot of the kernel's slot array at the yield point, set by
    /// [`MINI_YIELD_STOCK`]. The caller reads it via [`take_yield_slots`] and
    /// flushes it to the real frame before resuming the stock executor.
    static YIELD_SLOTS: core::cell::RefCell<Vec<i64>> = core::cell::RefCell::new(Vec::new());

}

/// Maximum number of params for a single CallInternal. Wasm functions rarely
/// exceed 8 params; 16 gives ample headroom without heap allocation.
const MAX_CALL_PARAMS: usize = 16;

std::thread_local! {
    /// Combined staging buffer + length for callee params. One TLS access
    /// instead of two per read/write side. The full 128-byte array is copied
    /// via Cell::set/get, but the merge halves the `.with()` call overhead.
    static CALL_STAGING: core::cell::Cell<([i64; MAX_CALL_PARAMS], usize)> =
        const { core::cell::Cell::new(([0i64; MAX_CALL_PARAMS], 0)) };
}

/// Opaque execution context for [`call_internal_residual`], set by `run_jit`
/// via [`set_call_runner`]. Stores a function pointer + data pointer to a
/// closure-like struct that lives on `run_jit`'s stack — valid for the entire
/// kernel run.
///
/// We use a concrete fn pointer + `*mut ()` instead of `*mut dyn Trait` to
/// avoid fat pointer complexity. The fn pointer signature is:
/// `fn(data: *mut (), func_addr: usize, params: &[i64]) -> i64`
type CallRunnerFn = fn(*mut (), usize, &[i64]) -> i64;
/// Imported-call runner: `fn(data, func_index: u32, params) -> i64`.
type CallImportedRunnerFn = fn(*mut (), u32, &[i64]) -> i64;
/// Indirect-call runner: `fn(data, table: u32, func_type: u32, runtime_index: u64, params) -> i64`.
type CallIndirectRunnerFn = fn(*mut (), u32, u32, u64, &[i64]) -> i64;

std::thread_local! {
    static CALL_RUNNER_FN: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    static CALL_RUNNER_DATA: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    static CALL_IMPORTED_RUNNER_FN: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    static CALL_INDIRECT_RUNNER_FN: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Register all call runners for the current kernel run.
pub(crate) fn set_call_runner(f: CallRunnerFn, data: *mut ()) {
    CALL_RUNNER_FN.with(|c| c.set(f as usize));
    CALL_RUNNER_DATA.with(|c| c.set(data as usize));
}

/// Register imported/indirect call runners.
pub(crate) fn set_imported_call_runners(
    imported: CallImportedRunnerFn,
    indirect: CallIndirectRunnerFn,
) {
    CALL_IMPORTED_RUNNER_FN.with(|c| c.set(imported as usize));
    CALL_INDIRECT_RUNNER_FN.with(|c| c.set(indirect as usize));
}

/// Clear the call runner (called after `run_persistent` returns).
pub(crate) fn clear_call_runner() {
    CALL_RUNNER_FN.with(|c| c.set(0));
    CALL_RUNNER_DATA.with(|c| c.set(0));
    CALL_IMPORTED_RUNNER_FN.with(|c| c.set(0));
    CALL_INDIRECT_RUNNER_FN.with(|c| c.set(0));
}

/// Records the per-run global raw-pointer table (see [`GLOBALS_CTX`]). The caller
/// owns the backing slice for the run's duration.
fn set_globals_ctx(table: *const *mut u64, count: usize) {
    GLOBALS_CTX.with(|c| c.set((table, count)));
}

/// Stage callee params into the fixed-size [`CALL_STAGING`] buffer. Called
/// from the [`MINI_CALL_RESIDUAL`] dispatch arm before invoking
/// [`call_internal_residual`]. No heap allocation.
fn call_stage_params(buf: &[i64]) {
    let mut arr = [0i64; MAX_CALL_PARAMS];
    let n = buf.len().min(MAX_CALL_PARAMS);
    arr[..n].copy_from_slice(&buf[..n]);
    CALL_STAGING.with(|c| c.set((arr, n)));
}

/// Residual: execute an internal function call. Reads the staged params from
/// [`CALL_STAGING`] and delegates to the [`CallRunnerFn`] registered by
/// `run_jit`. The call runner pushes a root frame on a separate Stack, runs
/// `execute_until_done`, and returns the callee's i64 return value.
///
/// Marked `#[dont_look_inside]` so the JIT treats this as an opaque call —
/// the callee's bytecode is not traced, matching PyPy's `ll_portal_runner`
/// residual semantics.
#[majit_macros::dont_look_inside]
extern "C" fn call_internal_residual(func_addr: i64, n_params: i64) -> i64 {
    let runner_fn = CALL_RUNNER_FN.with(|c| c.get());
    let runner_data = CALL_RUNNER_DATA.with(|c| c.get());
    if runner_fn == 0 {
        // No call runner registered — should not happen for an eligible function.
        set_residual_trap(crate::TrapCode::UnreachableCodeReached);
        return 0;
    }
    let f: CallRunnerFn = unsafe { core::mem::transmute::<usize, CallRunnerFn>(runner_fn) };
    let data = runner_data as *mut ();
    let (staging, n) = CALL_STAGING.with(|c| c.get());
    f(data, func_addr as usize, &staging[..n])
}

/// Residual: execute an imported function call. Reads staged params and
/// delegates to the [`CallImportedRunnerFn`] which resolves the `Func` handle
/// through the store and dispatches to Wasm or Host.
#[majit_macros::dont_look_inside]
extern "C" fn call_imported_residual(func_index: i64, n_params: i64) -> i64 {
    let runner_fn = CALL_IMPORTED_RUNNER_FN.with(|c| c.get());
    let runner_data = CALL_RUNNER_DATA.with(|c| c.get());
    if runner_fn == 0 {
        set_residual_trap(crate::TrapCode::UnreachableCodeReached);
        return 0;
    }
    let f: CallImportedRunnerFn =
        unsafe { core::mem::transmute::<usize, CallImportedRunnerFn>(runner_fn) };
    let data = runner_data as *mut ();
    let (staging, n) = CALL_STAGING.with(|c| c.get());
    f(data, func_index as u32, &staging[..n])
}

/// Residual: execute an indirect function call. Reads staged params and
/// delegates to the [`CallIndirectRunnerFn`] which performs table lookup,
/// null check, type check, and Wasm/Host dispatch.
#[majit_macros::dont_look_inside]
extern "C" fn call_indirect_residual(
    table: i64,
    func_type: i64,
    runtime_index: i64,
    n_params: i64,
) -> i64 {
    let runner_fn = CALL_INDIRECT_RUNNER_FN.with(|c| c.get());
    let runner_data = CALL_RUNNER_DATA.with(|c| c.get());
    if runner_fn == 0 {
        set_residual_trap(crate::TrapCode::UnreachableCodeReached);
        return 0;
    }
    let f: CallIndirectRunnerFn =
        unsafe { core::mem::transmute::<usize, CallIndirectRunnerFn>(runner_fn) };
    let data = runner_data as *mut ();
    let (staging, n) = CALL_STAGING.with(|c| c.get());
    f(data, table as u32, func_type as u32, runtime_index as u64, &staging[..n])
}

/// Residual read of an integer global's raw `lo64` bits by wasm global index.
/// Marked `#[dont_look_inside]` so it stays a real call in the compiled trace.
/// The index is validated by wasm and the table covers every instance global, so
/// an out-of-range index is unreachable; it returns 0 defensively.
#[majit_macros::dont_look_inside]
extern "C" fn global_get(idx: i64) -> i64 {
    let (table, count) = GLOBALS_CTX.with(|c| c.get());
    if idx < 0 || idx as usize >= count {
        return 0;
    }
    let slot = unsafe { *table.add(idx as usize) };
    unsafe { *slot as i64 }
}

/// Residual write of raw `lo64` bits into an integer global by wasm global index.
/// `global.set` only targets mutable globals (wasm validation), whose table entry
/// is a sound `*mut` from `get_raw_ptr`; the write touches only the low 64 bits,
/// leaving any high (simd) bits untouched.
#[majit_macros::dont_look_inside]
extern "C" fn global_set(idx: i64, val: i64) {
    let (table, count) = GLOBALS_CTX.with(|c| c.get());
    if idx < 0 || idx as usize >= count {
        return;
    }
    let slot = unsafe { *table.add(idx as usize) };
    unsafe { *slot = val as u64 };
}

/// Sets the linear-memory context for the next kernel run and clears the
/// per-run store flag, trap code, and bail flag.
fn set_mem_ctx(base: i64, len: i64) {
    MEM_CTX.with(|c| c.set((base, len)));
    MEM_DID_STORE.with(|d| d.set(false));
    TRAP_CODE.with(|c| c.set(crate::TrapCode::MemoryOutOfBounds));
    BAIL_TO_STOCK.with(|b| b.set(false));
    YIELD_TO_STOCK.with(|y| y.set(false));
}

/// Update only the MEM_CTX base/len without resetting per-run flags.
/// Called after a residual imported/indirect call that may trigger memory.grow.
pub(crate) fn update_mem_ctx(base: i64, len: i64) {
    MEM_CTX.with(|c| c.set((base, len)));
}

/// Returns `true` (once) if the last kernel run hit a `MINI_RETURN_BAIL`
/// instruction. Clears the flag on read.
pub fn take_bail_to_stock() -> bool {
    BAIL_TO_STOCK.with(|b| b.replace(false))
}

/// Returns `true` (once) if the last kernel run hit a [`MINI_YIELD_STOCK`]
/// instruction (a CallInternal). Clears the flag on read.
pub fn take_yield_to_stock() -> bool {
    YIELD_TO_STOCK.with(|y| y.replace(false))
}

/// The byte offset recorded by the last [`MINI_YIELD_STOCK`]. Valid only when
/// [`take_yield_to_stock`] returned true.
pub(crate) fn take_yield_offset() -> i64 {
    YIELD_BYTE_OFFSET.with(|c| c.get())
}

/// Takes the kernel's slot snapshot from the last [`MINI_YIELD_STOCK`], leaving
/// an empty Vec behind.
pub(crate) fn take_yield_slots() -> Vec<i64> {
    YIELD_SLOTS.with(|c| c.replace(Vec::new()))
}

/// Flags a trap from a residual (e.g. a trapping f64→int conversion), recording
/// the exact trap code. Latches on [`MEM_TRAP`] so the first trap in program
/// order keeps its code and stops any later store, matching the stock executor.
pub(crate) fn set_residual_trap(code: crate::TrapCode) {
    if !MEM_TRAP.with(|t| t.get()) {
        MEM_TRAP.with(|t| t.set(true));
        TRAP_CODE.with(|c| c.set(code));
    }
}

/// Reads and clears the residual-memory trap flag.
pub(crate) fn take_mem_trap() -> bool {
    MEM_TRAP.with(|t| t.replace(false))
}

/// The trap code recorded for the trap that fired this run (valid only when
/// [`take_mem_trap`] returned true). Set each run by [`set_mem_ctx`].
pub(crate) fn take_trap_code() -> crate::TrapCode {
    TRAP_CODE.with(|c| c.get())
}

/// Reads and clears the "a store was committed this run" flag. When this is true
/// after a [`take_mem_trap`], the JIT run already mutated memory and [`run_jit`]
/// must raise the trap directly rather than re-running the stock executor.
pub(crate) fn take_mem_did_store() -> bool {
    MEM_DID_STORE.with(|d| d.replace(false))
}

/// Residual i64 load from the default linear memory: `*(base + ea)`,
/// bounds-checked against `len` (both read from [`MEM_CTX`]). Out of bounds sets
/// [`MEM_TRAP`] and returns 0 (the caller falls back to the stock executor).
/// Marked `#[dont_look_inside]` so the metainterp lowers it to a real residual
/// call that stays a call in the compiled trace rather than being traced through.
#[majit_macros::dont_look_inside]
extern "C" fn mem_load_i64(ea: i64) -> i64 {
    let (base, len) = MEM_CTX.with(|c| c.get());
    // `ea` arrives as `(ptr & 0xffff_ffff) + offset`, so it is non-negative.
    if ea < 0 || ea + 8 > len {
        MEM_TRAP.with(|t| t.set(true));
        return 0;
    }
    unsafe { core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const i64) }
}

/// Residual i32 load: reads 4 bytes and sign-extends to a canonical i32 (the
/// kernel's i32 representation). Bounds-checked like [`mem_load_i64`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_load_i32(ea: i64) -> i64 {
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 4 > len {
        MEM_TRAP.with(|t| t.set(true));
        return 0;
    }
    let v = unsafe {
        core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const i32)
    };
    i64::from(v)
}

/// Residual unsigned-byte load: reads 1 byte zero-extended (0..=255).
/// Bounds-checked like [`mem_load_i64`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_load_u8(ea: i64) -> i64 {
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 1 > len {
        MEM_TRAP.with(|t| t.set(true));
        return 0;
    }
    let v = unsafe { core::ptr::read((base as usize).wrapping_add(ea as usize) as *const u8) };
    i64::from(v)
}

/// Residual signed-byte load: reads 1 byte sign-extended (-128..=127).
/// Bounds-checked like [`mem_load_i64`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_load_i8(ea: i64) -> i64 {
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 1 > len {
        MEM_TRAP.with(|t| t.set(true));
        return 0;
    }
    let v = unsafe { core::ptr::read((base as usize).wrapping_add(ea as usize) as *const i8) };
    i64::from(v)
}

/// Residual unsigned-16-bit load: reads 2 bytes zero-extended (0..=65535).
/// Bounds-checked like [`mem_load_i64`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_load_u16(ea: i64) -> i64 {
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 2 > len {
        MEM_TRAP.with(|t| t.set(true));
        return 0;
    }
    let v = unsafe {
        core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const u16)
    };
    i64::from(v)
}

/// Residual signed-16-bit load: reads 2 bytes sign-extended (-32768..=32767).
/// Bounds-checked like [`mem_load_i64`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_load_i16(ea: i64) -> i64 {
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 2 > len {
        MEM_TRAP.with(|t| t.set(true));
        return 0;
    }
    let v = unsafe {
        core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const i16)
    };
    i64::from(v)
}

/// Residual 32-bit store: writes the low 32 bits of `val` to `mem[ea..ea+4]`.
///
/// Bounds-checked like the loads. To keep the discarded-or-trapped JIT run's
/// memory equal to the stock executor's at the trap point, stores are applied in
/// program order and the first out-of-bounds store sets [`MEM_TRAP`] and stops
/// all subsequent stores (the early `MEM_TRAP` check), so no store past the trap
/// is ever applied.
#[majit_macros::dont_look_inside]
extern "C" fn mem_store_i32(ea: i64, val: i64) {
    if MEM_TRAP.with(|t| t.get()) {
        return;
    }
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 4 > len {
        MEM_TRAP.with(|t| t.set(true));
        return;
    }
    unsafe {
        core::ptr::write_unaligned(
            (base as usize).wrapping_add(ea as usize) as *mut u32,
            val as u32,
        );
    }
    MEM_DID_STORE.with(|d| d.set(true));
}

/// Residual 64-bit store: writes all 8 bytes of `val` to `mem[ea..ea+8]`.
/// Bounds-checked and program-ordered like [`mem_store_i32`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_store_i64(ea: i64, val: i64) {
    if MEM_TRAP.with(|t| t.get()) {
        return;
    }
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 8 > len {
        MEM_TRAP.with(|t| t.set(true));
        return;
    }
    unsafe {
        core::ptr::write_unaligned((base as usize).wrapping_add(ea as usize) as *mut i64, val);
    }
    MEM_DID_STORE.with(|d| d.set(true));
}

/// Residual 8-bit store: writes the low byte of `val` to `mem[ea]`.
/// Bounds-checked and program-ordered like [`mem_store_i32`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_store_u8(ea: i64, val: i64) {
    if MEM_TRAP.with(|t| t.get()) {
        return;
    }
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 1 > len {
        MEM_TRAP.with(|t| t.set(true));
        return;
    }
    unsafe {
        core::ptr::write(
            (base as usize).wrapping_add(ea as usize) as *mut u8,
            val as u8,
        );
    }
    MEM_DID_STORE.with(|d| d.set(true));
}

/// Residual 16-bit store: writes the low 2 bytes of `val` to `mem[ea..ea+2]`.
/// Bounds-checked and program-ordered like [`mem_store_i32`].
#[majit_macros::dont_look_inside]
extern "C" fn mem_store_u16(ea: i64, val: i64) {
    if MEM_TRAP.with(|t| t.get()) {
        return;
    }
    let (base, len) = MEM_CTX.with(|c| c.get());
    if ea < 0 || ea + 2 > len {
        MEM_TRAP.with(|t| t.set(true));
        return;
    }
    unsafe {
        core::ptr::write_unaligned(
            (base as usize).wrapping_add(ea as usize) as *mut u16,
            val as u16,
        );
    }
    MEM_DID_STORE.with(|d| d.set(true));
}

/// Residual `memory.copy` within the same linear memory (memory[0]).
/// Performs bounds-checked `copy_within` (handles overlapping regions).
/// Out-of-bounds sets `MEM_TRAP`. Skipped if a prior trap already occurred.
#[majit_macros::dont_look_inside]
extern "C" fn mem_copy_within(dst: i64, src: i64, copy_len: i64) {
    if MEM_TRAP.with(|t| t.get()) {
        return;
    }
    let (base, mem_len) = MEM_CTX.with(|c| c.get());
    let n = copy_len as u64;
    let d = dst as u64;
    let s = src as u64;
    if d.checked_add(n).is_none_or(|end| end as i64 > mem_len)
        || s.checked_add(n).is_none_or(|end| end as i64 > mem_len)
    {
        MEM_TRAP.with(|t| t.set(true));
        return;
    }
    let base = base as usize;
    let d = d as usize;
    let s = s as usize;
    let n = n as usize;
    // SAFETY: bounds checked above; base is the linear memory allocation.
    unsafe {
        let ptr = base as *mut u8;
        core::ptr::copy(ptr.add(s), ptr.add(d), n);
    }
    MEM_DID_STORE.with(|d| d.set(true));
}

/// Residual f64 arithmetic, `sel`-dispatched: 0=add, 1=sub, 2=mul, 3=div. Both
/// operands and the result are i64 bit-patterns (the slots hold f64 values as
/// their raw bits). The bit-casts live inside this `#[dont_look_inside]` helper,
/// so they never reach the trace (an in-trace float cast has no lowering). Folding
/// the four operations behind one selector keeps the force-inlined dispatch below
/// the 256 register/const ceiling. Pure — re-executed each compiled iteration.
/// `sub` / `div` are non-commutative and take operands in wasm order; f64 ops never
/// trap (`/0` yields IEEE inf/NaN).
#[majit_macros::dont_look_inside]
extern "C" fn f64_arith(sel: i64, a_bits: i64, b_bits: i64) -> i64 {
    let a = f64::from_bits(a_bits as u64);
    let b = f64::from_bits(b_bits as u64);
    let r = match sel {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        _ => a / b,
    };
    r.to_bits() as i64
}

/// Residual f32 arithmetic, `sel`-dispatched: 0=add, 1=sub, 2=mul, 3=div. An f32
/// value lives as its raw 32 bits in the low half of an i64 (the `freg32`
/// accumulator / a slot); the bit-casts and the operation stay inside this
/// `#[dont_look_inside]` helper, never reaching the trace. The result is the f32
/// bit pattern zero-extended into an i64. Folding the four operations behind one
/// selector keeps the force-inlined dispatch below the 256 register/const ceiling.
/// Pure, re-executed each compiled iteration; f32 ops never trap (`/0` yields IEEE
/// inf/NaN). `sub` / `div` are non-commutative and take operands in wasm order.
#[majit_macros::dont_look_inside]
extern "C" fn f32_arith(sel: i64, a_bits: i64, b_bits: i64) -> i64 {
    let a = f32::from_bits(a_bits as u32);
    let b = f32::from_bits(b_bits as u32);
    let r = match sel {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        _ => a / b,
    };
    i64::from(r.to_bits())
}

/// Residual f32 binary combine, `sel`-dispatched: 0=min, 1=max,
/// 2=copysign(a,b), 3=copysign(b,a). min/max carry wasm semantics
/// (NaN-propagating, signed-zero aware); copysign takes the magnitude of the
/// first argument and the sign of the second. `sel` 3 swaps the operands for the
/// `_Rsr` form, where the magnitude operand is the slot and the sign is the
/// accumulator. The f32 bits live in the low half of an i64; the bit-casts stay
/// inside the residual (never reaching the trace).
#[majit_macros::dont_look_inside]
extern "C" fn f32_minmax(sel: i64, a_bits: i64, b_bits: i64) -> i64 {
    let a = f32::from_bits(a_bits as u32);
    let b = f32::from_bits(b_bits as u32);
    let r = match sel {
        0 => crate::core::wasm::f32_min(a, b),
        1 => crate::core::wasm::f32_max(a, b),
        2 => crate::core::wasm::f32_copysign(a, b),
        _ => crate::core::wasm::f32_copysign(b, a),
    };
    i64::from(r.to_bits())
}

/// Residual f32 comparisons, `sel`-dispatched: 0=lt, 1=le, 2=eq, 3=ne,
/// 4=lt swapped, 5=le swapped. Returns a 0/1 i32 value (into `ireg`). Rust `<` /
/// `<=` / `==` are false for NaN and `!=` is true, matching wasm. One selector arm
/// folds the compare forms.
#[majit_macros::dont_look_inside]
extern "C" fn f32_cmp(sel: i64, a_bits: i64, b_bits: i64) -> i64 {
    let a = f32::from_bits(a_bits as u32);
    let b = f32::from_bits(b_bits as u32);
    let r = match sel {
        0 => a < b,
        1 => a <= b,
        2 => a == b,
        4 => b < a,
        5 => b <= a,
        _ => a != b,
    };
    r as i64
}

/// Residual f32 unary ops, `sel`-dispatched: 0=abs, 1=neg, 2=sqrt, 3=ceil,
/// 4=floor, 5=trunc, 6=nearest. Reuses the core crate's `wasm::f32_*` helpers
/// (std or libm). The bit-casts stay inside the residual; one selector arm folds
/// the seven ops (see `f32_arith`). The result is the f32 bit pattern in an i64.
#[majit_macros::dont_look_inside]
extern "C" fn f32_unary(sel: i64, a_bits: i64) -> i64 {
    let a = f32::from_bits(a_bits as u32);
    let r = match sel {
        0 => crate::core::wasm::f32_abs(a),
        1 => crate::core::wasm::f32_neg(a),
        2 => crate::core::wasm::f32_sqrt(a),
        3 => crate::core::wasm::f32_ceil(a),
        4 => crate::core::wasm::f32_floor(a),
        5 => crate::core::wasm::f32_trunc(a),
        _ => crate::core::wasm::f32_nearest(a),
    };
    i64::from(r.to_bits())
}

/// Residual integer→f32 conversion, `sel`-dispatched: 0=i32_s, 1=i32_u (u32),
/// 2=i64_s, 3=i64_u (u64). The integer operand is in the low bits of `a`; the
/// width/sign casts live inside the residual (never in the trace). The result is
/// the f32 bit pattern in an i64 (into `freg32`). One selector folds the four.
#[majit_macros::dont_look_inside]
extern "C" fn f32_convert(sel: i64, a: i64) -> i64 {
    let r = match sel {
        0 => crate::core::wasm::f32_convert_i32_s(a as i32),
        1 => crate::core::wasm::f32_convert_i32_u(a as u32),
        2 => crate::core::wasm::f32_convert_i64_s(a),
        _ => crate::core::wasm::f32_convert_i64_u(a as u64),
    };
    i64::from(r.to_bits())
}

/// Residual f32→f64 promotion (`F64PromoteF32`): reads f32 bits, widens to f64 bits
/// (into `freg64`). The exact promotion (incl. NaN payload) stays in the residual.
#[majit_macros::dont_look_inside]
extern "C" fn promote_f32_f64(a_bits: i64) -> i64 {
    crate::core::wasm::f64_promote_f32(f32::from_bits(a_bits as u32)).to_bits() as i64
}

/// Residual f64→f32 demotion (`F32DemoteF64`): reads f64 bits, narrows to f32 bits
/// (into `freg32`). The rounding / NaN handling stays in the residual.
#[majit_macros::dont_look_inside]
extern "C" fn demote_f64_f32(a_bits: i64) -> i64 {
    i64::from(crate::core::wasm::f32_demote_f64(f64::from_bits(a_bits as u64)).to_bits())
}

/// Saturating f32→integer truncation, `sel`-dispatched: 0=i32_s, 1=u32, 2=i64_s,
/// 3=u64. Never traps (NaN→0, out-of-range clamps). Reads the f32 bits, writes the
/// integer result into `ireg`.
#[majit_macros::dont_look_inside]
extern "C" fn f32_trunc_sat(sel: i64, a_bits: i64) -> i64 {
    let a = f32::from_bits(a_bits as u32);
    match sel {
        0 => crate::core::wasm::i32_trunc_sat_f32_s(a) as i64,
        1 => crate::core::wasm::i32_trunc_sat_f32_u(a) as i64,
        2 => crate::core::wasm::i64_trunc_sat_f32_s(a),
        _ => crate::core::wasm::i64_trunc_sat_f32_u(a) as i64,
    }
}

/// Trapping f32→integer truncation, `sel`-dispatched: 0=i32_s, 1=u32, 2=i64_s,
/// 3=u64. NaN / out-of-range trap via `set_residual_trap` (see `f64_trunc`).
#[majit_macros::dont_look_inside]
extern "C" fn f32_trunc(sel: i64, a_bits: i64) -> i64 {
    let a = f32::from_bits(a_bits as u32);
    let r = match sel {
        0 => crate::core::wasm::i32_trunc_f32_s(a).map(|v| v as i64),
        1 => crate::core::wasm::i32_trunc_f32_u(a).map(|v| v as i64),
        2 => crate::core::wasm::i64_trunc_f32_s(a),
        _ => crate::core::wasm::i64_trunc_f32_u(a).map(|v| v as i64),
    };
    match r {
        Ok(v) => v,
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}

/// Residual f64 comparisons, `sel`-dispatched: 0=lt, 1=le, 2=eq, 3=ne,
/// 4=lt swapped, 5=le swapped. Returns a 0/1 i32 value. Rust `<` / `<=` / `==` are
/// false for NaN and `!=` is true, matching wasm. One selector arm folds the
/// compare forms.
#[majit_macros::dont_look_inside]
extern "C" fn f64_cmp(sel: i64, a_bits: i64, b_bits: i64) -> i64 {
    let a = f64::from_bits(a_bits as u64);
    let b = f64::from_bits(b_bits as u64);
    let r = match sel {
        0 => a < b,
        1 => a <= b,
        2 => a == b,
        4 => b < a,
        5 => b <= a,
        _ => a != b,
    };
    r as i64
}
/// Residual f64 unary ops, `sel`-dispatched: 0=abs, 1=neg, 2=sqrt, 3=ceil,
/// 4=floor, 5=trunc, 6=nearest. The float rounding intrinsics differ between std
/// and no_std, so reuse the core crate's `wasm::f64_*` helpers (which pick std or
/// libm). The bit-casts stay inside the residual and never reach the trace; one
/// selector arm folds the seven ops (see `f64_arith`).
#[majit_macros::dont_look_inside]
extern "C" fn f64_unary(sel: i64, a_bits: i64) -> i64 {
    let a = f64::from_bits(a_bits as u64);
    let r = match sel {
        0 => crate::core::wasm::f64_abs(a),
        1 => crate::core::wasm::f64_neg(a),
        2 => crate::core::wasm::f64_sqrt(a),
        3 => crate::core::wasm::f64_ceil(a),
        4 => crate::core::wasm::f64_floor(a),
        5 => crate::core::wasm::f64_trunc(a),
        _ => crate::core::wasm::f64_nearest(a),
    };
    r.to_bits() as i64
}
/// Residual f64 binary combine, `sel`-dispatched: 0=min, 1=max,
/// 2=copysign(a,b), 3=copysign(b,a). min/max carry wasm semantics
/// (NaN-propagating, signed-zero aware); copysign takes the magnitude of the
/// first argument and the sign of the second. `sel` 3 swaps the operands for
/// the `_Rsr` form, where the magnitude operand is the slot and the sign is the
/// accumulator. The bit-casts stay inside the residual.
#[majit_macros::dont_look_inside]
extern "C" fn f64_minmax(sel: i64, a_bits: i64, b_bits: i64) -> i64 {
    let a = f64::from_bits(a_bits as u64);
    let b = f64::from_bits(b_bits as u64);
    let r = match sel {
        0 => crate::core::wasm::f64_min(a, b),
        1 => crate::core::wasm::f64_max(a, b),
        2 => crate::core::wasm::f64_copysign(a, b),
        _ => crate::core::wasm::f64_copysign(b, a),
    };
    r.to_bits() as i64
}
// Widening integer→f64 conversion, `sel`-dispatched: 0=i32_s, 1=i32_u (u32),
// 2=i64_s, 3=i64_u (u64). The slot/accumulator holds the integer in its low bits;
// the width/sign casts live inside the residual and never reach the trace. One
// selector folds the four (see `f64_arith`).
#[majit_macros::dont_look_inside]
extern "C" fn f64_convert(sel: i64, a: i64) -> i64 {
    let r = match sel {
        0 => crate::core::wasm::f64_convert_i32_s(a as i32),
        1 => crate::core::wasm::f64_convert_i32_u(a as u32),
        2 => crate::core::wasm::f64_convert_i64_s(a),
        _ => crate::core::wasm::f64_convert_i64_u(a as u64),
    };
    r.to_bits() as i64
}
// Saturating f64→integer truncation, `sel`-dispatched: 0=i32_s, 1=u32, 2=i64_s,
// 3=u64. Never traps (NaN→0, out-of-range clamps to the integer min/max). The
// `f64::from_bits` and result canonicalization stay in the residual.
#[majit_macros::dont_look_inside]
extern "C" fn f64_trunc_sat(sel: i64, a_bits: i64) -> i64 {
    let a = f64::from_bits(a_bits as u64);
    match sel {
        0 => crate::core::wasm::i32_trunc_sat_f64_s(a) as i64,
        1 => crate::core::wasm::i32_trunc_sat_f64_u(a) as i64,
        2 => crate::core::wasm::i64_trunc_sat_f64_s(a),
        _ => crate::core::wasm::i64_trunc_sat_f64_u(a) as i64,
    }
}
// Trapping f64→integer truncation, `sel`-dispatched: 0=i32_s, 1=u32, 2=i64_s,
// 3=u64. NaN and out-of-range inputs trap: the residual flags it via
// `set_residual_trap` (which also stops any later store) and returns 0; the run
// completes and `run_jit` surfaces the trap (re-running stock for a pure trap, or
// raising the recorded code directly when a store already committed).
#[majit_macros::dont_look_inside]
extern "C" fn f64_trunc(sel: i64, a_bits: i64) -> i64 {
    let a = f64::from_bits(a_bits as u64);
    let r = match sel {
        0 => crate::core::wasm::i32_trunc_f64_s(a).map(|v| v as i64),
        1 => crate::core::wasm::i32_trunc_f64_u(a).map(|v| v as i64),
        2 => crate::core::wasm::i64_trunc_f64_s(a),
        _ => crate::core::wasm::i64_trunc_f64_u(a).map(|v| v as i64),
    };
    match r {
        Ok(v) => v,
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
// Integer bit-count unary ops (never trap). The bit-width extraction (`as u32` /
// `as u64` picking the operand width from the low bits) and the `leading_zeros` /
// `trailing_zeros` / `count_ones` intrinsics stay inside the residual — the
// metainterp has no lowering for them, so they must not reach the trace. The
// result (`0..=32` / `0..=64`) is a small non-negative integer written to `ireg`.
// One selector residual per width folds clz/ctz/popcnt (sel 0/1/2), keeping the
// force-inlined dispatch pool small (see the selector-collapse note in the epic).
#[majit_macros::dont_look_inside]
extern "C" fn i32_bitcount(sel: i64, a: i64) -> i64 {
    let a = a as u32;
    i64::from(match sel {
        0 => a.leading_zeros(),
        1 => a.trailing_zeros(),
        _ => a.count_ones(),
    })
}
#[majit_macros::dont_look_inside]
extern "C" fn i64_bitcount(sel: i64, a: i64) -> i64 {
    let a = a as u64;
    i64::from(match sel {
        0 => a.leading_zeros(),
        1 => a.trailing_zeros(),
        _ => a.count_ones(),
    })
}
// Integer division / remainder. These CAN trap (division by zero, and signed
// `INT_MIN / -1` overflow), so they delegate to the same `core::wasm::*` helpers
// the stock executor uses (guaranteeing bit-exact parity) and, on a trap, latch
// the exact code via `set_residual_trap` and return 0 — `run_jit` then re-runs
// the function on the stock executor (pure) or raises the recorded code (if a
// store already committed), exactly like the trapping f64→int truncations. The
// unsigned forms canonicalize their narrow result to a sign-extended i64.
#[majit_macros::dont_look_inside]
extern "C" fn i32_div_s(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i32_div_s(lhs as i32, rhs as i32) {
        Ok(v) => i64::from(v),
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
#[majit_macros::dont_look_inside]
extern "C" fn i32_div_u(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i32_div_u(lhs as u32, rhs as u32) {
        Ok(v) => i64::from(v as i32),
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
#[majit_macros::dont_look_inside]
extern "C" fn i32_rem_s(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i32_rem_s(lhs as i32, rhs as i32) {
        Ok(v) => i64::from(v),
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
#[majit_macros::dont_look_inside]
extern "C" fn i32_rem_u(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i32_rem_u(lhs as u32, rhs as u32) {
        Ok(v) => i64::from(v as i32),
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
#[majit_macros::dont_look_inside]
extern "C" fn i64_div_s(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i64_div_s(lhs, rhs) {
        Ok(v) => v,
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
#[majit_macros::dont_look_inside]
extern "C" fn i64_div_u(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i64_div_u(lhs as u64, rhs as u64) {
        Ok(v) => v as i64,
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
#[majit_macros::dont_look_inside]
extern "C" fn i64_rem_s(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i64_rem_s(lhs, rhs) {
        Ok(v) => v,
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}
#[majit_macros::dont_look_inside]
extern "C" fn i64_rem_u(lhs: i64, rhs: i64) -> i64 {
    match crate::core::wasm::i64_rem_u(lhs as u64, rhs as u64) {
        Ok(v) => v as i64,
        Err(code) => {
            set_residual_trap(code);
            0
        }
    }
}

struct WasmKernelState {
    /// The wasm frame's cell slots (locals + stack), seeded from the caller's
    /// frame. Virtualizable so a CloseLoop guard deopt reads the live values.
    slots: Vec<i64>,
    /// Integer accumulator (`Reg<i64>`, `ireg`). Scalar state field so it
    /// compiles to a native register in the JIT trace, avoiding the
    /// virtualizable array overhead that split_dispatch sub-JitCodes incur.
    accum0: i64,
    /// f64 accumulator (`Reg<f64>`, `freg64`, held as raw i64 bits).
    accum1: i64,
    /// f32 accumulator (`Reg<f32>`, `freg32`, raw bits in the low 32).
    accum2: i64,
}

/// Stores a slot snapshot for [`MINI_YIELD_STOCK`]. Isolated from the kernel
/// loop so the `#[jit_interp]` proc macro does not need to parse RefCell
/// borrows in a closure that captures mutable state.
fn yield_set_slots(slots: Vec<i64>) {
    YIELD_SLOTS.with(|c| {
        *c.borrow_mut() = slots;
    });
}

#[majit_macros::jit_interp(
    state = WasmKernelState,
    env = MiniCode,
    calls = {
        mem_load_i64 => residual_int,
        mem_load_i32 => residual_int,
        mem_load_u8 => residual_int,
        mem_load_i8 => residual_int,
        mem_load_u16 => residual_int,
        mem_load_i16 => residual_int,
        mem_store_i32 => residual_void_cannot_raise,
        mem_store_i64 => residual_void_cannot_raise,
        mem_store_u8 => residual_void_cannot_raise,
        mem_store_u16 => residual_void_cannot_raise,
        f64_arith => residual_int,
        f64_cmp => residual_int,
        f64_unary => residual_int,
        f64_minmax => residual_int,
        f64_convert => residual_int,
        f64_trunc_sat => residual_int,
        f64_trunc => residual_int,
        i32_bitcount => residual_int,
        i64_bitcount => residual_int,
        i32_div_s => residual_int,
        i32_div_u => residual_int,
        i32_rem_s => residual_int,
        i32_rem_u => residual_int,
        i64_div_s => residual_int,
        i64_div_u => residual_int,
        i64_rem_s => residual_int,
        i64_rem_u => residual_int,
        f32_arith => residual_int,
        f32_minmax => residual_int,
        f32_cmp => residual_int,
        f32_unary => residual_int,
        f32_convert => residual_int,
        promote_f32_f64 => residual_int,
        demote_f64_f32 => residual_int,
        f32_trunc_sat => residual_int,
        f32_trunc => residual_int,
        global_get => residual_int,
        global_set => residual_void_cannot_raise,
        call_internal_residual => residual_int,
    },
    greens = [pc, program],
    state_fields = {
        slots: [int; virt],
        accum0: int,
        accum1: int,
        accum2: int,
    },
    // Route pure forward-advancing arms (copy / ALU / compare / select) through
    // per-arm sub-JitCodes that RETURN the advanced pc, so the dispatch JitCode
    // no longer holds every arm body and stays well under the 256 register/const
    // ceiling.  Branch, return, and residual-call arms stay force-inlined.
    split_dispatch = true,
    switch_dispatch = true,
)]
fn wasm_mainloop(
    mut driver: &mut majit_metainterp::JitDriver<WasmKernelState>,
    program: &MiniCode,
    init_slots: &[i64],
) -> i64 {
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = WasmKernelState {
        slots: init_slots.to_vec(),
        accum0: 0i64,
        accum1: 0i64,
        accum2: 0i64,
    };

    loop {
        jit_merge_point!();
        let op = program[pc];
        match op {
            MINI_I32_ADD_SI_WB => {
                let dst = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let imm = program[pc + 3];
                // i32 wrap-around via i64 shifts (sign-extend the low 32 bits).
                // `(x as i32).wrapping_add(..)` aborts the trace at this op; the
                // shift form uses only i64 add/shift, which the tracer compiles.
                let v = ((state.slots[lhs] + imm) << 32) >> 32;
                state.slots[dst] = v;
                state.accum0 = v;
                pc += 4;
            }
            MINI_BR_I32_NE_RI => {
                let tgt = program[pc + 1] as usize;
                let imm = program[pc + 2];
                if (state.accum0 as i32) != (imm as i32) {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 3;
            }
            MINI_RETURN_R => {
                return state.accum0;
            }
            MINI_RETURN_F_R => {
                // Return the f64 accumulator bits as the result slot value.
                return state.accum1;
            }
            MINI_RETURN_F32_R => {
                // Return the f32 accumulator bits (low 32) as the result.
                return state.accum2 & 0xFFFF_FFFF;
            }
            MINI_RETURN_VOID => {
                return 0;
            }
            MINI_COPY_SI => {
                let dst = program[pc + 1] as usize;
                state.slots[dst] = program[pc + 2];
                pc += 3;
            }
            MINI_COPY_SS => {
                let dst = program[pc + 1] as usize;
                let src = program[pc + 2] as usize;
                state.slots[dst] = state.slots[src];
                pc += 3;
            }
            MINI_COPY_SR => {
                let dst = program[pc + 1] as usize;
                state.slots[dst] = state.accum0;
                pc += 2;
            }
            MINI_COPY_RS => {
                let src = program[pc + 1] as usize;
                state.accum0 = state.slots[src];
                pc += 2;
            }
            MINI_COPY_RI => {
                let imm = program[pc + 1];
                state.accum0 = imm;
                pc += 2;
            }
            MINI_COPY_S_FR => {
                let dst = program[pc + 1] as usize;
                // Spill the f64 accumulator (`freg64`) — a bit-identical 64-bit move.
                state.slots[dst] = state.accum1;
                pc += 2;
            }
            MINI_I64_ADD_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // i64 add: plain `+` traces (release wraps mod 2^64 like wasm).
                state.accum0 = state.slots[lhs] + state.slots[rhs];
                pc += 3;
            }
            MINI_I64_ADD_SS_WB => {
                let dst = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                let v = state.slots[lhs] + state.slots[rhs];
                state.slots[dst] = v;
                state.accum0 = v;
                pc += 4;
            }
            MINI_I64_MUL_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // i64 mul: plain `*` traces (release wraps mod 2^64 like wasm).
                state.accum0 = state.slots[lhs] * state.slots[rhs];
                pc += 3;
            }
            MINI_I64_OR_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = state.slots[lhs] | state.slots[rhs];
                pc += 3;
            }
            MINI_I64_SUB_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // i64 sub: plain `-` traces (release wraps mod 2^64); non-commutative.
                state.accum0 = state.slots[lhs] - state.slots[rhs];
                pc += 3;
            }
            MINI_I32_MUL_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // i32 mul: sign-extend the low 32 bits (`as i32` aborts the trace).
                state.accum0 = ((state.slots[lhs] * state.slots[rhs]) << 32) >> 32;
                pc += 3;
            }
            MINI_I32_ADD_RS_WB => {
                let dst = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // i32 add of the accumulator and a slot, written back to both.
                let v = ((state.accum0 + state.slots[rhs]) << 32) >> 32;
                state.slots[dst] = v;
                state.accum0 = v;
                pc += 3;
            }
            MINI_I32_ADD_SS_WB => {
                let dst = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                // i32 add of two slots, written back to both. The shift form
                // sign-extends the low 32 bits; a narrowing cast aborts the trace.
                let v = ((state.slots[lhs] + state.slots[rhs]) << 32) >> 32;
                state.slots[dst] = v;
                state.accum0 = v;
                pc += 4;
            }
            MINI_I32_XOR_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Sign-extend the low 32 bits so the result is a canonical i32.
                state.accum0 = ((state.slots[lhs] ^ state.slots[rhs]) << 32) >> 32;
                pc += 3;
            }
            MINI_I32_AND_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = ((state.slots[lhs] & state.slots[rhs]) << 32) >> 32;
                pc += 3;
            }
            MINI_I32_OR_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = ((state.slots[lhs] | state.slots[rhs]) << 32) >> 32;
                pc += 3;
            }
            MINI_I32_SUB_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // i32 sub (non-commutative), sign-extended low 32 bits.
                state.accum0 = ((state.slots[lhs] - state.slots[rhs]) << 32) >> 32;
                pc += 3;
            }
            MINI_BR_U32_LE_SS => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                // Unsigned i32 `<=`: mask to the low 32 bits (a non-negative i64),
                // then compare as i64 — signed order matches unsigned order on
                // 0..2^32-1. `&`/`<=` trace; a narrowing `as u32` cast would not.
                if (state.slots[lhs] & 0xFFFF_FFFF) <= (state.slots[rhs] & 0xFFFF_FFFF) {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_ALWAYS => {
                let tgt = program[pc + 1] as usize;
                if tgt < pc {
                    can_enter_jit!(driver, tgt, &mut state, program, || {});
                }
                pc = tgt;
                continue;
            }
            MINI_BR_I64_EQ_SS => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                if state.slots[lhs] == state.slots[rhs] {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_I64_LE_SS => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                // Signed i64 `<=`: plain comparison traces.
                if state.slots[lhs] <= state.slots[rhs] {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_I64_LE_SI => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let imm = program[pc + 3];
                // Signed i64 `<=` against an immediate.
                if state.slots[lhs] <= imm {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_I32_LE_SS => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                // Signed i32 `<=`: sign-extend both low-32-bit operands so the
                // i64 compare matches i32 signed order.
                if ((state.slots[lhs] << 32) >> 32) <= ((state.slots[rhs] << 32) >> 32) {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_I32_LT_SI => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let imm = program[pc + 3];
                // Signed i32 `<`: sign-extend the low 32 bits of the slot (the
                // immediate is pre-sign-extended) so the i64 compare matches i32
                // signed order. The shift traces; a narrowing cast does not.
                if ((state.slots[lhs] << 32) >> 32) < imm {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_I64_LT_IR => {
                let tgt = program[pc + 1] as usize;
                let imm = program[pc + 2];
                // Signed i64 `imm < ireg`: the accumulator is the right operand.
                if imm < state.accum0 {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 3;
            }
            MINI_I64_XOR_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = state.slots[lhs] ^ state.slots[rhs];
                pc += 3;
            }
            MINI_I64_AND_RI_WR => {
                let imm = program[pc + 1];
                state.accum0 = state.accum0 & imm;
                pc += 2;
            }
            MINI_I64_OR_RI_WR => {
                let imm = program[pc + 1];
                state.accum0 = state.accum0 | imm;
                pc += 2;
            }
            MINI_I64_AND_SI_WR => {
                let lhs = program[pc + 1] as usize;
                let imm = program[pc + 2];
                state.accum0 = state.slots[lhs] & imm;
                pc += 3;
            }
            MINI_I64_ADD_RS_WB => {
                let dst = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                let v = state.accum0 + state.slots[rhs];
                state.slots[dst] = v;
                state.accum0 = v;
                pc += 3;
            }
            MINI_I64_SHL_SI => {
                let src = program[pc + 1] as usize;
                let shift = program[pc + 2];
                // Left shift zero-fills from the right; `<<` traces directly.
                state.accum0 = state.slots[src] << shift;
                pc += 3;
            }
            MINI_I32_SHL_SI => {
                let src = program[pc + 1] as usize;
                let shift = program[pc + 2];
                // i32 left shift; re-canonicalize the low 32 bits.
                state.accum0 = ((state.slots[src] << shift) << 32) >> 32;
                pc += 3;
            }
            MINI_I32_ROTL_SI => {
                let src = program[pc + 1] as usize;
                let k = program[pc + 2];
                // i32 rotate-left by k in 1..=31. Work on the zero-extended low 32
                // bits so `>>` zero-fills the wrapped bits; `<< 32 >> 32` discards
                // the bits `<< k` carried past bit 31 and re-canonicalizes.
                let x = state.slots[src] & 0xFFFF_FFFF;
                state.accum0 = (((x << k) | (x >> (32 - k))) << 32) >> 32;
                pc += 3;
            }
            MINI_I32_ROTR_SI => {
                let src = program[pc + 1] as usize;
                let k = program[pc + 2];
                // i32 rotate-right by k in 1..=31, dual to the rotate-left arm.
                let x = state.slots[src] & 0xFFFF_FFFF;
                state.accum0 = (((x >> k) | (x << (32 - k))) << 32) >> 32;
                pc += 3;
            }
            MINI_I32_BITCOUNT_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // i32 bit count (sel: 0=clz, 1=ctz, 2=popcnt) of the low 32 bits.
                state.accum0 = i32_bitcount(sel, state.slots[src]);
                pc += 3;
            }
            MINI_I64_BITCOUNT_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // i64 bit count (sel: 0=clz, 1=ctz, 2=popcnt).
                state.accum0 = i64_bitcount(sel, state.slots[src]);
                pc += 3;
            }
            MINI_I32_DIV_S => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i32_div_s(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_I32_DIV_U => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i32_div_u(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_I32_REM_S => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i32_rem_s(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_I32_REM_U => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i32_rem_u(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_I64_DIV_S => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i64_div_s(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_I64_DIV_U => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i64_div_u(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_I64_REM_S => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i64_rem_s(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_I64_REM_U => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 = i64_rem_u(state.slots[lhs], state.slots[rhs]);
                pc += 3;
            }
            MINI_U32_SHR_RI => {
                let shift = program[pc + 1];
                // i32 logical right shift of the accumulator: mask to the low 32
                // bits (zero-fill), shift, then sign-extend to a canonical i32.
                state.accum0 = (((state.accum0 & 0xFFFF_FFFF) >> shift) << 32) >> 32;
                pc += 2;
            }
            MINI_I32_LT_SI_R => {
                let lhs = program[pc + 1] as usize;
                let imm = program[pc + 2];
                // Compare RESULT (not a branch): a 0/1 value into the accumulator.
                state.accum0 = if ((state.slots[lhs] << 32) >> 32) < imm {
                    1
                } else {
                    0
                };
                pc += 3;
            }
            MINI_I32_LT_RS_R => {
                let rhs = program[pc + 1] as usize;
                // Signed i32 compare RESULT with the left operand in the
                // accumulator: a 0/1 value (both operands sign-extended).
                let l = (state.accum0 << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l < r { 1 } else { 0 };
                pc += 2;
            }
            MINI_I32_LT_SR_R => {
                let lhs = program[pc + 1] as usize;
                // Signed i32 compare RESULT with the right operand in the
                // accumulator: a 0/1 value (both operands sign-extended).
                let l = (state.slots[lhs] << 32) >> 32;
                let r = (state.accum0 << 32) >> 32;
                state.accum0 = if l < r { 1 } else { 0 };
                pc += 2;
            }
            MINI_SELECT => {
                let true_slot = program[pc + 1] as usize;
                let false_slot = program[pc + 2] as usize;
                // Branchless `select`: a value-`if` over non-constant arms aborts
                // the trace, so normalize the condition (low 32 bits non-zero) to
                // `c ∈ {0, 1}` via the compare-result form, then `f + (t - f) * c`
                // — exact in wrapping i64.
                let c = if (state.accum0 & 0xFFFF_FFFF) != 0 {
                    1
                } else {
                    0
                };
                let t = state.slots[true_slot];
                let f = state.slots[false_slot];
                state.accum0 = f + (t - f) * c;
                pc += 3;
            }
            MINI_I32_EQ_RS_R => {
                let rhs = program[pc + 1] as usize;
                // i32 equality RESULT (0/1) with the left operand in the
                // accumulator; both operands sign-extended from their low 32 bits.
                let l = (state.accum0 << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l == r { 1 } else { 0 };
                pc += 2;
            }
            MINI_I32_NE_RS_R => {
                let rhs = program[pc + 1] as usize;
                // i32 inequality RESULT (0/1) with the left operand in the
                // accumulator; both operands sign-extended from their low 32 bits.
                let l = (state.accum0 << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l != r { 1 } else { 0 };
                pc += 2;
            }
            MINI_I64_EQ_RS_R => {
                let rhs = program[pc + 1] as usize;
                // Full i64 equality RESULT (0/1), left operand in the accumulator.
                state.accum0 = if state.accum0 == state.slots[rhs] {
                    1
                } else {
                    0
                };
                pc += 2;
            }
            MINI_I64_NE_RS_R => {
                let rhs = program[pc + 1] as usize;
                // Full i64 inequality RESULT (0/1), left operand in the accumulator.
                state.accum0 = if state.accum0 != state.slots[rhs] {
                    1
                } else {
                    0
                };
                pc += 2;
            }
            MINI_I64_LT_RS_R => {
                let rhs = program[pc + 1] as usize;
                // Signed i64 less-than RESULT (0/1), left operand in the accumulator.
                state.accum0 = if state.accum0 < state.slots[rhs] {
                    1
                } else {
                    0
                };
                pc += 2;
            }
            MINI_I32_LE_RS_R => {
                let rhs = program[pc + 1] as usize;
                // Signed i32 `<=` RESULT (0/1), left operand in the accumulator;
                // both operands sign-extended from their low 32 bits.
                let l = (state.accum0 << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l <= r { 1 } else { 0 };
                pc += 2;
            }
            MINI_U32_LT_RS_R => {
                let rhs = program[pc + 1] as usize;
                // Unsigned i32 `<` RESULT (0/1): mask both operands to their low 32
                // bits (non-negative i64) so the signed `<` realizes unsigned order.
                let l = state.accum0 & 0xFFFF_FFFF;
                let r = state.slots[rhs] & 0xFFFF_FFFF;
                state.accum0 = if l < r { 1 } else { 0 };
                pc += 2;
            }
            MINI_U32_LE_RS_R => {
                let rhs = program[pc + 1] as usize;
                // Unsigned i32 `<=` RESULT (0/1): masked low 32 bits of both.
                let l = state.accum0 & 0xFFFF_FFFF;
                let r = state.slots[rhs] & 0xFFFF_FFFF;
                state.accum0 = if l <= r { 1 } else { 0 };
                pc += 2;
            }
            MINI_I32_EQ_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Signed i32 equality RESULT: sign-extend both low 32-bit slots.
                let l = (state.slots[lhs] << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l == r { 1 } else { 0 };
                pc += 3;
            }
            MINI_I32_NE_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Signed i32 inequality RESULT: sign-extend both low 32-bit slots.
                let l = (state.slots[lhs] << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l != r { 1 } else { 0 };
                pc += 3;
            }
            MINI_I32_LT_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Signed i32 compare RESULT: sign-extend both low 32-bit slots.
                let l = (state.slots[lhs] << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l < r { 1 } else { 0 };
                pc += 3;
            }
            MINI_I32_LE_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Signed i32 compare RESULT: sign-extend both low 32-bit slots.
                let l = (state.slots[lhs] << 32) >> 32;
                let r = (state.slots[rhs] << 32) >> 32;
                state.accum0 = if l <= r { 1 } else { 0 };
                pc += 3;
            }
            MINI_U32_LT_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Unsigned i32 `<` RESULT: masked low 32 bits of both slots.
                let l = state.slots[lhs] & 0xFFFF_FFFF;
                let r = state.slots[rhs] & 0xFFFF_FFFF;
                state.accum0 = if l < r { 1 } else { 0 };
                pc += 3;
            }
            MINI_U32_LE_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Unsigned i32 `<=` RESULT: masked low 32 bits of both slots.
                let l = state.slots[lhs] & 0xFFFF_FFFF;
                let r = state.slots[rhs] & 0xFFFF_FFFF;
                state.accum0 = if l <= r { 1 } else { 0 };
                pc += 3;
            }
            MINI_I64_EQ_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Full i64 equality RESULT: compare both slots as-is.
                state.accum0 = if state.slots[lhs] == state.slots[rhs] {
                    1
                } else {
                    0
                };
                pc += 3;
            }
            MINI_I64_NE_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Full i64 inequality RESULT: compare both slots as-is.
                state.accum0 = if state.slots[lhs] != state.slots[rhs] {
                    1
                } else {
                    0
                };
                pc += 3;
            }
            MINI_I64_LT_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Signed i64 compare RESULT: `slots[lhs] < slots[rhs]` as 0/1.
                state.accum0 = if state.slots[lhs] < state.slots[rhs] {
                    1
                } else {
                    0
                };
                pc += 3;
            }
            MINI_I64_LE_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Signed i64 compare RESULT: `slots[lhs] <= slots[rhs]` as 0/1.
                state.accum0 = if state.slots[lhs] <= state.slots[rhs] {
                    1
                } else {
                    0
                };
                pc += 3;
            }
            MINI_U64_LT_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Flipping the sign bit realizes unsigned i64 order.
                let flip = U64_ORDER_FLIP;
                let l = state.slots[lhs] ^ flip;
                let r = state.slots[rhs] ^ flip;
                state.accum0 = if l < r { 1 } else { 0 };
                pc += 3;
            }
            MINI_U64_LE_SS_R => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                // Flipping the sign bit realizes unsigned i64 order.
                let flip = U64_ORDER_FLIP;
                let l = state.slots[lhs] ^ flip;
                let r = state.slots[rhs] ^ flip;
                state.accum0 = if l <= r { 1 } else { 0 };
                pc += 3;
            }
            MINI_I64_LT_IS_R => {
                let imm = program[pc + 1];
                let rhs = program[pc + 2] as usize;
                // Signed i64 compare RESULT: `imm < slots[rhs]` as a 0/1 value.
                state.accum0 = if imm < state.slots[rhs] { 1 } else { 0 };
                pc += 3;
            }
            MINI_I64_LT_SI_R => {
                let lhs = program[pc + 1] as usize;
                let imm = program[pc + 2];
                // Signed i64 compare RESULT: `slots[lhs] < imm` as a 0/1 value.
                state.accum0 = if state.slots[lhs] < imm { 1 } else { 0 };
                pc += 3;
            }
            MINI_I64_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                // The dynamic address is the accumulator (an unsigned 32-bit wasm
                // address); add the static offset, then load via the residual
                // (which reads the memory base/len from the thread-local context).
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum0 = mem_load_i64(ea);
                pc += 2;
            }
            MINI_F64_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                // Same 8-byte read as `MINI_I64_LOAD_MEM0_OFF`, but the address is
                // the integer accumulator and the f64 bits land in `freg64`.
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum1 = mem_load_i64(ea);
                pc += 2;
            }
            MINI_F64_ARITH_RS => {
                let sel = program[pc + 1];
                let rhs = program[pc + 2] as usize;
                // f64 `acc OP slot` (sel: 0=add, 1=sub, 2=mul, 3=div). The slot and
                // the f64 accumulator (`freg64`) hold f64 bit patterns; the bit-casts
                // and the operation live inside `f64_arith`. Add is commutative, so an
                // `acc = slot + acc` form maps here as `acc OP slot`; sub/div take the
                // accumulator as the left operand (wasm order).
                state.accum1 = f64_arith(sel, state.accum1, state.slots[rhs]);
                pc += 3;
            }
            MINI_F32_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                // 4-byte read (reuses the bounds-checked i32 load); the f32 bits
                // land in the f32 accumulator (`freg32`, low 32). The address is
                // the integer accumulator.
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum2 = mem_load_i32(ea);
                pc += 2;
            }
            MINI_F32_STORE_SR => {
                let ptr_slot = program[pc + 1] as usize;
                let offset = program[pc + 2];
                // 4-byte store of the f32 accumulator (`freg32`); reuses the i32
                // store, which writes the low 4 bytes of the value.
                let ea = (state.slots[ptr_slot] & 0xFFFF_FFFF) + offset;
                mem_store_i32(ea, state.accum2);
                pc += 3;
            }
            MINI_COPY_S_F32R => {
                let dst = program[pc + 1] as usize;
                // Spill the f32 accumulator (`freg32`) — a 32-bit move (the high
                // bits are unused; stores/ops take the low 32).
                state.slots[dst] = state.accum2;
                pc += 2;
            }
            MINI_F32_ARITH_RS => {
                let sel = program[pc + 1];
                let rhs = program[pc + 2] as usize;
                // f32 `acc OP slot` (sel: 0=add, 1=sub, 2=mul, 3=div). The slot and
                // the f32 accumulator (`freg32`) hold f32 bit patterns; the bit-casts
                // and the operation live inside `f32_arith`. Add is commutative, so an
                // `acc = slot + acc` form maps here as `acc OP slot`; sub/div take the
                // accumulator as the left operand (wasm order).
                state.accum2 = f32_arith(sel, state.accum2, state.slots[rhs]);
                pc += 3;
            }
            MINI_F32_MINMAX_RS => {
                let sel = program[pc + 1];
                let rhs = program[pc + 2] as usize;
                // f32 min/max/copysign of the accumulator (`freg32`) and a slot
                // (sel: 0=min, 1=max, 2=copysign(acc,slot), 3=copysign(slot,acc)).
                // The bit-casts and wasm semantics live inside `f32_minmax`.
                state.accum2 = f32_minmax(sel, state.accum2, state.slots[rhs]);
                pc += 3;
            }
            MINI_F32_CMP_RS_R => {
                let sel = program[pc + 1];
                let rhs = program[pc + 2] as usize;
                // f32 compare RESULT (0/1, an integer) with the left operand in the
                // f32 accumulator; the 0/1 result lands in the integer accumulator.
                // sel: 0=lt, 1=le, 2=eq, 3=ne.
                state.accum0 = f32_cmp(sel, state.accum2, state.slots[rhs]);
                pc += 3;
            }
            MINI_F32_UNARY_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // f32 unary from a slot into the f32 accumulator (`freg32`). sel:
                // 0=abs, 1=neg, 2=sqrt, 3=ceil, 4=floor, 5=trunc, 6=nearest.
                state.accum2 = f32_unary(sel, state.slots[src]);
                pc += 3;
            }
            MINI_F32_CVT_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // integer (slot) → f32 (`freg32`). sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
                state.accum2 = f32_convert(sel, state.slots[src]);
                pc += 3;
            }
            MINI_F64_PROMOTE_S => {
                let src = program[pc + 1] as usize;
                // f32 (slot) → f64 (`freg64`).
                state.accum1 = promote_f32_f64(state.slots[src]);
                pc += 2;
            }
            MINI_F32_DEMOTE_S => {
                let src = program[pc + 1] as usize;
                // f64 (slot) → f32 (`freg32`).
                state.accum2 = demote_f64_f32(state.slots[src]);
                pc += 2;
            }
            MINI_I32_REINTERP_F32 => {
                // i32.reinterpret_f32 accumulator form: the f32 bit pattern in
                // `freg32` becomes an i32 in `ireg` (sign-extended, the kernel's i32
                // canonical form). A pure bit move — no residual.
                state.accum0 = (state.accum2 as i32) as i64;
                pc += 1;
            }
            MINI_F32_REINTERP_I32 => {
                // f32.reinterpret_i32 accumulator form: the i32 bit pattern in `ireg`
                // becomes an f32 in `freg32` (low 32 bits, zero-extended).
                state.accum2 = state.accum0 & 0xFFFF_FFFF;
                pc += 1;
            }
            MINI_I64_REINTERP_F64 => {
                // i64.reinterpret_f64: the f64 bit pattern in `freg64` → `ireg`.
                // Both are 64-bit, so a plain copy.
                state.accum0 = state.accum1;
                pc += 1;
            }
            MINI_F64_REINTERP_I64 => {
                // f64.reinterpret_i64: the i64 bit pattern in `ireg` → `freg64`.
                state.accum1 = state.accum0;
                pc += 1;
            }
            MINI_F32_TRUNC_SAT_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // saturating f32→int (never traps): read the f32 slot, write `ireg`.
                // sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
                state.accum0 = f32_trunc_sat(sel, state.slots[src]);
                pc += 3;
            }
            MINI_F32_TRUNC_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // trapping f32→int: read the f32 slot, write `ireg` (or latch a trap).
                // sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
                state.accum0 = f32_trunc(sel, state.slots[src]);
                pc += 3;
            }
            MINI_GLOBAL_GET_R => {
                let idx = program[pc + 1];
                // Read integer global `idx`'s raw bits into the accumulator.
                state.accum0 = global_get(idx);
                pc += 2;
            }
            MINI_GLOBAL_GET_F32 => {
                let idx = program[pc + 1];
                // Read an f32 global's raw bits (low 32) into the f32 accumulator.
                state.accum2 = global_get(idx) & 0xFFFF_FFFF;
                pc += 2;
            }
            MINI_GLOBAL_GET_F64 => {
                let idx = program[pc + 1];
                // Read an f64 global's raw bits into the f64 accumulator.
                state.accum1 = global_get(idx);
                pc += 2;
            }
            MINI_GLOBAL_SET_S => {
                let idx = program[pc + 1];
                let src = program[pc + 2] as usize;
                // Write a slot's raw bits into integer global `idx`.
                global_set(idx, state.slots[src]);
                pc += 3;
            }
            MINI_F64_CMP_RS_R => {
                let sel = program[pc + 1];
                let rhs = program[pc + 2] as usize;
                // f64 compare RESULT (0/1, an integer) with the left operand in the
                // f64 accumulator; the 0/1 result lands in the integer accumulator.
                // sel: 0=lt, 1=le, 2=eq, 3=ne.
                state.accum0 = f64_cmp(sel, state.accum1, state.slots[rhs]);
                pc += 3;
            }
            MINI_F64_UNARY_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // f64 unary from a slot into the f64 accumulator. sel: 0=abs, 1=neg,
                // 2=sqrt, 3=ceil, 4=floor, 5=trunc, 6=nearest.
                state.accum1 = f64_unary(sel, state.slots[src]);
                pc += 3;
            }
            MINI_F64_MINMAX_RS => {
                let sel = program[pc + 1];
                let rhs = program[pc + 2] as usize;
                // f64 `min`/`max(acc, slot)` (sel: 0=min, 1=max).
                state.accum1 = f64_minmax(sel, state.accum1, state.slots[rhs]);
                pc += 3;
            }
            MINI_F64_CVT_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // int→f64 widening: read the integer slot, write the f64 accumulator.
                // sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
                state.accum1 = f64_convert(sel, state.slots[src]);
                pc += 3;
            }
            MINI_F64_TRUNC_SAT_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // saturating f64→int (never traps): read the f64 slot, write `ireg`.
                // sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
                state.accum0 = f64_trunc_sat(sel, state.slots[src]);
                pc += 3;
            }
            MINI_F64_TRUNC_S => {
                let sel = program[pc + 1];
                let src = program[pc + 2] as usize;
                // trapping f64→int: read the f64 slot, write `ireg` (or latch a trap).
                // sel: 0=i32_s, 1=u32, 2=i64_s, 3=u64.
                state.accum0 = f64_trunc(sel, state.slots[src]);
                pc += 3;
            }
            MINI_I32_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum0 = mem_load_i32(ea);
                pc += 2;
            }
            MINI_U8_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum0 = mem_load_u8(ea);
                pc += 2;
            }
            MINI_I8_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum0 = mem_load_i8(ea);
                pc += 2;
            }
            MINI_U16_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum0 = mem_load_u16(ea);
                pc += 2;
            }
            MINI_I16_LOAD_MEM0_OFF => {
                let offset = program[pc + 1];
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                state.accum0 = mem_load_i16(ea);
                pc += 2;
            }
            MINI_I32_STORE_SR => {
                let ptr_slot = program[pc + 1] as usize;
                let offset = program[pc + 2];
                let ea = (state.slots[ptr_slot] & 0xFFFF_FFFF) + offset;
                mem_store_i32(ea, state.accum0);
                pc += 3;
            }
            MINI_I64_STORE_SR => {
                let ptr_slot = program[pc + 1] as usize;
                let offset = program[pc + 2];
                let ea = (state.slots[ptr_slot] & 0xFFFF_FFFF) + offset;
                mem_store_i64(ea, state.accum0);
                pc += 3;
            }
            MINI_F64_STORE_SR => {
                let ptr_slot = program[pc + 1] as usize;
                let offset = program[pc + 2];
                // Same 8-byte store as `MINI_I64_STORE_SR`, but the value is the
                // f64 accumulator (`freg64`) rather than the integer one.
                let ea = (state.slots[ptr_slot] & 0xFFFF_FFFF) + offset;
                mem_store_i64(ea, state.accum1);
                pc += 3;
            }
            MINI_I32_STORE8_SR => {
                let ptr_slot = program[pc + 1] as usize;
                let offset = program[pc + 2];
                let ea = (state.slots[ptr_slot] & 0xFFFF_FFFF) + offset;
                mem_store_u8(ea, state.accum0);
                pc += 3;
            }
            MINI_I32_STORE16_SR => {
                let ptr_slot = program[pc + 1] as usize;
                let offset = program[pc + 2];
                let ea = (state.slots[ptr_slot] & 0xFFFF_FFFF) + offset;
                mem_store_u16(ea, state.accum0);
                pc += 3;
            }
            MINI_I32_STORE_RS => {
                let offset = program[pc + 1];
                let val_slot = program[pc + 2] as usize;
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                mem_store_i32(ea, state.slots[val_slot]);
                pc += 3;
            }
            MINI_I64_STORE_RS => {
                let offset = program[pc + 1];
                let val_slot = program[pc + 2] as usize;
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                mem_store_i64(ea, state.slots[val_slot]);
                pc += 3;
            }
            MINI_I32_STORE8_RS => {
                let offset = program[pc + 1];
                let val_slot = program[pc + 2] as usize;
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                mem_store_u8(ea, state.slots[val_slot]);
                pc += 3;
            }
            MINI_I32_STORE16_RS => {
                let offset = program[pc + 1];
                let val_slot = program[pc + 2] as usize;
                let ea = (state.accum0 & 0xFFFF_FFFF) + offset;
                mem_store_u16(ea, state.slots[val_slot]);
                pc += 3;
            }
            MINI_I64_SEXT32 => {
                // i64.extend_i32_s: sign-extend the low 32 bits.
                state.accum0 = (state.accum0 << 32) >> 32;
                pc += 1;
            }
            MINI_I64_SEXT32_S => {
                // i64.extend_i32_s of a slot: sign-extend its low 32 bits.
                let src = program[pc + 1] as usize;
                state.accum0 = (state.slots[src] << 32) >> 32;
                pc += 2;
            }
            MINI_U64_SHR_SI => {
                let src = program[pc + 1] as usize;
                let shift = program[pc + 2];
                let mask = program[pc + 3];
                // Logical shift-right: the arithmetic `>>` sign-extends, so mask
                // off the high `shift` bits to reproduce the zero-fill. Both `>>`
                // (constant shift) and `&` trace.
                state.accum0 = (state.slots[src] >> shift) & mask;
                pc += 4;
            }
            MINI_BR_I64_EQ_SI => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let imm = program[pc + 3];
                if state.slots[lhs] == imm {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_I64_NE_RI => {
                let tgt = program[pc + 1] as usize;
                let imm = program[pc + 2];
                if state.accum0 != imm {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 3;
            }
            MINI_BR_I64_NE_SS => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                if state.slots[lhs] != state.slots[rhs] {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_BR_U64_LT_SS => {
                let tgt = program[pc + 1] as usize;
                let lhs = program[pc + 2] as usize;
                let rhs = program[pc + 3] as usize;
                let flip = U64_ORDER_FLIP;
                if (state.slots[lhs] ^ flip) < (state.slots[rhs] ^ flip) {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            MINI_RETURN_S => {
                let src = program[pc + 1] as usize;
                return state.slots[src];
            }
            MINI_CALL_RESIDUAL => {
                // Execute an internal function call via the registered call
                // runner. Stage params into the fixed-size TLS buffer (no heap
                // allocation), call the residual, store the return value in
                // the params area head slot — wasmi's calling convention
                // places the result at params.span().head(), and the
                // translator's subsequent instructions read it from there.
                let func_addr = program[pc + 1];
                let params_start = program[pc + 2] as usize;
                let params_len = program[pc + 3] as usize;
                let mut buf = [0i64; MAX_CALL_PARAMS];
                let n = if params_len < MAX_CALL_PARAMS {
                    params_len
                } else {
                    MAX_CALL_PARAMS
                };
                let mut i = 0;
                while i < n {
                    buf[i] = state.slots[params_start + i];
                    i += 1;
                }
                CALL_STAGING.with(|c| c.set((buf, n)));
                let result = call_internal_residual(func_addr, n as i64);
                state.slots[params_start] = result;
                state.accum0 = result;
                pc += 4;
            }
            MINI_CALL_IMPORTED => {
                // Execute an imported function call. Same param staging
                // as MINI_CALL_RESIDUAL, but delegates to the imported-call
                // runner which resolves the Func through the store.
                let func_index = program[pc + 1];
                let params_start = program[pc + 2] as usize;
                let params_len = program[pc + 3] as usize;
                let mut buf = [0i64; MAX_CALL_PARAMS];
                let n = if params_len < MAX_CALL_PARAMS {
                    params_len
                } else {
                    MAX_CALL_PARAMS
                };
                let mut i = 0;
                while i < n {
                    buf[i] = state.slots[params_start + i];
                    i += 1;
                }
                CALL_STAGING.with(|c| c.set((buf, n)));
                let result = call_imported_residual(func_index, n as i64);
                state.slots[params_start] = result;
                state.accum0 = result;
                pc += 4;
            }
            MINI_CALL_INDIRECT => {
                // Execute an indirect function call. Reads the runtime
                // table index from a slot, stages params, and delegates
                // to the indirect-call runner for table lookup + type check.
                let table = program[pc + 1];
                let func_type = program[pc + 2];
                let index_slot = program[pc + 3] as usize;
                let params_start = program[pc + 4] as usize;
                let params_len = program[pc + 5] as usize;
                let runtime_index = state.slots[index_slot];
                let mut buf = [0i64; MAX_CALL_PARAMS];
                let n = if params_len < MAX_CALL_PARAMS {
                    params_len
                } else {
                    MAX_CALL_PARAMS
                };
                let mut i = 0;
                while i < n {
                    buf[i] = state.slots[params_start + i];
                    i += 1;
                }
                CALL_STAGING.with(|c| c.set((buf, n)));
                let result =
                    call_indirect_residual(table, func_type, runtime_index, n as i64);
                state.slots[params_start] = result;
                state.accum0 = result;
                pc += 6;
            }
            MINI_TRAP => {
                // Unconditional trap (wasm `unreachable`). Set the trap code
                // and return so run_jit can surface the trap.
                let code = program[pc + 1];
                set_residual_trap(
                    crate::TrapCode::try_from(code as u8)
                        .unwrap_or(crate::TrapCode::UnreachableCodeReached),
                );
                return 0;
            }
            MINI_MEMORY_SIZE => {
                // Return memory size in pages (mem_len / 65536) into ireg.
                let (_base, len) = MEM_CTX.with(|c| c.get());
                state.accum0 = if len > 0 { len / 65536 } else { 0 };
                pc += 1;
            }
            MINI_U64_SHR_SS_WR => {
                let lhs = program[pc + 1] as usize;
                let rhs = program[pc + 2] as usize;
                state.accum0 =
                    ((state.slots[lhs] as u64) >> ((state.slots[rhs] as u64) & 63)) as i64;
                pc += 3;
            }
            MINI_MEM_COPY_WITHIN => {
                let dst_slot = program[pc + 1] as usize;
                let src_slot = program[pc + 2] as usize;
                let len_slot = program[pc + 3] as usize;
                mem_copy_within(
                    state.slots[dst_slot],
                    state.slots[src_slot],
                    state.slots[len_slot],
                );
                pc += 4;
            }
            MINI_YIELD_STOCK => {
                // Yield to the stock executor at the recorded byte offset.
                // The caller flushes the slot snapshot to the real frame and
                // resumes the stock executor at the CallInternal instruction.
                let byte_offset = program[pc + 1];
                let num_slots = program[pc + 2] as usize;
                YIELD_TO_STOCK.with(|y| y.set(true));
                YIELD_BYTE_OFFSET.with(|c| c.set(byte_offset));
                let mut slots_copy = alloc::vec![0i64; num_slots];
                let mut i = 0;
                while i < num_slots {
                    slots_copy[i] = state.slots[i];
                    i += 1;
                }
                yield_set_slots(slots_copy);
                return 0;
            }
            MINI_RETURN_BAIL => {
                // Signal the caller (run_jit) to fall back to stock executor.
                BAIL_TO_STOCK.with(|b| b.set(true));
                return 0;
            }
            // MINI_HALT / any other word: ineligible at runtime (should not
            // happen for a statically-eligible MiniProgram).
            _ => break,
        }
    }
    i64::MIN
}

/// Hot-count threshold for the persistent driver. Counts merge-point visits
/// *across* calls, so a function with a short per-call loop still warms up when
/// called repeatedly.
const THRESHOLD: u32 = 4;

std::thread_local! {
    /// One persistent driver shared by every wasm function on this thread.
    ///
    /// The `#[jit_interp]` green key includes the program pointer
    /// (`program.as_ptr()`), so each function's compiled loop is keyed distinctly
    /// within this single driver. Persisting it across calls is what makes the
    /// JIT tier viable for repeated short calls: `JitDriver::new` spawns a
    /// background invalidation thread and starts with an empty compiled-loop
    /// table, so a fresh driver per call both leaks threads and recompiles every
    /// call (the per-call-driver shape panics under load).
    static DRIVER: core::cell::RefCell<Option<majit_metainterp::JitDriver<WasmKernelState>>> =
        core::cell::RefCell::new(None);

    /// Persistent driver for callees executed via CALL_ASSEMBLER (the
    /// `run_callee` path). Separate from [`DRIVER`] so a callee can run the
    /// MiniProgram dispatch while the caller's `run_persistent` still holds
    /// DRIVER's borrow. Uses the same compile threshold so the callee's hot
    /// loop compiles and is reused across calls.
    static CALLEE_DRIVER: core::cell::RefCell<Option<majit_metainterp::JitDriver<WasmKernelState>>> =
        core::cell::RefCell::new(None);

    /// Per-function cache keyed by the compiled function's op-stream pointer: the
    /// prepassed MiniProgram plus its adaptive tier policy. Prepass runs once per
    /// function, and the cached `words` Vec gives the program a stable heap
    /// address so the driver's green-keyed compiled loop is reused across calls.
    /// `None` marks a function the prepass rejected. Folding the tier policy in
    /// here (rather than a parallel map) lets one lookup serve both the prepass
    /// and the per-call tier decision. See [`TierPolicy`].
    static PROGRAMS: core::cell::RefCell<
        std::collections::HashMap<usize, Option<CachedFunc>>,
    > = core::cell::RefCell::new(std::collections::HashMap::new());
}

/// A JIT-eligible function's cached prepass output and adaptive tier policy.
struct CachedFunc {
    program: super::prepass::MiniProgram,
    policy: TierPolicy,
}

/// How many JIT / stock calls to time before committing a function to a tier.
const PROBE_JIT_CALLS: u32 = 8;
const PROBE_STOCK_CALLS: u32 = 4;

/// Adaptive choice of execution tier for one wasm function.
///
/// The JIT tier wins when a function's per-call loop runs long enough to
/// amortize the compiled-loop entry plus the loop-exit guard deopt; when the
/// per-call work is tiny (a loop that trips only a handful of times) the stock
/// executor — which carries no per-call JIT machinery — is faster. The trip
/// count is a runtime argument, not a static property, so probe: time the first
/// [`PROBE_JIT_CALLS`] calls on the JIT, then [`PROBE_STOCK_CALLS`] on the stock
/// executor, and commit to whichever had the lower (min) per-call time. A
/// function called only a handful of times (e.g. one giant single-call loop)
/// never finishes probing and so keeps running on the JIT it started on.
enum TierPolicy {
    Probe {
        jit_calls: u32,
        min_jit_ns: u64,
        stock_calls: u32,
        min_stock_ns: u64,
    },
    Jit,
    Stock,
}

/// The action [`TierPolicy::next_action`] selects for one call.
#[derive(Clone, Copy)]
pub(crate) enum TierAction {
    Jit,
    Stock,
    ProbeJit,
    ProbeStock,
}

impl TierPolicy {
    /// Fold a timed JIT probe into the policy (no-op once committed).
    fn record_jit(&mut self, ns: u64) {
        if let TierPolicy::Probe {
            jit_calls,
            min_jit_ns,
            ..
        } = self
        {
            *jit_calls += 1;
            *min_jit_ns = (*min_jit_ns).min(ns);
        }
    }

    /// Fold a timed stock probe into the policy (no-op once committed).
    fn record_stock(&mut self, ns: u64) {
        if let TierPolicy::Probe {
            stock_calls,
            min_stock_ns,
            ..
        } = self
        {
            *stock_calls += 1;
            *min_stock_ns = (*min_stock_ns).min(ns);
        }
    }

    /// Pick how to run this call, advancing the probe state machine. Once both
    /// probe quotas are met it commits in place to [`TierPolicy::Jit`] or
    /// [`TierPolicy::Stock`]; the probe counts themselves advance only when a
    /// timed run is recorded (`record_*`).
    fn next_action(&mut self) -> TierAction {
        // Force JIT tier for all eligible functions (bypasses probing).
        #[cfg(feature = "std")]
        if std::env::var_os("WASMI_MAJIT_FORCE_JIT").is_some() {
            *self = TierPolicy::Jit;
            return TierAction::Jit;
        }
        match self {
            TierPolicy::Jit => TierAction::Jit,
            TierPolicy::Stock => TierAction::Stock,
            TierPolicy::Probe { jit_calls, .. } if *jit_calls < PROBE_JIT_CALLS => {
                TierAction::ProbeJit
            }
            TierPolicy::Probe { stock_calls, .. } if *stock_calls < PROBE_STOCK_CALLS => {
                TierAction::ProbeStock
            }
            TierPolicy::Probe {
                min_jit_ns,
                min_stock_ns,
                ..
            } => {
                let stock_wins = *min_stock_ns < *min_jit_ns;
                *self = if stock_wins {
                    TierPolicy::Stock
                } else {
                    TierPolicy::Jit
                };
                if stock_wins {
                    TierAction::Stock
                } else {
                    TierAction::Jit
                }
            }
        }
    }
}

/// Record a timed probe of the JIT tier for `key`.
pub(crate) fn record_probe_jit(key: usize, ns: u64) {
    PROGRAMS.with(|p| {
        if let Some(Some(cached)) = p.borrow_mut().get_mut(&key) {
            cached.policy.record_jit(ns);
        }
    });
}

/// Record a timed probe of the stock executor for `key`.
pub(crate) fn record_probe_stock(key: usize, ns: u64) {
    PROGRAMS.with(|p| {
        if let Some(Some(cached)) = p.borrow_mut().get_mut(&key) {
            cached.policy.record_stock(ns);
        }
    });
}

/// Prepass `ops` once (cached by op-stream pointer), advance the adaptive tier
/// state machine, and report this call's dispatch — all under a single cache
/// lookup.
///
/// Returns `Some((num_slots, writes_result, action, slot_map))` for a
/// JIT-eligible function (its MiniProgram is cached under `ops.as_ptr()`),
/// or `None` to run on the stock executor — either the prepass rejected the
/// function or majit is disabled. `writes_result` is false for a no-result
/// function (the caller must not write a result slot). `slot_map` maps each
/// dense slot index to the original frame slot index (for seed/writeback).
/// The cache key for the matching [`run_persistent`] call is
/// `ops.as_ptr() as usize`. Folding the prepass and tier decision into one
/// [`PROGRAMS`] borrow avoids a second per-call map lookup on the committed
/// steady state.
pub(crate) fn ensure_cached(
    ops: &[u8],
    len_local_slots: u16,
    len_stack_slots: u16,
) -> Option<(usize, bool, TierAction, Vec<u16>)> {
    if !super::majit_enabled() {
        return None;
    }
    let key = ops.as_ptr() as usize;
    PROGRAMS.with(|p| {
        let mut progs = p.borrow_mut();
        let entry = progs.entry(key).or_insert_with(|| {
            let result = super::prepass::prepass(ops, len_local_slots, len_stack_slots);
            #[cfg(feature = "std")]
            if std::env::var_os("WASMI_MAJIT_STATS").is_some() {
                match &result {
                    Some(p) => {
                        // Find YIELD_STOCK word positions for diagnostics.
                        let yield_pos: alloc::vec::Vec<usize> = p.words.iter().enumerate()
                            .filter(|(_, w)| **w == super::prepass::MINI_YIELD_STOCK)
                            .map(|(i, _)| i).collect();
                        let bail_pos: alloc::vec::Vec<usize> = p.words.iter().enumerate()
                            .filter(|(_, w)| **w == super::prepass::MINI_RETURN_BAIL)
                            .map(|(i, _)| i).collect();
                        let trap_pos: alloc::vec::Vec<usize> = p.words.iter().enumerate()
                            .filter(|(_, w)| **w == super::prepass::MINI_TRAP)
                            .map(|(i, _)| i).collect();
                        eprintln!(
                            "[majit-prepass] ELIGIBLE key={:#x} ops={} → {} words, num_slots={} (locals={} stack={}, unique={}), yield_or_bail={}, globals={}, loop_header={:?}, yield={:?} bail={:?} trap={:?}",
                            key, ops.len(), p.words.len(), p.num_slots, len_local_slots, len_stack_slots, p.unique_slot_count, p.has_yield_or_bail, p.uses_globals, p.loop_header_word, yield_pos, bail_pos, trap_pos,
                        );
                    }
                    None => eprintln!(
                        "[majit-prepass] INELIGIBLE key={:#x} ops={}",
                        key, ops.len(),
                    ),
                }
            }
            result.map(|program| {
                CachedFunc {
                    program,
                    policy: TierPolicy::Probe {
                        jit_calls: 0,
                        min_jit_ns: u64::MAX,
                        stock_calls: 0,
                        min_stock_ns: u64::MAX,
                    },
                }
            })
        });
        let cached = entry.as_mut()?;
        Some((
            cached.program.num_slots,
            cached.program.writes_result,
            cached.policy.next_action(),
            cached.program.slot_map.clone(),
        ))
    })
}

/// Prepass (and cache) a callee function identified by its op stream. Returns
/// `Some((key, num_slots, uses_globals))` if the callee is JIT-eligible AND
/// has no yield/bail/trap ops (i.e., can run to completion on the MiniProgram
/// dispatch without needing a stock executor fallback), or `None` otherwise.
/// Used by the CALL_ASSEMBLER path in `call_runner_fn`.
pub(crate) fn ensure_callee_cached(
    ops: &[u8],
    len_local_slots: u16,
    len_stack_slots: u16,
) -> Option<(usize, usize, bool, Vec<u16>)> {
    let key = ops.as_ptr() as usize;
    PROGRAMS.with(|p| {
        let mut progs = p.borrow_mut();
        let entry = progs.entry(key).or_insert_with(|| {
            super::prepass::prepass(ops, len_local_slots, len_stack_slots).map(|program| {
                CachedFunc {
                    program,
                    policy: TierPolicy::Probe {
                        jit_calls: 0,
                        min_jit_ns: u64::MAX,
                        stock_calls: 0,
                        min_stock_ns: u64::MAX,
                    },
                }
            })
        });
        let cached = entry.as_ref()?;
        // Reject callees that contain yield/bail/trap ops — they cannot run
        // to completion on the CALL_ASSEMBLER path.
        if cached.program.has_yield_or_bail {
            return None;
        }
        Some((key, cached.program.num_slots, cached.program.uses_globals, cached.program.slot_map.clone()))
    })
}

/// Whether the cached, eligible function at `key` references any global. The
/// caller uses this to skip resolving the instance's global raw pointers for a
/// globals-free function.
pub(crate) fn key_uses_globals(key: usize) -> bool {
    PROGRAMS.with(|p| {
        p.borrow()
            .get(&key)
            .and_then(|c| c.as_ref())
            .is_some_and(|c| c.program.uses_globals)
    })
}

/// Run an already-cached, eligible function (`key = ops.as_ptr()`) on the
/// persistent driver, seeding the frame cells from `init_slots`.
///
/// The cached program's stable address lets the driver reuse its compiled loop
/// across calls instead of recompiling each time.
pub(crate) fn run_persistent(
    key: usize,
    init_slots: &[i64],
    mem_base: i64,
    mem_len: i64,
    globals_table: *const *mut u64,
    globals_count: usize,
) -> Option<i64> {
    set_mem_ctx(mem_base, mem_len);
    set_globals_ctx(globals_table, globals_count);
    // Extract a raw pointer to the program's words and drop the PROGRAMS
    // borrow before entering wasm_mainloop. The HashMap entry is never
    // removed, and the Vec<i64> backing the words has a stable heap address,
    // so the pointer stays valid. Releasing the borrow is required so
    // callee calls (CALL_ASSEMBLER path) can borrow PROGRAMS without a
    // RefCell re-entrancy panic.
    let (words_data, words_len): (*const i64, usize) = PROGRAMS.with(|p| {
        let progs = p.borrow();
        let program = progs
            .get(&key)
            .and_then(|c| c.as_ref())
            .map(|c| &c.program)
            .expect("run_persistent: program must be cached and eligible");
        (program.words.as_ptr(), program.words.len())
    });
    // SAFETY: the HashMap entry is never removed, and the Vec heap allocation
    // is stable (no resize after prepass). The pointer is valid for the
    // duration of the run.
    let words: &MiniCode = unsafe { core::slice::from_raw_parts(words_data, words_len) };
    // try_borrow_mut: when a yield-to-stock op (e.g. MemoryCopy) resumes the
    // stock executor and the stock executor calls another eligible function,
    // DRIVER is still borrowed by the outer run_persistent. Return None so the
    // caller falls back to the stock executor for the nested call.
    DRIVER.with(|d| {
        match d.try_borrow_mut() {
            Ok(mut slot) => {
                if slot.is_none() {
                    *slot = Some(new_driver(THRESHOLD, words, init_slots));
                }
                let driver = slot.as_mut().unwrap();
                Some(wasm_mainloop(driver, words, init_slots))
            }
            Err(_) => {
                // DRIVER is busy (nested call via CALL_RESIDUAL). Fall through
                // to CALLEE_DRIVER for one level of re-entrancy before giving
                // up to stock.
                #[cfg(feature = "std")]
                if std::env::var_os("WASMI_MAJIT_STATS").is_some() {
                    eprintln!("[majit-kernel] DRIVER_BUSY key={:#x} words_len={} → try CALLEE_DRIVER", key, words_len);
                }
                None
            }
        }
    }).or_else(|| {
        // DRIVER was busy — try CALLEE_DRIVER as a fallback, but ONLY for
        // functions that have a loop (loop_header). Non-looping functions
        // gain nothing from JIT and can cause miscompiles when run on a
        // shared driver that was created for a different function shape.
        let has_loop = PROGRAMS.with(|p| {
            p.borrow()
                .get(&key)
                .and_then(|c| c.as_ref())
                .is_some_and(|c| c.program.loop_header_word.is_some())
        });
        if !has_loop {
            return None;
        }
        CALLEE_DRIVER.with(|d| {
            match d.try_borrow_mut() {
                Ok(mut slot) => {
                    if slot.is_none() {
                        *slot = Some(new_driver(THRESHOLD, words, init_slots));
                    }
                    let driver = slot.as_mut().unwrap();
                    Some(wasm_mainloop(driver, words, init_slots))
                }
                Err(_) => None, // both drivers busy — fall back to stock
            }
        })
    })
}

/// Run a callee function on the MiniProgram dispatch (CALL_ASSEMBLER path).
///
/// Called from `call_runner_fn` when the callee has a cached MiniProgram.
/// Instead of running the callee through the stock handler-threaded executor,
/// this runs it on the flat i64 MiniProgram dispatch — the same loop majit
/// traces and compiles. Uses [`CALLEE_DRIVER`] (separate from the caller's
/// [`DRIVER`]) so there is no RefCell re-entrancy conflict.
///
/// TLS state (`MEM_CTX`, `GLOBALS_CTX`, trap flags) is saved before the callee
/// runs and restored after, since the callee may reference different globals
/// or trigger different trap states.
///
/// Returns `Some(result)` if the callee ran successfully on the MiniProgram
/// dispatch, or `None` if the callee has no cached MiniProgram (caller should
/// fall back to the stock executor).
pub(crate) fn run_callee(
    callee_ops_key: usize,
    init_slots: &[i64],
    mem_base: i64,
    mem_len: i64,
    globals_table: *const *mut u64,
    globals_count: usize,
) -> Option<i64> {
    // Save the caller's TLS state so it is restored after the callee returns.
    let saved_mem = MEM_CTX.with(|c| c.get());
    let saved_globals = GLOBALS_CTX.with(|c| c.get());
    let saved_trap = MEM_TRAP.with(|t| t.get());
    let saved_did_store = MEM_DID_STORE.with(|d| d.get());
    let saved_trap_code = TRAP_CODE.with(|c| c.get());
    let saved_bail = BAIL_TO_STOCK.with(|b| b.get());
    let saved_yield = YIELD_TO_STOCK.with(|y| y.get());

    // Set up the callee's TLS context.
    set_mem_ctx(mem_base, mem_len);
    set_globals_ctx(globals_table, globals_count);

    // Extract the program words pointer under a short borrow, then release
    // the PROGRAMS borrow before entering wasm_mainloop — same reason as
    // run_persistent (avoid re-entrancy if the callee itself calls another
    // function).
    let words_raw: Option<(*const i64, usize)> = PROGRAMS.with(|p| {
        let progs = p.borrow();
        progs
            .get(&callee_ops_key)
            .and_then(|c| c.as_ref())
            .map(|c| (c.program.words.as_ptr(), c.program.words.len()))
    });
    let result = match words_raw {
        Some((data, len)) => {
            // SAFETY: same as run_persistent — HashMap entry never removed,
            // Vec heap allocation stable.
            let words: &MiniCode = unsafe { core::slice::from_raw_parts(data, len) };
            // try_borrow_mut: if the callee itself does a CallInternal that
            // recurses back into run_callee, CALLEE_DRIVER is already
            // borrowed. In that case, return None to fall back to stock.
            CALLEE_DRIVER.with(|d| {
                match d.try_borrow_mut() {
                    Ok(mut slot) => {
                        if slot.is_none() {
                            *slot = Some(new_driver(THRESHOLD, words, init_slots));
                        }
                        let driver = slot.as_mut().unwrap();
                        Some(wasm_mainloop(driver, words, init_slots))
                    }
                    Err(_) => None, // recursive call — fall back to stock
                }
            })
        }
        None => None,
    };

    // Restore the caller's TLS state. The callee may have set MEM_TRAP or
    // other flags that would confuse the caller's post-run checks.
    MEM_CTX.with(|c| c.set(saved_mem));
    GLOBALS_CTX.with(|c| c.set(saved_globals));
    MEM_TRAP.with(|t| t.set(saved_trap));
    MEM_DID_STORE.with(|d| d.set(saved_did_store));
    TRAP_CODE.with(|c| c.set(saved_trap_code));
    BAIL_TO_STOCK.with(|b| b.set(saved_bail));
    YIELD_TO_STOCK.with(|y| y.set(saved_yield));

    result
}

/// Build a driver and install its canonical liveness once. The install is
/// program-independent (the generated `build_meta` ignores its args), so any
/// first program/state seeds it.
fn new_driver(
    threshold: u32,
    program: &MiniCode,
    init_slots: &[i64],
) -> majit_metainterp::JitDriver<WasmKernelState> {
    // No quasi-immutable state exists in the wasm kernel (plain integer reds over
    // a fixed MiniProgram), so disable the periodic loop-invalidation timer: it
    // would only force pointless re-tracing across calls.
    let mut driver = majit_metainterp::JitDriver::with_options(threshold, false);
    // Raise the guard-failure threshold out of reach so a hot loop-exit guard
    // never starts a bridge trace. This kernel's loop exits straight into
    // `MINI_RETURN_*`, and the `#[jit_interp]` dispatch-jitcode does not lower a
    // function return into an attachable FINISH bridge, so the bridge trace
    // aborts and never compiles — it would re-trace (and re-deopt through the
    // blackhole) on every call for no benefit. With bridges off, the exit takes
    // the correct blackhole recovery resume each call. (Giant single-call loops
    // never get a hot exit guard, so they are unaffected either way.)
    driver.set_trace_eagerness(u32::MAX);
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after| {
        KERNEL_COMPILES.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        if std::env::var_os("WASMI_MAJIT_STATS").is_some() {
            eprintln!(
                "[majit-kernel] COMPILE #{} green={:?} (ops {} → {})",
                KERNEL_COMPILES.load(Ordering::Relaxed),
                _green_key,
                _ops_before,
                _ops_after,
            );
        }
    });
    driver.set_on_guard_failure(|_green_key, _a, _b| {
        let n = KERNEL_GUARD_FAILS.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "std")]
        if n < 5 && std::env::var_os("WASMI_MAJIT_STATS").is_some() {
            eprintln!("[majit-kernel] GUARD_FAIL #{}", n + 1);
        }
    });
    let seed = WasmKernelState {
        slots: init_slots.to_vec(),
        accum0: 0i64,
        accum1: 0i64,
        accum2: 0i64,
    };
    {
        use majit_metainterp::JitState as _;
        seed.build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    driver
}

/// Test helper: run a [`MiniProgram`] on a fresh one-off driver with an explicit
/// `threshold` (the persistent path uses [`run_persistent`]).
///
/// [`MiniProgram`]: super::prepass::MiniProgram
#[cfg(test)]
pub(crate) fn run_kernel(
    program: &MiniCode,
    init_slots: &[i64],
    threshold: u32,
    mem_base: i64,
    mem_len: i64,
) -> i64 {
    set_mem_ctx(mem_base, mem_len);
    set_globals_ctx(core::ptr::null(), 0);
    let mut driver = new_driver(threshold, program, init_slots);
    wasm_mainloop(&mut driver, program, init_slots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::handler::majit::prepass::{MiniProgram, NUM_SCRATCH, prepass};
    use crate::{Engine, Module};

    /// Serializes tests that run the kernel and read the global
    /// [`KERNEL_COMPILES`] / [`KERNEL_GUARD_FAILS`] evidence counters: a kernel
    /// run on another test thread would otherwise increment them mid-assertion.
    /// Every kernel-running test holds this for its duration. Poison-tolerant so
    /// one failing test doesn't cascade into spurious failures.
    fn serial_kernel_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

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

    fn compile_counter() -> MiniProgram {
        let wasm = wat::parse_str(COUNTER_WAT).expect("wat parse");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm[..]).expect("module");
        let ef = module.engine_func_by_index(0).expect("engine func 0");
        engine
            .with_compiled_ops(ef, |ops, l, s| prepass(ops, l, s))
            .expect("compiled")
            .expect("eligible")
    }

    const FACTORIAL_WAT: &str = r#"
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

    fn compile_factorial() -> MiniProgram {
        let wasm = wat::parse_str(FACTORIAL_WAT).expect("wat parse");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm[..]).expect("module");
        let ef = module.engine_func_by_index(0).expect("engine func 0");
        engine
            .with_compiled_ops(ef, |ops, l, s| prepass(ops, l, s))
            .expect("compiled")
            .expect("eligible")
    }

    /// Seed `slots[0] = n`, the rest zero, sized to the program's real slot count
    /// plus the reserved scratch slots (see [`super::prepass::NUM_SCRATCH`]).
    fn seed(n: i64, num_slots: usize) -> Vec<i64> {
        let mut slots = alloc::vec![0i64; num_slots + NUM_SCRATCH];
        slots[0] = n;
        slots
    }

    /// M3a: the prepass→kernel pipeline computes `count-via-locals` correctly
    /// (it decrements `$n` to 0 and returns it) for several inputs, exercising
    /// the CloseLoop guard-exit deopt each run, without the wasmi frame splice.
    #[test]
    fn kernel_runs_count_via_locals() {
        let _serial = serial_kernel_guard();
        let mp = compile_counter();
        for n in [1i64, 2, 5, 40, 1000] {
            let slots = seed(n, mp.num_slots);
            let result = run_kernel(&mp.words, &slots, 3, 0, 0);
            assert_eq!(result, 0, "count-via-locals({n}) must return 0");
        }
    }

    /// i64 multiply lowers and traces: factorial via the kernel matches the
    /// closed-form value across inputs, and the hot loop compiles (evidence the
    /// `MINI_I64_MUL_SS_WR` arm's `*` lowered into the trace, not aborted it).
    #[test]
    fn kernel_runs_factorial_i64_mul() {
        let _serial = serial_kernel_guard();
        let mp = compile_factorial();
        fn fact(n: i64) -> i64 {
            (1..=n).product()
        }
        // n <= 20 so the reference product stays within i64 (no wrap to compare).
        for n in [0i64, 1, 2, 5, 10, 20] {
            let slots = seed(n, mp.num_slots);
            let result = run_kernel(&mp.words, &slots, 3, 0, 0);
            assert_eq!(result, fact(n), "fact({n})");
        }
        // A hot run must compile the multiply loop (i64 `*` lowers in-trace).
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let slots = seed(40, mp.num_slots);
        let _ = run_kernel(&mp.words, &slots, 3, 0, 0);
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "i64-multiply loop must compile"
        );
    }

    /// Regime-2 repro: a persistent driver shared across many short `count(8)`
    /// calls, with bridges ENABLED at a low eagerness so the hot loop-exit guard
    /// compiles a bridge after a couple of failures. Reproduces the corrupt
    /// bridge resume-pc (currently the bench's silent-corruption / OOB). Run with
    /// `MAJIT_LOG=1 --nocapture` to dump `[merge-pc]` / `[bridge]` diagnostics.
    #[test]
    fn regime2_bridge_resume_pc() {
        let _serial = serial_kernel_guard();
        let mp = compile_counter();
        // Shared driver, bridges on (eagerness 2 → bridge after 2 guard fails).
        let mut driver = majit_metainterp::JitDriver::with_options(3, false);
        driver.set_trace_eagerness(2);
        let seed0 = WasmKernelState {
            slots: seed(8, mp.num_slots),
            accum0: 0i64,
            accum1: 0i64,
            accum2: 0i64,
        };
        {
            use majit_metainterp::JitState as _;
            seed0
                .build_meta(0, &mp.words)
                .install_canonical_liveness(&mut driver);
        }
        for call in 0..8 {
            let slots = seed(8, mp.num_slots);
            let result = wasm_mainloop(&mut driver, &mp.words, &slots);
            assert_eq!(result, 0, "count(8) call #{call} must return 0");
        }
    }

    /// M3: end-to-end through the real wasmi call path. `count-via-locals(n)` is
    /// routed to the JIT tier (`init_wasm_func_call` -> prepass -> kernel) and
    /// returns the same result as the stock executor (0, for every input), with
    /// the hot run driving the kernel to compile.
    ///
    /// `n = 0` is excluded: it decrements through the entire i32 range back to 0
    /// (~2^32 iterations) under any correct implementation.
    #[test]
    fn end_to_end_count_via_locals_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, COUNTER_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "count-via-locals")
            .expect("typed func");

        for n in [1i32, 5, 40, 1000] {
            let result = func.call(&mut store, n).expect("call");
            assert_eq!(result, 0, "count-via-locals({n}) via JIT tier must be 0");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the hot loop",
        );
    }

    /// M4: a real, richer `.wasm` — `fibonacci_iter` (a pure i64 loop with three
    /// locals, a forward `block`-break branch, and a loop back-edge) — runs
    /// end-to-end on the JIT tier and matches the stock `fib(n)` for several
    /// inputs, with the hot loop compiling. `n <= 90` keeps `fib(n)` within i64.
    #[test]
    fn end_to_end_fibonacci_iter_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, FIB_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "fibonacci_iter")
            .expect("typed func");

        let expected = [
            (0i64, 0i64),
            (1, 1),
            (2, 1),
            (3, 2),
            (7, 13),
            (10, 55),
            (20, 6765),
            (40, 102_334_155),
            (90, 2_880_067_194_370_816_120),
        ];
        for (n, want) in expected {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                want,
                "fib_iter({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the fib loop",
        );
    }

    /// A real i32 loop — `sum(i*i for i in 0..n)` — runs end-to-end on the JIT
    /// tier, exercising i32 multiply, the accumulator-plus-slot i32 add, the
    /// unsigned loop-exit compare, and an unconditional back-edge, and matches the
    /// stock result with the hot loop compiling. `n <= 100` keeps the sum in i32.
    #[test]
    fn end_to_end_i32_sumsq_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SUMSQ_WAT: &str = r#"
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SUMSQ_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "f")
            .expect("typed func");

        for n in [0i32, 1, 2, 5, 10, 40, 100] {
            let want: i32 = (0..n).map(|i| i * i).sum();
            assert_eq!(func.call(&mut store, n).expect("call"), want, "sumsq({n})");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the sum-of-squares loop",
        );
    }

    /// A real i64 loop mixing `i64.xor`, `i64.and` against the accumulator, and a
    /// signed loop-exit compare runs end-to-end on the JIT tier and matches the
    /// stock result, with the hot loop compiling.
    #[test]
    fn end_to_end_mix_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const MIX_WAT: &str = r#"
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

        fn mix(n: i64) -> i64 {
            let mut h: i64 = -1;
            let mut i: i64 = 0;
            while i < n {
                h = (h ^ i) & 1023;
                i += 1;
            }
            h
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, MIX_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 2, 5, 10, 40, 200] {
            assert_eq!(func.call(&mut store, n).expect("call"), mix(n), "mix({n})");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the mixing loop",
        );
    }

    /// A popcount loop using a logical shift-right runs end-to-end on the JIT
    /// tier and matches the stock result, including for negative inputs (high bit
    /// set) — proving the mask-based logical shift terminates and counts
    /// correctly where an arithmetic shift would spin forever.
    #[test]
    fn end_to_end_popcount_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const POPCOUNT_WAT: &str = r#"
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, POPCOUNT_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for x in [
            0i64,
            1,
            2,
            3,
            255,
            1024,
            -1,
            i64::MIN,
            i64::MAX,
            0x5555_5555_5555_5555,
        ] {
            let want = i64::from((x as u64).count_ones());
            assert_eq!(
                func.call(&mut store, x).expect("call"),
                want,
                "popcount({x})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the popcount loop",
        );
    }

    /// A countdown-sum loop — a signed `<= imm` exit guard, a two-slot
    /// slot-and-reg add, and an `i - 1` decrement lowered as add-with-negated-
    /// immediate — runs end-to-end on the JIT tier and matches the stock result.
    #[test]
    fn end_to_end_countdown_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const COUNTDOWN_WAT: &str = r#"
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

        fn countdown(n: i64) -> i64 {
            let mut sum: i64 = 0;
            let mut i: i64 = n;
            while i > 0 {
                sum += i;
                i -= 1;
            }
            sum
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, COUNTDOWN_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 2, 5, 10, 100, -3, 500] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                countdown(n),
                "countdown({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the countdown loop",
        );
    }

    /// An i32 countup-sum loop — a signed i32 `< imm` exit guard and a two-slot
    /// i32 add with wraparound — runs end-to-end on the JIT tier and matches the
    /// stock result, including a negative input (exercises i32 signed compare).
    #[test]
    fn end_to_end_i32_signed_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const I32_SIGNED_WAT: &str = r#"
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

        fn sum_i32(n: i32) -> i32 {
            let mut sum: i32 = 0;
            let mut i: i32 = n;
            while i >= 1 {
                sum = sum.wrapping_add(i);
                i -= 1;
            }
            sum
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, I32_SIGNED_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "f")
            .expect("typed func");

        for n in [0i32, 1, 2, 5, 10, 100, -7, 1000] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                sum_i32(n),
                "sum_i32({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the i32 countup loop",
        );
    }

    /// A bottom-tested loop whose back-edge is a conditional `i64.gt_s` (lowered
    /// as `imm < ireg`) runs end-to-end on the JIT tier and matches the stock
    /// result, including the `n <= 0` skip case.
    #[test]
    fn end_to_end_bottom_tested_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const BOTTOM_WAT: &str = r#"
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

        fn sum_down(n: i64) -> i64 {
            let mut sum: i64 = 0;
            let mut i: i64 = n;
            while i > 0 {
                sum += i;
                i -= 1;
            }
            sum
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, BOTTOM_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 2, 5, 10, 100, -4, 500] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                sum_down(n),
                "sum_down({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the bottom-tested loop",
        );
    }

    /// An i64 bitwise-OR accumulation loop runs end-to-end on the JIT tier and
    /// matches the stock result. Its `i + 1` step exercises the pre-materialized
    /// add inside the hot loop, so this also guards that pre-materialization
    /// stays correct alongside a freshly added op.
    #[test]
    fn end_to_end_or_accum_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const OR_WAT: &str = r#"
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

        fn or_accum(n: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut i: i64 = 0;
            while i < n {
                acc |= i;
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, OR_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 2, 5, 10, 17, 64, 200] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                or_accum(n),
                "or_accum({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the OR-accumulation loop",
        );
    }

    /// An accumulation loop summing `a - i` (two-variable i64 subtraction) runs
    /// end-to-end on the JIT tier and matches the stock result.
    #[test]
    fn end_to_end_sub_accum_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SUB_WAT: &str = r#"
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

        fn sub_accum(a: i64, b: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut i: i64 = 0;
            while i < b {
                acc = acc.wrapping_add(a.wrapping_sub(i));
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SUB_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<(i64, i64), i64>(&store, "f")
            .expect("typed func");

        for (a, b) in [(0i64, 0i64), (10, 1), (10, 5), (100, 20), (-5, 8), (7, 200)] {
            assert_eq!(
                func.call(&mut store, (a, b)).expect("call"),
                sub_accum(a, b),
                "sub_accum({a}, {b})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the subtraction loop",
        );
    }

    /// A running-max loop with an `if` (a forward conditional branch and join)
    /// runs end-to-end on the JIT tier and matches the stock result, proving
    /// branchy control flow inside the loop traces and compiles correctly.
    #[test]
    fn end_to_end_runningmax_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const MAX_WAT: &str = r#"
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

        fn running_max(n: i64) -> i64 {
            let mut max: i64 = 0;
            let mut i: i64 = 0;
            while i < n {
                if i > max {
                    max = i;
                }
                i += 1;
            }
            max
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, MAX_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 2, 5, 10, 100, -3, 300] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                running_max(n),
                "running_max({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the branchy loop",
        );
    }

    /// A loop chaining i32 `xor`/`and`/`or`/`sub` (accumulator-OP-slot forms
    /// pre-materialized into a scratch slot) runs end-to-end on the JIT tier and
    /// matches the stock result, including a negative input — exercising both i32
    /// bitwise wraparound and the scratch pre-materialization of several ops.
    #[test]
    fn end_to_end_i32_bitwise_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const BITWISE_WAT: &str = r#"
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

        fn bitwise(n: i32) -> i32 {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                acc ^= i;
                acc &= n;
                acc |= i;
                acc = acc.wrapping_sub(i);
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, BITWISE_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "f")
            .expect("typed func");

        for n in [0i32, 1, 2, 5, 10, 33, 100, -4] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                bitwise(n),
                "bitwise({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the i32 bitwise loop",
        );
    }

    /// A loop counting up to an i64 equality exit (`i == n`) runs end-to-end on
    /// the JIT tier and matches the stock result.
    #[test]
    fn end_to_end_eq_guard_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const EQ_WAT: &str = r#"
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

        // Counts until `i == n`; for n < 0 it never matches within the i64 range
        // reachable here, so only non-negative inputs are meaningful to compare.
        fn count_to(n: i64) -> i64 {
            let mut count: i64 = 0;
            let mut i: i64 = 0;
            while i != n {
                count += 1;
                i += 1;
            }
            count
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, EQ_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 2, 5, 10, 100, 1000] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                count_to(n),
                "count_to({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the equality-guard loop",
        );
    }

    /// A loop accumulating `i << 2` (i64 left-shift by a constant) runs
    /// end-to-end on the JIT tier and matches the stock result.
    #[test]
    fn end_to_end_shl_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SHL_WAT: &str = r#"
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

        fn shl_accum(n: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut i: i64 = 0;
            while i < n {
                acc = acc.wrapping_add(i << 2);
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SHL_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 2, 5, 10, 100, -3, 500] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                shl_accum(n),
                "shl_accum({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the left-shift loop",
        );
    }

    /// A loop accumulating `(i << 3) >>u 1` (i32 left shift then logical right
    /// shift) runs end-to-end on the JIT tier and matches the stock result,
    /// including a negative input that exercises the i32 wraparound.
    #[test]
    fn end_to_end_i32_shift_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SHIFT_WAT: &str = r#"
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

        fn shift_accum(n: i32) -> i32 {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                acc = acc.wrapping_add((((i << 3) as u32) >> 1) as i32);
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SHIFT_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "f")
            .expect("typed func");

        for n in [0i32, 1, 2, 5, 10, 100, -4, 1000] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                shift_accum(n),
                "shift_accum({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the i32 shift loop",
        );
    }

    /// A loop accumulating `rotl(i, 7) + rotr(i, 3)` (i32 constant-amount
    /// rotates) runs end-to-end on the JIT tier and matches a Rust reference.
    /// Both rotate results feed an `i32.add`, so wasmi emits the accumulator
    /// `I32Rotl_Rsi` / `I32Rotr_Rsi` forms the prepass lowers.
    #[test]
    fn end_to_end_i32_rotate_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const ROTATE_WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.rotl (local.get $i) (i32.const 7))))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.rotr (local.get $i) (i32.const 3))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        fn rotate_accum(n: i32) -> i32 {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                acc = acc.wrapping_add(i.rotate_left(7));
                acc = acc.wrapping_add(i.rotate_right(3));
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, ROTATE_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "f")
            .expect("typed func");

        for n in [0i32, 1, 2, 5, 10, 100, -4, 1000] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                rotate_accum(n),
                "rotate_accum({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the i32 rotate loop",
        );
    }

    /// Bit-count unary ops (`clz` / `ctz` / `popcnt`) run end-to-end on the JIT
    /// tier for both widths. The i32 function counts a local (`I32Clz_Rs` slot
    /// form); the i64 function counts a computed value (`I64Clz_Rr` accumulator
    /// form), so both the `_Rs` and `_Rr` lowerings are exercised, including the
    /// `clz(0) = ctz(0) = width` edge case (loop starts / passes through zero).
    #[test]
    fn end_to_end_bitcount_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const BITCOUNT_WAT: &str = r#"
            (module
                (func (export "bits32") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.add (local.get $acc) (i32.clz (local.get $i))))
                            (local.set $acc (i32.add (local.get $acc) (i32.ctz (local.get $i))))
                            (local.set $acc (i32.add (local.get $acc) (i32.popcnt (local.get $i))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc))
                (func (export "bits64") (param $n i64) (result i64)
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

        fn bits32(n: i32) -> i32 {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                let u = i as u32;
                acc = acc.wrapping_add(u.leading_zeros() as i32);
                acc = acc.wrapping_add(u.trailing_zeros() as i32);
                acc = acc.wrapping_add(u.count_ones() as i32);
                i += 1;
            }
            acc
        }
        fn bits64(n: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut i: i64 = n;
            while i != 0 {
                let x = i.wrapping_mul(i) as u64;
                acc = acc.wrapping_add(i64::from(x.leading_zeros()));
                acc = acc.wrapping_add(i64::from(x.trailing_zeros()));
                acc = acc.wrapping_add(i64::from(x.count_ones()));
                i -= 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, BITCOUNT_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f32 = instance
            .get_typed_func::<i32, i32>(&store, "bits32")
            .expect("bits32 func");
        let f64f = instance
            .get_typed_func::<i64, i64>(&store, "bits64")
            .expect("bits64 func");

        for n in [0i32, 1, 2, 5, 10, 100, -4, 1000] {
            assert_eq!(
                f32.call(&mut store, n).expect("call"),
                bits32(n),
                "bits32({n})"
            );
        }
        for n in [0i64, 1, 2, 7, 40, 500] {
            assert_eq!(
                f64f.call(&mut store, n).expect("call"),
                bits64(n),
                "bits64({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the bit-count loops",
        );
    }

    /// GCD by subtraction (an `if`/`else` two-way branch in the loop body) runs
    /// end-to-end on the JIT tier and matches a Rust reference.
    #[test]
    fn end_to_end_gcd_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const GCD_WAT: &str = r#"
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

        fn gcd(mut a: i64, mut b: i64) -> i64 {
            while a != b {
                if a > b {
                    a -= b;
                } else {
                    b -= a;
                }
            }
            a
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, GCD_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<(i64, i64), i64>(&store, "f")
            .expect("typed func");

        for (a, b) in [
            (12i64, 8i64),
            (48, 36),
            (7, 5),
            (100, 100),
            (1, 1),
            (6, 9),
            (17, 5),
            (1000, 24),
        ] {
            assert_eq!(
                func.call(&mut store, (a, b)).expect("call"),
                gcd(a, b),
                "gcd({a},{b})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the GCD loop",
        );
    }

    /// Integer division and remainder (signed and unsigned, i32 and i64) run
    /// end-to-end on the JIT tier and match Rust references, and a division by
    /// zero reached mid-loop (after the loop has compiled) still traps — the
    /// residual latches the trap code and `run_jit` recovers via the stock
    /// executor, exactly like the trapping f64→int truncations.
    #[test]
    fn end_to_end_divrem_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const DIVREM_WAT: &str = r#"
            (module
                (func (export "divmod32") (param $n i32) (param $d i32) (result i32)
                    (local $acc i32) (local $i i32) (local $x i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $x (i32.sub (local.get $i) (i32.const 50)))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.div_s (local.get $x) (local.get $d))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.rem_s (local.get $x) (local.get $d))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.div_u (local.get $x) (local.get $d))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.rem_u (local.get $x) (local.get $d))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc))
                (func (export "divmod64") (param $n i64) (param $d i64) (result i64)
                    (local $acc i64) (local $i i64) (local $x i64)
                    (local.set $i (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $i)))
                            (local.set $x (i64.sub (local.get $i) (i64.const 50)))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.div_s (local.get $x) (local.get $d))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.rem_s (local.get $x) (local.get $d))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.div_u (local.get $x) (local.get $d))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.rem_u (local.get $x) (local.get $d))))
                            (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc))
                (func (export "trapdiv") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.div_s (i32.const 1000)
                                    (i32.sub (local.get $i) (i32.const 20)))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        fn divmod32(n: i32, d: i32) -> i32 {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                let x = i.wrapping_sub(50);
                acc = acc.wrapping_add(x / d);
                acc = acc.wrapping_add(x % d);
                acc = acc.wrapping_add((x as u32 / d as u32) as i32);
                acc = acc.wrapping_add((x as u32 % d as u32) as i32);
                i += 1;
            }
            acc
        }
        fn divmod64(n: i64, d: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut i: i64 = n;
            while i != 0 {
                let x = i.wrapping_sub(50);
                acc = acc.wrapping_add(x / d);
                acc = acc.wrapping_add(x % d);
                acc = acc.wrapping_add((x as u64 / d as u64) as i64);
                acc = acc.wrapping_add((x as u64 % d as u64) as i64);
                i -= 1;
            }
            acc
        }
        // `None` marks the div-by-zero trap (reached when `i == 20`).
        fn trapdiv(n: i32) -> Option<i32> {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                let d = i.wrapping_sub(20);
                if d == 0 {
                    return None;
                }
                acc = acc.wrapping_add(1000 / d);
                i += 1;
            }
            Some(acc)
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, DIVREM_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f32 = instance
            .get_typed_func::<(i32, i32), i32>(&store, "divmod32")
            .expect("divmod32 func");
        let f64f = instance
            .get_typed_func::<(i64, i64), i64>(&store, "divmod64")
            .expect("divmod64 func");
        let ftrap = instance
            .get_typed_func::<i32, i32>(&store, "trapdiv")
            .expect("trapdiv func");

        for (n, d) in [(1i32, 1i32), (5, 3), (40, 7), (200, 13), (100, 1), (33, 5)] {
            assert_eq!(
                f32.call(&mut store, (n, d)).expect("call"),
                divmod32(n, d),
                "divmod32({n},{d})"
            );
        }
        for (n, d) in [(1i64, 1i64), (7, 3), (40, 7), (500, 13), (60, 1)] {
            assert_eq!(
                f64f.call(&mut store, (n, d)).expect("call"),
                divmod64(n, d),
                "divmod64({n},{d})"
            );
        }
        // Non-trapping cases (loop never reaches `i == 20`).
        for n in [5i32, 20] {
            assert_eq!(
                ftrap.call(&mut store, n).expect("call"),
                trapdiv(n).expect("no trap"),
                "trapdiv({n})"
            );
        }
        // Div-by-zero cases: the loop runs (and compiles) before it traps at
        // `i == 20`, so the JIT tier must recover the trap, not return a value.
        for n in [21i32, 50, 100] {
            assert!(
                ftrap.call(&mut store, n).is_err(),
                "trapdiv({n}) must trap on the div by zero"
            );
        }
        // A non-trapping call after the traps confirms the trap state was reset.
        assert_eq!(
            ftrap.call(&mut store, 5).expect("call"),
            trapdiv(5).expect("no trap"),
            "trapdiv(5) after traps"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the division loops",
        );
    }

    /// `x + const` in value position (the sum feeds another op, not a `local.set`)
    /// runs end-to-end on the JIT tier and matches a Rust reference — exercising
    /// the `I32/I64 Add_Rsi` (slot + imm) and `_Rri` (accumulator + imm) forms,
    /// which pre-materialize the immediate and reuse the two-operand add. If any
    /// were unlowered the loop would run on stock and `KERNEL_COMPILES` stay 0.
    #[test]
    fn end_to_end_add_immediate_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const ADDVAL_WAT: &str = r#"
            (module
                (func (export "addval32") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.mul (i32.add (local.get $i) (i32.const 3))
                                         (local.get $i))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.add (i32.mul (local.get $i) (local.get $i))
                                         (i32.const 5))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc))
                (func (export "addval64") (param $n i64) (result i64)
                    (local $acc i64) (local $j i64)
                    (local.set $j (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $j)))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.mul (i64.add (local.get $j) (i64.const 7))
                                         (local.get $j))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.add (i64.mul (local.get $j) (local.get $j))
                                         (i64.const 11))))
                            (local.set $j (i64.sub (local.get $j) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        fn addval32(n: i32) -> i32 {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                acc = acc.wrapping_add(i.wrapping_add(3).wrapping_mul(i));
                acc = acc.wrapping_add(i.wrapping_mul(i).wrapping_add(5));
                i += 1;
            }
            acc
        }
        fn addval64(n: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut j: i64 = n;
            while j != 0 {
                acc = acc.wrapping_add(j.wrapping_add(7).wrapping_mul(j));
                acc = acc.wrapping_add(j.wrapping_mul(j).wrapping_add(11));
                j -= 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, ADDVAL_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f32 = instance
            .get_typed_func::<i32, i32>(&store, "addval32")
            .expect("addval32 func");
        let f64f = instance
            .get_typed_func::<i64, i64>(&store, "addval64")
            .expect("addval64 func");

        for n in [0i32, 1, 5, 40, 200, 1000, 50000] {
            assert_eq!(
                f32.call(&mut store, n).expect("call"),
                addval32(n),
                "addval32({n})"
            );
        }
        for n in [0i64, 1, 7, 40, 300, 1000] {
            assert_eq!(
                f64f.call(&mut store, n).expect("call"),
                addval64(n),
                "addval64({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the value-position add loops",
        );
    }

    /// Value-position `sub` in its remaining operand shapes (slot-slot, slot minus
    /// a computed value, `const - x`) runs end-to-end on the JIT tier and matches
    /// a Rust reference — exercising `I32Sub_R{ss,sr,ir,is}` and `I64Sub_R{rs,sr,
    /// ir,is}`, all pre-materialized onto the two-slot sub. `KERNEL_COMPILES >= 1`
    /// proves they lowered (an unlowered form would drop the loop to stock).
    #[test]
    fn end_to_end_sub_value_forms_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SUBVAL_WAT: &str = r#"
            (module
                (func (export "subval32") (param $n i32) (result i32)
                    (local $acc i32) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.sub (local.get $n) (local.get $i))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.sub (local.get $i) (i32.mul (local.get $i) (local.get $i)))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.sub (i32.const 1000) (local.get $i))))
                            (local.set $acc (i32.add (local.get $acc)
                                (i32.sub (i32.const 500) (i32.mul (local.get $i) (local.get $i)))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc))
                (func (export "subval64") (param $n i64) (result i64)
                    (local $acc i64) (local $j i64)
                    (local.set $j (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $j)))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.sub (i64.mul (local.get $j) (local.get $j)) (local.get $j))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.sub (local.get $j) (i64.mul (local.get $j) (local.get $j)))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.sub (i64.const 1000) (local.get $j))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.sub (i64.const 500) (i64.mul (local.get $j) (local.get $j)))))
                            (local.set $j (i64.sub (local.get $j) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        fn subval32(n: i32) -> i32 {
            let mut acc: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                acc = acc.wrapping_add(n.wrapping_sub(i));
                acc = acc.wrapping_add(i.wrapping_sub(i.wrapping_mul(i)));
                acc = acc.wrapping_add(1000i32.wrapping_sub(i));
                acc = acc.wrapping_add(500i32.wrapping_sub(i.wrapping_mul(i)));
                i += 1;
            }
            acc
        }
        fn subval64(n: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut j: i64 = n;
            while j != 0 {
                acc = acc.wrapping_add(j.wrapping_mul(j).wrapping_sub(j));
                acc = acc.wrapping_add(j.wrapping_sub(j.wrapping_mul(j)));
                acc = acc.wrapping_add(1000i64.wrapping_sub(j));
                acc = acc.wrapping_add(500i64.wrapping_sub(j.wrapping_mul(j)));
                j -= 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SUBVAL_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f32 = instance
            .get_typed_func::<i32, i32>(&store, "subval32")
            .expect("subval32 func");
        let f64f = instance
            .get_typed_func::<i64, i64>(&store, "subval64")
            .expect("subval64 func");

        for n in [0i32, 1, 5, 40, 200, 1000, 50000] {
            assert_eq!(
                f32.call(&mut store, n).expect("call"),
                subval32(n),
                "subval32({n})"
            );
        }
        for n in [0i64, 1, 7, 40, 300, 1000] {
            assert_eq!(
                f64f.call(&mut store, n).expect("call"),
                subval64(n),
                "subval64({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the value-position sub loops",
        );
    }

    /// An i64 `j * const` loop (`I64Mul_Rsi`, full i64 immediate) runs end-to-end
    /// on the JIT tier and matches a Rust reference.
    #[test]
    fn end_to_end_i64_mul_imm_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $acc i64) (local $j i64)
                    (local.set $j (local.get $n))
                    (block $break
                        (loop $continue
                            (br_if $break (i64.eqz (local.get $j)))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.mul (local.get $j) (i64.const 7))))
                            (local.set $acc (i64.add (local.get $acc)
                                (i64.mul (local.get $j) (i64.const 1000003))))
                            (local.set $j (i64.sub (local.get $j) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        fn f(n: i64) -> i64 {
            let mut acc: i64 = 0;
            let mut j: i64 = n;
            while j != 0 {
                acc = acc.wrapping_add(j.wrapping_mul(7));
                acc = acc.wrapping_add(j.wrapping_mul(1000003));
                j -= 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 7, 40, 300, 5000] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                f(n),
                "mul_imm({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the i64 mul-immediate loop",
        );
    }

    /// A loop adding an i32 comparison used as a 0/1 value (`count += i < 100`)
    /// runs end-to-end on the JIT tier and matches a Rust reference — exercising
    /// the `if … { 1 } else { 0 }` compare-result form in the kernel.
    #[test]
    fn end_to_end_cmp_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const CMP_WAT: &str = r#"
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

        fn count_lt_100(n: i32) -> i32 {
            let mut count: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                count += (i < 100) as i32;
                i += 1;
            }
            count
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, CMP_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "f")
            .expect("typed func");

        // Spans the threshold both ways: n below 100 (always true) and above.
        for n in [0i32, 1, 50, 100, 150, 500] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                count_lt_100(n),
                "count_lt_100({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the compare-as-value loop",
        );
    }

    /// A loop adding an i64-operand comparison used as a 0/1 value
    /// (`count += i > 10`) runs end-to-end on the JIT tier and matches a Rust
    /// reference — exercising the i64 compare-result form.
    #[test]
    fn end_to_end_i64_cmp_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const CMP_WAT: &str = r#"
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

        fn count_gt_10(n: i64) -> i32 {
            let mut count: i32 = 0;
            let mut i: i64 = 0;
            while i < n {
                count += (i > 10) as i32;
                i += 1;
            }
            count
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, CMP_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i64, i32>(&store, "f")
            .expect("typed func");

        for n in [0i64, 1, 11, 12, 50, 500] {
            assert_eq!(
                func.call(&mut store, n).expect("call"),
                count_gt_10(n),
                "count_gt_10({n})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the i64 compare-as-value loop",
        );
    }

    /// A loop summing an i64 array out of linear memory runs end-to-end on the
    /// JIT tier (the load goes through the residual helper) and matches a Rust
    /// reference, and an out-of-bounds access faithfully traps — the residual
    /// flags it and `run_jit` falls back to the stock executor.
    #[test]
    fn end_to_end_array_sum_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SUM_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        let values: alloc::vec::Vec<i64> = (0..N as i64).map(|k| (k + 1) * 3 - 7).collect();
        let mut bytes = alloc::vec![0u8; N * 8];
        for (k, v) in values.iter().enumerate() {
            bytes[k * 8..k * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SUM_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write array");
        let func = instance
            .get_typed_func::<(i32, i32), i64>(&store, "f")
            .expect("typed func");

        let ref_sum = |m: usize| -> i64 { values[..m].iter().copied().sum() };

        // Correctness across partial sums.
        for m in [0usize, 1, 2, 500, N] {
            let got = func.call(&mut store, (BASE, m as i32)).expect("call");
            assert_eq!(got, ref_sum(m), "sum of first {m}");
        }
        // Drive the loop hot so the JIT tier compiles it.
        for _ in 0..40 {
            assert_eq!(
                func.call(&mut store, (BASE, N as i32)).expect("call"),
                ref_sum(N)
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the array-sum loop",
        );

        // Out of bounds: a count that walks off the single 64 KiB page must trap,
        // exactly as the stock executor would.
        assert!(
            func.call(&mut store, (BASE, 9000)).is_err(),
            "an out-of-bounds load must trap",
        );
    }

    /// A loop chaining i64 multiply / or / xor against folded constants over each
    /// loaded element runs end-to-end on the JIT tier and matches a Rust reference.
    /// The `_Rri` forms lower onto the existing two-slot ops via pre-materialization
    /// (no dedicated kernel arm), so the scratch copies must fold away correctly.
    #[test]
    fn end_to_end_i64_const_alu_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        // Include negatives and large magnitudes so the wrapping multiply and the
        // sign bits of the or/xor are exercised.
        let values: alloc::vec::Vec<i64> = (0..N as i64)
            .map(|k| (k - 480).wrapping_mul(0x0001_0000_0001))
            .collect();
        let mut bytes = alloc::vec![0u8; N * 8];
        for (k, v) in values.iter().enumerate() {
            bytes[k * 8..k * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let reference = |m: usize| -> i64 {
            values[..m]
                .iter()
                .fold(0i64, |s, &v| s.wrapping_add((v.wrapping_mul(3) | 5) ^ 7))
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write");
        let func = instance
            .get_typed_func::<(i32, i32), i64>(&store, "f")
            .expect("typed func");

        for m in [0usize, 1, 2, 500, N] {
            assert_eq!(
                func.call(&mut store, (BASE, m as i32)).expect("call"),
                reference(m)
            );
        }
        let mut got = 0i64;
        for _ in 0..40 {
            got = func.call(&mut store, (BASE, N as i32)).expect("call");
        }
        assert_eq!(
            got,
            reference(N),
            "i64 const-alu chain must match the reference"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the const-alu loop",
        );
    }

    /// A loop chaining i64 multiply / or / xor against local (slot) operands over
    /// each loaded element runs end-to-end on the JIT tier and matches a Rust
    /// reference. The `_Rrs` forms lower onto the two-slot ops via the accumulator
    /// scratch-copy, which must fold away correctly.
    #[test]
    fn end_to_end_i64_slot_alu_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        const B: i64 = 6;
        const C: i64 = 0x00FF_0000_0F0F;
        const D: i64 = -0x1234_5678;
        let values: alloc::vec::Vec<i64> = (0..N as i64)
            .map(|k| (k - 480).wrapping_mul(0x0001_0000_0001))
            .collect();
        let mut bytes = alloc::vec![0u8; N * 8];
        for (k, v) in values.iter().enumerate() {
            bytes[k * 8..k * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let reference = |m: usize| -> i64 {
            values[..m]
                .iter()
                .fold(0i64, |s, &v| s.wrapping_add((v.wrapping_mul(B) | C) ^ D))
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write");
        let func = instance
            .get_typed_func::<(i32, i32, i64, i64, i64), i64>(&store, "f")
            .expect("typed func");

        for m in [0usize, 1, 2, 500, N] {
            assert_eq!(
                func.call(&mut store, (BASE, m as i32, B, C, D))
                    .expect("call"),
                reference(m)
            );
        }
        let mut got = 0i64;
        for _ in 0..40 {
            got = func
                .call(&mut store, (BASE, N as i32, B, C, D))
                .expect("call");
        }
        assert_eq!(
            got,
            reference(N),
            "i64 slot-alu chain must match the reference"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the slot-alu loop",
        );
    }

    /// A loop chaining i32 multiply / xor / or / and against folded constants over
    /// each loaded element runs end-to-end on the JIT tier and matches a Rust
    /// reference (wrapping i32 arithmetic, negatives included). The `_Rri` forms
    /// lower onto the existing two-slot i32 ops via pre-materialization.
    #[test]
    fn end_to_end_i32_const_alu_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        let values: alloc::vec::Vec<i32> = (0..N as i32)
            .map(|k| (k - 480).wrapping_mul(0x0010_0001))
            .collect();
        let mut bytes = alloc::vec![0u8; N * 4];
        for (k, v) in values.iter().enumerate() {
            bytes[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let reference = |m: usize| -> i32 {
            values[..m].iter().fold(0i32, |s, &v| {
                s.wrapping_add((((v.wrapping_mul(3)) ^ 5) | 6) & 12)
            })
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write");
        let func = instance
            .get_typed_func::<(i32, i32), i32>(&store, "f")
            .expect("typed func");

        for m in [0usize, 1, 2, 500, N] {
            assert_eq!(
                func.call(&mut store, (BASE, m as i32)).expect("call"),
                reference(m)
            );
        }
        let mut got = 0i32;
        for _ in 0..40 {
            got = func.call(&mut store, (BASE, N as i32)).expect("call");
        }
        assert_eq!(
            got,
            reference(N),
            "i32 const-alu chain must match the reference"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the i32 const-alu loop",
        );
    }

    /// A loop chaining i32 multiply then xor against local (slot) operands over
    /// each loaded element runs end-to-end on the JIT tier and matches a Rust
    /// reference (wrapping i32 arithmetic, negatives included). The `_Rrs` forms
    /// lower onto the two-slot i32 ops via the accumulator scratch-copy.
    #[test]
    fn end_to_end_i32_slot_alu_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        const B: i32 = 0x0001_0003;
        const C: i32 = -0x3FFF_0001;
        let values: alloc::vec::Vec<i32> = (0..N as i32)
            .map(|k| (k - 480).wrapping_mul(0x0010_0001))
            .collect();
        let mut bytes = alloc::vec![0u8; N * 4];
        for (k, v) in values.iter().enumerate() {
            bytes[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let reference = |m: usize| -> i32 {
            values[..m]
                .iter()
                .fold(0i32, |s, &v| s.wrapping_add(v.wrapping_mul(B) ^ C))
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write");
        let func = instance
            .get_typed_func::<(i32, i32, i32, i32), i32>(&store, "f")
            .expect("typed func");

        for m in [0usize, 1, 2, 500, N] {
            assert_eq!(
                func.call(&mut store, (BASE, m as i32, B, C)).expect("call"),
                reference(m)
            );
        }
        let mut got = 0i32;
        for _ in 0..40 {
            got = func.call(&mut store, (BASE, N as i32, B, C)).expect("call");
        }
        assert_eq!(
            got,
            reference(N),
            "i32 slot-alu chain must match the reference"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the i32 slot-alu loop",
        );
    }

    /// A loop summing an i32 array out of linear memory (sign-extending each
    /// element) runs end-to-end on the JIT tier and matches a Rust reference,
    /// including negative elements that exercise the sign-extend.
    #[test]
    fn end_to_end_i32_array_sum_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SUM_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        // Alternating signs so the sign-extend matters.
        let values: alloc::vec::Vec<i32> = (0..N as i32)
            .map(|k| if k % 2 == 0 { k * 5 } else { -(k * 5 + 1) })
            .collect();
        let mut bytes = alloc::vec![0u8; N * 4];
        for (k, v) in values.iter().enumerate() {
            bytes[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SUM_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write array");
        let func = instance
            .get_typed_func::<(i32, i32), i64>(&store, "f")
            .expect("typed func");

        let ref_sum = |m: usize| -> i64 { values[..m].iter().map(|&v| i64::from(v)).sum() };

        for m in [0usize, 1, 2, 500, N] {
            let got = func.call(&mut store, (BASE, m as i32)).expect("call");
            assert_eq!(got, ref_sum(m), "i32 sum of first {m}");
        }
        for _ in 0..40 {
            assert_eq!(
                func.call(&mut store, (BASE, N as i32)).expect("call"),
                ref_sum(N)
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the i32 array-sum loop",
        );
        assert!(
            func.call(&mut store, (BASE, 20000)).is_err(),
            "an out-of-bounds i32 load must trap",
        );
    }

    /// A loop summing the unsigned bytes of a buffer from linear memory runs
    /// end-to-end on the JIT tier and matches a Rust reference, including bytes
    /// with the high bit set (which must zero-extend, not sign-extend).
    #[test]
    fn end_to_end_byte_sum_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SUM_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        // Values spanning 0..=255 so the high bit is set for many bytes.
        let bytes: alloc::vec::Vec<u8> = (0..N).map(|k| (k * 7 + 3) as u8).collect();

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SUM_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write buffer");
        let func = instance
            .get_typed_func::<(i32, i32), i64>(&store, "f")
            .expect("typed func");

        let ref_sum = |m: usize| -> i64 { bytes[..m].iter().map(|&b| i64::from(b)).sum() };

        for m in [0usize, 1, 2, 500, N] {
            let got = func.call(&mut store, (BASE, m as i32)).expect("call");
            assert_eq!(got, ref_sum(m), "byte sum of first {m}");
        }
        for _ in 0..40 {
            assert_eq!(
                func.call(&mut store, (BASE, N as i32)).expect("call"),
                ref_sum(N)
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the byte-sum loop",
        );
        assert!(
            func.call(&mut store, (BASE, 80000)).is_err(),
            "an out-of-bounds byte load must trap",
        );
    }

    /// A loop summing the signed bytes of a buffer from linear memory runs
    /// end-to-end on the JIT tier and matches a Rust reference, including
    /// negative bytes that must sign-extend.
    #[test]
    fn end_to_end_signed_byte_sum_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const SUM_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
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

        const N: usize = 1000;
        const BASE: i32 = 64;
        // Many values with the high bit set so the sign-extend matters.
        let signed: alloc::vec::Vec<i8> = (0..N).map(|k| (k as i32 * 13 - 400) as i8).collect();
        let bytes: alloc::vec::Vec<u8> = signed.iter().map(|&v| v as u8).collect();

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SUM_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &bytes)
            .expect("write buffer");
        let func = instance
            .get_typed_func::<(i32, i32), i64>(&store, "f")
            .expect("typed func");

        let ref_sum = |m: usize| -> i64 { signed[..m].iter().map(|&b| i64::from(b)).sum() };

        for m in [0usize, 1, 2, 500, N] {
            let got = func.call(&mut store, (BASE, m as i32)).expect("call");
            assert_eq!(got, ref_sum(m), "signed byte sum of first {m}");
        }
        for _ in 0..40 {
            assert_eq!(
                func.call(&mut store, (BASE, N as i32)).expect("call"),
                ref_sum(N)
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have run and compiled the signed-byte-sum loop",
        );
        assert!(
            func.call(&mut store, (BASE, 80000)).is_err(),
            "an out-of-bounds signed byte load must trap",
        );
    }

    /// Loops summing 16-bit elements (unsigned and signed) from linear memory
    /// run end-to-end on the JIT tier and match a Rust reference, with OOB
    /// trapping. The signed variant includes negative halfwords.
    #[test]
    fn end_to_end_u16_loads_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // Builds an `f(ptr, n) = sum of n 16-bit elements` instance. The instance
        // is returned (not dropped) so its op-stream stays alive: the JIT caches
        // are keyed by the op-stream pointer, and a freed module's address could
        // be reused, aliasing a stale cached program.
        fn build(
            engine: &Engine,
            wat: &str,
            raw: &[u8],
        ) -> (Store<()>, crate::TypedFunc<(i32, i32), i64>) {
            let mut store = Store::new(engine, ());
            let module = Module::new(engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, raw)
                .expect("write buffer");
            let func = instance
                .get_typed_func::<(i32, i32), i64>(&store, "f")
                .expect("typed func");
            (store, func)
        }

        let check = |store: &mut Store<()>,
                     func: &crate::TypedFunc<(i32, i32), i64>,
                     reference: &dyn Fn(usize) -> i64| {
            KERNEL_COMPILES.store(0, Ordering::Relaxed);
            for m in [0usize, 1, 2, 500, N] {
                assert_eq!(
                    func.call(&mut *store, (BASE, m as i32)).expect("call"),
                    reference(m),
                    "16-bit sum of first {m}"
                );
            }
            for _ in 0..40 {
                assert_eq!(
                    func.call(&mut *store, (BASE, N as i32)).expect("call"),
                    reference(N)
                );
            }
            assert!(
                KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
                "the JIT tier must have compiled the 16-bit-sum loop",
            );
            assert!(
                func.call(&mut *store, (BASE, 40000)).is_err(),
                "an out-of-bounds 16-bit load must trap",
            );
        };

        const U16_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $sum i64) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.extend_i32_u
                                        (i32.load16_u
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 2)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;
        const I16_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result i64)
                    (local $sum i64) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.extend_i32_s
                                        (i32.load16_s
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 2)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;

        let u16_vals: alloc::vec::Vec<u16> = (0..N).map(|k| (k * 61 + 7) as u16).collect();
        let i16_vals: alloc::vec::Vec<i16> =
            (0..N).map(|k| (k as i32 * 131 - 30000) as i16).collect();
        let mut u16_raw = alloc::vec![0u8; N * 2];
        let mut i16_raw = alloc::vec![0u8; N * 2];
        for k in 0..N {
            u16_raw[k * 2..k * 2 + 2].copy_from_slice(&u16_vals[k].to_le_bytes());
            i16_raw[k * 2..k * 2 + 2].copy_from_slice(&i16_vals[k].to_le_bytes());
        }

        // Build both instances up front so neither op-stream is freed (and
        // possibly re-aliased) while the other runs.
        let engine = Engine::default();
        let (mut u16_store, u16_func) = build(&engine, U16_WAT, &u16_raw);
        let (mut i16_store, i16_func) = build(&engine, I16_WAT, &i16_raw);

        check(&mut u16_store, &u16_func, &|m| {
            u16_vals[..m].iter().map(|&v| i64::from(v)).sum()
        });
        check(&mut i16_store, &i16_func, &|m| {
            i16_vals[..m].iter().map(|&v| i64::from(v)).sum()
        });
    }

    /// Loops writing to linear memory run end-to-end on the JIT tier: a buffer
    /// fill (`i32.store`) leaves the correct bytes, an interleaved
    /// store-then-load of the same cell reads back its own write (residual
    /// load/store ordering is preserved), and an out-of-bounds store traps.
    #[test]
    fn end_to_end_i32_store_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                ;; buf[i] = i*i for i in 0..n
                (func (export "fill") (param $ptr i32) (param $n i32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 4)))
                                (i32.mul (local.get $i) (local.get $i)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue))))
                ;; buf[i] = i (the value is a local, so a slot-value store)
                (func (export "fill_local") (param $ptr i32) (param $n i32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 4)))
                                (local.get $i))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue))))
                ;; store i*i then immediately load buf[i] back, accumulating the sum
                ;; (the stored value is computed, so it lands in the accumulator)
                (func (export "raw") (param $ptr i32) (param $n i32) (result i64)
                    (local $sum i64) (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 4)))
                                (i32.mul (local.get $i) (local.get $i)))
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        let fill = instance
            .get_typed_func::<(i32, i32), ()>(&store, "fill")
            .expect("fill func");

        // Drive the fill loop hot; it is idempotent (same i*i each call).
        for _ in 0..40 {
            fill.call(&mut store, (BASE, N as i32)).expect("call fill");
        }
        let mut buf = alloc::vec![0u8; N * 4];
        memory
            .read(&store, BASE as usize, &mut buf)
            .expect("read back");
        for k in 0..N {
            let got = u32::from_le_bytes(buf[k * 4..k * 4 + 4].try_into().unwrap());
            assert_eq!(got, (k * k) as u32, "buf[{k}] after fill");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the store loop",
        );

        // Slot-value store: buf[i] = i (the value comes straight from a local).
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let fill_local = instance
            .get_typed_func::<(i32, i32), ()>(&store, "fill_local")
            .expect("fill_local func");
        for _ in 0..40 {
            fill_local
                .call(&mut store, (BASE, N as i32))
                .expect("call fill_local");
        }
        memory
            .read(&store, BASE as usize, &mut buf)
            .expect("read back");
        for k in 0..N {
            let got = u32::from_le_bytes(buf[k * 4..k * 4 + 4].try_into().unwrap());
            assert_eq!(got, k as u32, "buf[{k}] after fill_local");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the slot-value store loop",
        );

        // store-then-load the same cell each iteration: the load must observe the
        // store, so the sum is 0 + 1 + ... + (N-1).
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let raw = instance
            .get_typed_func::<(i32, i32), i64>(&store, "raw")
            .expect("raw func");
        let expect_sum: i64 = (0..N as i64).map(|i| i * i).sum();
        for _ in 0..40 {
            assert_eq!(
                raw.call(&mut store, (BASE, N as i32)).expect("call raw"),
                expect_sum,
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the store-then-load loop",
        );

        // An out-of-bounds store traps. This partially fills before trapping, so
        // it runs last.
        assert!(
            fill.call(&mut store, (BASE, 20000)).is_err(),
            "an out-of-bounds store must trap",
        );
    }

    /// A loop writing a 64-bit local to linear memory runs end-to-end on the JIT
    /// tier: `buf[i] = i + 1` (8-byte cells) leaves the correct bytes, and an
    /// out-of-bounds store traps.
    #[test]
    fn end_to_end_i64_store_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // acc starts at 0 and is bumped before each store, so buf[i] = i + 1.
        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "fill64") (param $ptr i32) (param $n i32)
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        let fill = instance
            .get_typed_func::<(i32, i32), ()>(&store, "fill64")
            .expect("fill64 func");

        // Idempotent (same i+1 each call); drive it hot.
        for _ in 0..40 {
            fill.call(&mut store, (BASE, N as i32))
                .expect("call fill64");
        }
        let mut buf = alloc::vec![0u8; N * 8];
        memory
            .read(&store, BASE as usize, &mut buf)
            .expect("read back");
        for k in 0..N {
            let got = i64::from_le_bytes(buf[k * 8..k * 8 + 8].try_into().unwrap());
            assert_eq!(got, k as i64 + 1, "buf[{k}] after fill64");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the 64-bit store loop",
        );

        assert!(
            fill.call(&mut store, (BASE, 20000)).is_err(),
            "an out-of-bounds 64-bit store must trap",
        );
    }

    /// Loops writing narrow (8/16-bit, wrapping) values to linear memory run
    /// end-to-end on the JIT tier: `buf[i] = i` truncated to a byte / halfword
    /// leaves the correct bytes, and an out-of-bounds store traps.
    #[test]
    fn end_to_end_narrow_store_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "fill8") (param $ptr i32) (param $n i32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store8
                                (i32.add (local.get $ptr) (local.get $i))
                                (local.get $i))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue))))
                (func (export "fill16") (param $ptr i32) (param $n i32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store16
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 2)))
                                (local.get $i))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");

        // 8-bit: buf[i] = i as u8.
        let fill8 = instance
            .get_typed_func::<(i32, i32), ()>(&store, "fill8")
            .expect("fill8 func");
        for _ in 0..40 {
            fill8
                .call(&mut store, (BASE, N as i32))
                .expect("call fill8");
        }
        let mut buf = alloc::vec![0u8; N];
        memory.read(&store, BASE as usize, &mut buf).expect("read8");
        for k in 0..N {
            assert_eq!(buf[k], k as u8, "buf8[{k}]");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the 8-bit store loop",
        );

        // 16-bit: buf[i] = i as u16.
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let fill16 = instance
            .get_typed_func::<(i32, i32), ()>(&store, "fill16")
            .expect("fill16 func");
        for _ in 0..40 {
            fill16
                .call(&mut store, (BASE, N as i32))
                .expect("call fill16");
        }
        let mut buf16 = alloc::vec![0u8; N * 2];
        memory
            .read(&store, BASE as usize, &mut buf16)
            .expect("read16");
        for k in 0..N {
            let got = u16::from_le_bytes(buf16[k * 2..k * 2 + 2].try_into().unwrap());
            assert_eq!(got, k as u16, "buf16[{k}]");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the 16-bit store loop",
        );

        assert!(
            fill16.call(&mut store, (BASE, 50000)).is_err(),
            "an out-of-bounds narrow store must trap",
        );
    }

    /// Storing a COMPUTED value (value in the accumulator, address in a slot)
    /// runs end-to-end on the JIT tier: a 64-bit `buf[i] = (i+1)+(i+1)` and an
    /// 8-bit `buf[i] = (i*i) as u8` leave the correct bytes.
    #[test]
    fn end_to_end_computed_store_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                ;; buf[i] = acc + acc, where acc = i + 1  => 2*(i+1)
                (func (export "f64") (param $ptr i32) (param $n i32)
                    (local $i i32) (local $acc i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc (i64.add (local.get $acc) (i64.const 1)))
                            (i64.store
                                (i32.add (local.get $ptr)
                                         (i32.mul (local.get $i) (i32.const 8)))
                                (i64.add (local.get $acc) (local.get $acc)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue))))
                ;; buf[i] = (i*i) as u8
                (func (export "f8") (param $ptr i32) (param $n i32)
                    (local $i i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i32.store8
                                (i32.add (local.get $ptr) (local.get $i))
                                (i32.mul (local.get $i) (local.get $i)))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))))
        "#;

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");

        let f64f = instance
            .get_typed_func::<(i32, i32), ()>(&store, "f64")
            .expect("f64 func");
        for _ in 0..40 {
            f64f.call(&mut store, (BASE, N as i32)).expect("call f64");
        }
        let mut buf = alloc::vec![0u8; N * 8];
        memory
            .read(&store, BASE as usize, &mut buf)
            .expect("read64");
        for k in 0..N {
            let got = i64::from_le_bytes(buf[k * 8..k * 8 + 8].try_into().unwrap());
            assert_eq!(got, 2 * (k as i64 + 1), "computed buf64[{k}]");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the computed 64-bit store",
        );

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let f8 = instance
            .get_typed_func::<(i32, i32), ()>(&store, "f8")
            .expect("f8 func");
        for _ in 0..40 {
            f8.call(&mut store, (BASE, N as i32)).expect("call f8");
        }
        let mut buf8 = alloc::vec![0u8; N];
        memory
            .read(&store, BASE as usize, &mut buf8)
            .expect("read8");
        for k in 0..N {
            assert_eq!(buf8[k], (k * k) as u8, "computed buf8[{k}]");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the computed 8-bit store",
        );
    }

    /// A loop counting array elements below a threshold — a signed i32 compare
    /// result accumulated as a 0/1 value — runs end-to-end on the JIT tier and
    /// matches a Rust reference across thresholds.
    #[test]
    fn end_to_end_compare_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "count_below")
                        (param $ptr i32) (param $n i32) (param $t i32) (result i32)
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
                    (local.get $count))
                (func (export "count_above")
                        (param $ptr i32) (param $n i32) (param $t i32) (result i32)
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

        // Mixed positive/negative values so the signed comparison matters.
        let vals: alloc::vec::Vec<i32> = (0..N).map(|k| (k as i32 * 37 - 15000)).collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, i32), i32>(&store, "count_below")
            .expect("count_below func");

        let below = |t: i32| vals.iter().filter(|&&v| v < t).count() as i32;
        for t in [-20000, 0, 5000, 50000] {
            for _ in 0..15 {
                assert_eq!(
                    f.call(&mut store, (BASE, N as i32, t)).expect("call"),
                    below(t),
                    "count_below threshold {t}",
                );
            }
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the compare-value loop",
        );

        // count_above uses the swapped compare-value form (slot vs accumulator).
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let g = instance
            .get_typed_func::<(i32, i32, i32), i32>(&store, "count_above")
            .expect("count_above func");
        let above = |t: i32| vals.iter().filter(|&&v| v > t).count() as i32;
        for t in [-20000, 0, 5000, 50000] {
            for _ in 0..15 {
                assert_eq!(
                    g.call(&mut store, (BASE, N as i32, t)).expect("call"),
                    above(t),
                    "count_above threshold {t}",
                );
            }
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the swapped compare-value loop",
        );
    }

    /// A running-max over an array via branchless `select` runs end-to-end on the
    /// JIT tier and matches a Rust reference, including the all-negative case
    /// (where the seed 0 must NOT win).
    #[test]
    fn end_to_end_select_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                ;; max(0, max over buf[0..n])  — seed is local $max = 0
                (func (export "running_max") (param $ptr i32) (param $n i32) (result i32)
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

        // Build BOTH modules up front and keep them alive: the JIT cache is keyed
        // by op-stream pointer, so a dropped module's address can be reused and
        // alias a stale cached trace (see the documented stale-cache gotcha).
        let build = |vals: &[i32]| {
            let mut raw = alloc::vec![0u8; vals.len() * 4];
            for (k, &v) in vals.iter().enumerate() {
                raw[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, WAT).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32), i32>(&store, "running_max")
                .expect("running_max func");
            (store, f)
        };

        // Mixed sign: the max is a large positive late in the array.
        let mixed: alloc::vec::Vec<i32> = (0..N).map(|k| (k as i32 * 31 - 9000)).collect();
        // All negative: seed 0 wins (select must keep the false arm every iter).
        let neg: alloc::vec::Vec<i32> = (0..N).map(|k| -(k as i32) - 1).collect();

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut store_m, f_m) = build(&mixed);
        let (mut store_n, f_n) = build(&neg);

        let mut got_m = 0;
        let mut got_n = 0;
        for _ in 0..40 {
            got_m = f_m
                .call(&mut store_m, (BASE, N as i32))
                .expect("call mixed");
            got_n = f_n.call(&mut store_n, (BASE, N as i32)).expect("call neg");
        }
        assert_eq!(
            got_m,
            mixed.iter().copied().fold(0, i32::max),
            "running_max mixed"
        );
        assert_eq!(got_n, 0, "running_max all-negative");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the select loop",
        );
    }

    /// A running f64 max using branchless `select` and `f64.gt` runs end-to-end on
    /// the JIT tier. wasmi emits the swapped compare-value form for the predicate.
    #[test]
    fn end_to_end_f64_swapped_cmp_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "running_max") (param $ptr i32) (param $n i32) (param $scratch i32) (result f64)
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

        let vals: alloc::vec::Vec<f64> = (0..N)
            .map(|k| match k % 7 {
                0 => -(k as f64) * 0.25 - 1.0,
                1 => (k as f64) * 0.5 - 125.0,
                _ => (k as f64).sin() * 300.0,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }
        let reference = vals
            .iter()
            .fold(0.0f64, |max, &v| if v > max { v } else { max });

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, i32), f64>(&store, "running_max")
            .expect("running_max func");

        let mut got = 0.0f64;
        for _ in 0..40 {
            got = f.call(&mut store, (BASE, N as i32, 0)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            reference.to_bits(),
            "f64 running max must be bit-exact",
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 swapped compare loop",
        );
    }

    /// Selects with i32 constant arms run end-to-end on the JIT tier and match
    /// Rust references for false-const, true-const, and both-const forms.
    #[test]
    fn end_to_end_select_const_arms_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const FALSE_CONST_WAT: &str = r#"
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
        const TRUE_CONST_WAT: &str = r#"
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
        const BOTH_CONST_WAT: &str = r#"
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

        fn sum_select(
            n: i32,
            true_val: impl Fn(i32) -> i32,
            false_val: impl Fn(i32) -> i32,
        ) -> i32 {
            (0..n)
                .map(|i| if i < 5 { true_val(i) } else { false_val(i) })
                .sum()
        }

        let engine = Engine::default();
        let build = |wat: &str| {
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let f = instance
                .get_typed_func::<i32, i32>(&store, "f")
                .expect("f func");
            (store, f)
        };

        let (mut false_store, false_f) = build(FALSE_CONST_WAT);
        let (mut true_store, true_f) = build(TRUE_CONST_WAT);
        let (mut both_store, both_f) = build(BOTH_CONST_WAT);

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let before = KERNEL_COMPILES.load(Ordering::Relaxed);
        for k in 0..40 {
            let n = 1000 + k;
            assert_eq!(
                false_f.call(&mut false_store, n).expect("call false"),
                sum_select(n, |i| i, |_| 7)
            );
            assert_eq!(
                true_f.call(&mut true_store, n).expect("call true"),
                sum_select(n, |_| 7, |i| i)
            );
            assert_eq!(
                both_f.call(&mut both_store, n).expect("call both"),
                sum_select(n, |_| 11, |_| 7)
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) - before >= 3,
            "the JIT tier must have compiled the const-arm select loops",
        );
    }

    /// An i64 select with a constant false arm and `i64.lt_s(slot, const)` as the
    /// condition runs end-to-end on the JIT tier.
    #[test]
    fn end_to_end_i64_select_const_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let before = KERNEL_COMPILES.load(Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("f func");

        for k in 0..40 {
            let n = 1000 + i64::from(k);
            let expected: i64 = (0..n).map(|i| if i < 5 { i } else { 7 }).sum();
            assert_eq!(f.call(&mut store, n).expect("call"), expected);
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) - before >= 1,
            "the JIT tier must have compiled the i64 const-arm select loop",
        );
    }

    /// i32 compare-values against constants are pre-materialized into slot/slot
    /// compare ops and run end-to-end on the JIT tier; `MAJIT_LOG=1` should show
    /// a `trace action ... -> CloseLoop` for this loop.
    #[test]
    fn end_to_end_i32_cmp_const_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i32) (result i32)
                    (local $i i32) (local $acc i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.eq (local.get $i) (i32.const 500))))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.ne (local.get $i) (i32.const 3))))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.le_s (local.get $i) (i32.const -3))))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.lt_u (local.get $i) (i32.const -3))))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.gt_u (local.get $i) (i32.const 7))))
                            (local.set $acc
                                (i32.add (local.get $acc)
                                    (i32.ge_s (local.get $i) (i32.const 100))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        fn reference(n: i32) -> i32 {
            let mut acc = 0;
            let mut i = 0;
            while i < n {
                acc += (i == 500) as i32;
                acc += (i != 3) as i32;
                acc += (i <= -3) as i32;
                acc += ((i as u32) < ((-3i32) as u32)) as i32;
                acc += ((i as u32) > 7u32) as i32;
                acc += (i >= 100) as i32;
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let before = KERNEL_COMPILES.load(Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<i32, i32>(&store, "f")
            .expect("f func");

        for k in 0..40 {
            let n = 1000 + k;
            assert_eq!(f.call(&mut store, n).expect("call"), reference(n));
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) - before >= 1,
            "the JIT tier must have compiled the i32 const compare-value loop",
        );
    }

    /// i64 compare-values against constants are pre-materialized into slot/slot
    /// compare ops and run end-to-end on the JIT tier; `MAJIT_LOG=1` should show
    /// a `trace action ... -> CloseLoop` for this loop.
    #[test]
    fn end_to_end_i64_cmp_const_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (func (export "f") (param $n i64) (result i64)
                    (local $i i64) (local $acc i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.extend_i32_u
                                        (i64.eq (local.get $i) (i64.const 500)))))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.extend_i32_u
                                        (i64.ne (local.get $i) (i64.const 3)))))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.extend_i32_u
                                        (i64.le_s (local.get $i) (i64.const -3)))))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.extend_i32_u
                                        (i64.lt_u (local.get $i) (i64.const -3)))))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.extend_i32_u
                                        (i64.gt_u (local.get $i) (i64.const 7)))))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.extend_i32_u
                                        (i64.ge_s (local.get $i) (i64.const 100)))))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        fn reference(n: i64) -> i64 {
            let mut acc = 0;
            let mut i = 0;
            while i < n {
                acc += (i == 500) as i64;
                acc += (i != 3) as i64;
                acc += (i <= -3) as i64;
                acc += ((i as u64) < ((-3i64) as u64)) as i64;
                acc += ((i as u64) > 7u64) as i64;
                acc += (i >= 100) as i64;
                i += 1;
            }
            acc
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let before = KERNEL_COMPILES.load(Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<i64, i64>(&store, "f")
            .expect("f func");

        for k in 0..40 {
            let n = 1000 + i64::from(k);
            assert_eq!(f.call(&mut store, n).expect("call"), reference(n));
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) - before >= 1,
            "the JIT tier must have compiled the i64 const compare-value loop",
        );
    }

    /// Counting occurrences (`mem[i] == t`) and non-matches (`mem[i] != t`) as
    /// 0/1 values runs end-to-end on the JIT tier and matches a Rust reference.
    /// Both modules are kept alive so the pointer-keyed JIT cache cannot alias a
    /// stale trace (per the documented stale-cache gotcha).
    #[test]
    fn end_to_end_equality_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // Each lane repeats a small cycle so a chosen target hits many times.
        let vals: alloc::vec::Vec<i32> = (0..N).map(|k| (k as i32 % 7) - 3).collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32, i32), i32>(&store, "count")
                .expect("count func");
            (store, f)
        };

        const EQ_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "count")
                        (param $ptr i32) (param $n i32) (param $t i32) (result i32)
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
        let ne_wat = EQ_WAT.replace("(i32.eq", "(i32.ne");

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut store_eq, f_eq) = build(EQ_WAT);
        let (mut store_ne, f_ne) = build(&ne_wat);

        let mut got_eq = 0;
        let mut got_ne = 0;
        for _ in 0..40 {
            for t in [-3, 0, 3] {
                got_eq = f_eq
                    .call(&mut store_eq, (BASE, N as i32, t))
                    .expect("eq call");
                got_ne = f_ne
                    .call(&mut store_ne, (BASE, N as i32, t))
                    .expect("ne call");
                assert_eq!(
                    got_eq,
                    vals.iter().filter(|&&v| v == t).count() as i32,
                    "count_eq t={t}",
                );
                assert_eq!(
                    got_ne,
                    vals.iter().filter(|&&v| v != t).count() as i32,
                    "count_ne t={t}",
                );
            }
        }
        let _ = (got_eq, got_ne);
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the equality-value loops",
        );
    }

    /// Counting i64 array elements that equal / differ-from / are-less-than a
    /// 64-bit target (each compare used as a 0/1 value) runs end-to-end on the JIT
    /// tier and matches a Rust reference, with values spanning the full i64 range
    /// so the full-width (non-sign-extended) comparison matters. All three modules
    /// stay alive to avoid the pointer-keyed stale-cache alias.
    #[test]
    fn end_to_end_i64_compare_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // Wide i64 values, some equal to the target, straddling 32-bit boundaries.
        const TARGET: i64 = 0x1234_5678_9abc_def0u64 as i64;
        let vals: alloc::vec::Vec<i64> = (0..N)
            .map(|k| match k % 4 {
                0 => TARGET,
                1 => -(k as i64) * 0x1_0000_0001,
                2 => (k as i64) << 40,
                _ => k as i64,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32, i64), i32>(&store, "count")
                .expect("count func");
            (store, f)
        };

        const EQ_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "count")
                        (param $ptr i32) (param $n i32) (param $t i64) (result i32)
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_eq, f_eq) = build(EQ_WAT);
        let (mut s_ne, f_ne) = build(&EQ_WAT.replace("(i64.eq", "(i64.ne"));
        let (mut s_lt, f_lt) = build(&EQ_WAT.replace("(i64.eq", "(i64.lt_s"));

        for _ in 0..40 {
            let eq = f_eq.call(&mut s_eq, (BASE, N as i32, TARGET)).expect("eq");
            let ne = f_ne.call(&mut s_ne, (BASE, N as i32, TARGET)).expect("ne");
            let lt = f_lt.call(&mut s_lt, (BASE, N as i32, TARGET)).expect("lt");
            assert_eq!(
                eq,
                vals.iter().filter(|&&v| v == TARGET).count() as i32,
                "i64 eq"
            );
            assert_eq!(
                ne,
                vals.iter().filter(|&&v| v != TARGET).count() as i32,
                "i64 ne"
            );
            assert_eq!(
                lt,
                vals.iter().filter(|&&v| v < TARGET).count() as i32,
                "i64 lt"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the i64 compare-value loops",
        );
    }

    /// Counting array elements with signed `<=` and unsigned `<` / `<=` (each
    /// compare used as a 0/1 value) runs end-to-end on the JIT tier and matches a
    /// Rust reference. Values span negatives and high-bit-set words so the
    /// signed-vs-unsigned distinction actually bites. All modules stay alive to
    /// avoid the pointer-keyed stale-cache alias.
    #[test]
    fn end_to_end_le_and_unsigned_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // Mix of negatives and large unsigned (high-bit-set) values.
        let vals: alloc::vec::Vec<i32> = (0..N)
            .map(|k| match k % 3 {
                0 => k as i32 - 400,
                1 => (0x8000_0000u32 | (k as u32)) as i32, // huge as unsigned, negative as signed
                _ => (k as i32) * 7,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32, i32), i32>(&store, "count")
                .expect("count func");
            (store, f)
        };

        const LES_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "count")
                        (param $ptr i32) (param $n i32) (param $t i32) (result i32)
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_les, f_les) = build(LES_WAT);
        let (mut s_ltu, f_ltu) = build(&LES_WAT.replace("(i32.le_s", "(i32.lt_u"));
        let (mut s_leu, f_leu) = build(&LES_WAT.replace("(i32.le_s", "(i32.le_u"));

        for &t in &[-100i32, 0, 0x4000_0000, 0x8000_0001u32 as i32] {
            for _ in 0..15 {
                let les = f_les.call(&mut s_les, (BASE, N as i32, t)).expect("les");
                let ltu = f_ltu.call(&mut s_ltu, (BASE, N as i32, t)).expect("ltu");
                let leu = f_leu.call(&mut s_leu, (BASE, N as i32, t)).expect("leu");
                assert_eq!(
                    les,
                    vals.iter().filter(|&&v| v <= t).count() as i32,
                    "le_s t={t}",
                );
                assert_eq!(
                    ltu,
                    vals.iter().filter(|&&v| (v as u32) < (t as u32)).count() as i32,
                    "lt_u t={t}",
                );
                assert_eq!(
                    leu,
                    vals.iter().filter(|&&v| (v as u32) <= (t as u32)).count() as i32,
                    "le_u t={t}",
                );
            }
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the le/unsigned compare-value loops",
        );
    }

    /// An f64 array-sum runs end-to-end on the JIT tier (the floats live in the
    /// i64 slots as bit patterns; the add is a residual that bit-casts) and yields
    /// the exact same f64 the stock executor would — bit-identical because the
    /// summation order is preserved.
    #[test]
    fn end_to_end_f64_sum_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result f64)
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

        // Values with fractional parts and mixed signs so rounding actually occurs.
        let vals: alloc::vec::Vec<f64> = (0..N)
            .map(|k| (k as f64) * 0.3 - 137.25 + (k as f64).sin())
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }
        // Sequential fold — the exact order the wasm loop accumulates in.
        let expected = vals.iter().fold(0.0f64, |a, &b| a + b);

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32), f64>(&store, "sum")
            .expect("sum func");

        let mut got = 0.0;
        for _ in 0..40 {
            got = f.call(&mut store, (BASE, N as i32)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f64 sum must be bit-exact"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 sum loop",
        );
    }

    /// An f64 loop combining sub / mul / div — `acc += (a[i] - t) * t / t` —
    /// runs end-to-end on the JIT tier with a bit-exact result (the residual ops
    /// preserve IEEE semantics and the accumulation order matches the stock
    /// executor). Includes the `t == 0` case so the IEEE inf/NaN path is exercised.
    #[test]
    fn end_to_end_f64_arith_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "acc") (param $ptr i32) (param $n i32) (param $t f64) (result f64)
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

        let vals: alloc::vec::Vec<f64> = (0..N)
            .map(|k| (k as f64) * 0.5 - 250.0 + (k as f64).cos())
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, f64), f64>(&store, "acc")
            .expect("acc func");

        // Reference: exactly the wasm accumulation order, same residual ops.
        let reference = |t: f64| vals.iter().fold(0.0f64, |a, &v| a + (v - t) * t / t);

        for &t in &[3.5f64, -2.0, 0.0] {
            let mut got = 0.0;
            for _ in 0..40 {
                got = f.call(&mut store, (BASE, N as i32, t)).expect("call");
            }
            assert_eq!(
                got.to_bits(),
                reference(t).to_bits(),
                "f64 arith must be bit-exact (t={t})",
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 arith loop",
        );
    }

    /// An f32 accumulation `acc += ((a[i] - t) * t) / t` over a linear-memory
    /// array runs end-to-end on the JIT tier BIT-EXACTLY — exercising the whole
    /// f32 Stage-1 surface: `f32.load` (into `freg32`), `f32.add` (`_Rsr`),
    /// `f32.sub`/`mul`/`div` (`_Rrs`), the f32 accumulator spill (`F32Copy`), and
    /// an f32 return. Proves the three-cell accumulator model (`accum[2]`).
    #[test]
    fn end_to_end_f32_arith_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "acc") (param $ptr i32) (param $n i32) (param $t f32) (result f32)
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

        let vals: alloc::vec::Vec<f32> = (0..N)
            .map(|k| (k as f32) * 0.5 - 250.0 + (k as f32).cos())
            .collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, f32), f32>(&store, "acc")
            .expect("acc func");

        // Reference: exactly the wasm accumulation order, same residual ops.
        let reference = |t: f32| vals.iter().fold(0.0f32, |a, &v| a + (v - t) * t / t);

        for &t in &[3.5f32, -2.0, 0.0] {
            let mut got = 0.0f32;
            for _ in 0..40 {
                got = f.call(&mut store, (BASE, N as i32, t)).expect("call");
            }
            assert_eq!(
                got.to_bits(),
                reference(t).to_bits(),
                "f32 arith must be bit-exact (t={t})",
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 arith loop",
        );
    }

    /// A loop accumulating `min(max(copysign(buf[i], sign), lo), hi)` runs
    /// end-to-end on the JIT tier with a bit-exact result, exercising the f32
    /// min/max/copysign residual (`sel` 0/1/2/3). `sign`/`lo`/`hi` are parameters
    /// (slots) so the ops lower onto the `_Rrs`/`_Rsr` forms; the data spans both
    /// signs and ±0.0. Runs with several sign/clamp parameter sets.
    #[test]
    fn end_to_end_f32_minmax_copysign_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "clamp_copysign")
                        (param $ptr i32) (param $n i32)
                        (param $sign f32) (param $lo f32) (param $hi f32) (result f32)
                    (local $i i32) (local $acc f32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (f32.add (local.get $acc)
                                    (f32.min
                                        (f32.max
                                            (f32.copysign
                                                (f32.load
                                                    (i32.add (local.get $ptr)
                                                             (i32.mul (local.get $i) (i32.const 4))))
                                                (local.get $sign))
                                            (local.get $lo))
                                        (local.get $hi))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        let vals: alloc::vec::Vec<f32> = (0..N)
            .map(|k| match k % 5 {
                1 => -0.0,
                2 => 0.0,
                3 => -(k as f32) * 0.25,
                _ => (k as f32) * 0.5 - 250.0,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }
        use crate::core::wasm;
        let reference = |sign: f32, lo: f32, hi: f32| -> f32 {
            vals.iter().fold(0.0f32, |a, &v| {
                a + wasm::f32_min(wasm::f32_max(wasm::f32_copysign(v, sign), lo), hi)
            })
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, f32, f32, f32), f32>(&store, "clamp_copysign")
            .expect("clamp_copysign func");

        for &(sign, lo, hi) in &[(-1.0f32, -50.0f32, 50.0f32), (2.0, -100.0, 100.0)] {
            let mut got = 0.0f32;
            for _ in 0..40 {
                got = f
                    .call(&mut store, (BASE, N as i32, sign, lo, hi))
                    .expect("call");
            }
            assert_eq!(
                got.to_bits(),
                reference(sign, lo, hi).to_bits(),
                "f32 min/max/copysign must be bit-exact (sign={sign}, lo={lo}, hi={hi})",
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 clamp/copysign loop",
        );
    }

    /// A loop chaining f32 arithmetic and min/max against folded `f32.const`
    /// operands (`_Rri`) runs end-to-end on the JIT tier with a bit-exact result.
    /// Each `f32.const` is fused as an inline immediate, pre-materialized into a
    /// scratch slot; this test exercises mul/sub/div/add/max/min const forms.
    #[test]
    fn end_to_end_f32_const_alu_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "f") (param $ptr i32) (param $n i32) (result f32)
                    (local $i i32) (local $acc f32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (f32.add (local.get $acc)
                                    (f32.min
                                        (f32.max
                                            (f32.add
                                                (f32.div
                                                    (f32.sub
                                                        (f32.mul
                                                            (f32.load
                                                                (i32.add (local.get $ptr)
                                                                         (i32.mul (local.get $i) (i32.const 4))))
                                                            (f32.const 2.0))
                                                        (f32.const 1.0))
                                                    (f32.const 3.0))
                                                (f32.const 0.5))
                                            (f32.const -10.0))
                                        (f32.const 10.0))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        let vals: alloc::vec::Vec<f32> = (0..N)
            .map(|k| (k as f32) * 0.3 - 150.0 + (k as f32).sin())
            .collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }
        use crate::core::wasm;
        // Mirror the WAT's operation order exactly (f32 arith == Rust IEEE f32).
        let reference = vals.iter().fold(0.0f32, |a, &v| {
            let e = (v * 2.0f32 - 1.0f32) / 3.0f32 + 0.5f32;
            a + wasm::f32_min(wasm::f32_max(e, -10.0f32), 10.0f32)
        });

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32), f32>(&store, "f")
            .expect("f func");

        let mut got = 0.0f32;
        for _ in 0..40 {
            got = f.call(&mut store, (BASE, N as i32)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            reference.to_bits(),
            "f32 const-operand arith/min/max must be bit-exact",
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 const-ALU loop",
        );
    }

    /// An f64 transform `out[i] = a[i] * scale` (a void function that f64-STOREs a
    /// computed value) runs end-to-end on the JIT tier: the output bytes are
    /// bit-exact, and an out-of-bounds destination traps.
    #[test]
    fn end_to_end_f64_store_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const SRC: i32 = 64;
        const DST: i32 = 64 + (N as i32) * 8;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "scale")
                        (param $src i32) (param $dst i32) (param $n i32) (param $s f64)
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

        let vals: alloc::vec::Vec<f64> = (0..N).map(|k| (k as f64) * 0.25 - 73.0).collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory.write(&mut store, SRC as usize, &raw).expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, i32, f64), ()>(&store, "scale")
            .expect("scale func");

        const SCALE: f64 = 1.5;
        for _ in 0..40 {
            f.call(&mut store, (SRC, DST, N as i32, SCALE))
                .expect("call");
        }
        let mut out = alloc::vec![0u8; N * 8];
        memory
            .read(&store, DST as usize, &mut out)
            .expect("read out");
        for k in 0..N {
            let got = f64::from_le_bytes(out[k * 8..k * 8 + 8].try_into().unwrap());
            assert_eq!(got.to_bits(), (vals[k] * SCALE).to_bits(), "scaled[{k}]");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 store loop",
        );

        // An out-of-bounds destination must trap (the page is 64 KiB).
        assert!(
            f.call(&mut store, (SRC, 60000, N as i32, SCALE)).is_err(),
            "an out-of-bounds f64 store must trap",
        );
    }

    /// The f32 counterpart of the store transform: `out[i] = a[i] * scale` (a void
    /// function that f32-STOREs a computed value, `F32StoreMem0Offset16_Sr`). Output
    /// bytes are bit-exact and an out-of-bounds destination traps.
    #[test]
    fn end_to_end_f32_store_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const SRC: i32 = 64;
        const DST: i32 = 64 + (N as i32) * 4;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "scale")
                        (param $src i32) (param $dst i32) (param $n i32) (param $s f32)
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

        let vals: alloc::vec::Vec<f32> = (0..N).map(|k| (k as f32) * 0.25 - 73.0).collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory.write(&mut store, SRC as usize, &raw).expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, i32, f32), ()>(&store, "scale")
            .expect("scale func");

        const SCALE: f32 = 1.5;
        for _ in 0..40 {
            f.call(&mut store, (SRC, DST, N as i32, SCALE))
                .expect("call");
        }
        let mut out = alloc::vec![0u8; N * 4];
        memory
            .read(&store, DST as usize, &mut out)
            .expect("read out");
        for k in 0..N {
            let got = f32::from_le_bytes(out[k * 4..k * 4 + 4].try_into().unwrap());
            assert_eq!(got.to_bits(), (vals[k] * SCALE).to_bits(), "scaled[{k}]");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 store loop",
        );

        // An out-of-bounds destination must trap. The page is 64 KiB (65536 bytes);
        // 1000 stride-4 stores from 64000 reach offset 65536 partway through.
        assert!(
            f.call(&mut store, (SRC, 64000, N as i32, SCALE)).is_err(),
            "an out-of-bounds f32 store must trap",
        );
    }

    /// Counting f64 array elements with `<` / `<=` / `==` / `!=` (each compare used
    /// as a 0/1 value) runs end-to-end on the JIT tier and matches a Rust
    /// reference, including NaN elements (where `!=` is true and the others false).
    /// All four modules stay alive to avoid the pointer-keyed stale-cache alias.
    #[test]
    fn end_to_end_f64_compare_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // Include a NaN every few lanes so the IEEE compare semantics are tested.
        let vals: alloc::vec::Vec<f64> = (0..N)
            .map(|k| {
                if k % 5 == 0 {
                    f64::NAN
                } else {
                    (k as f64) * 0.5 - 200.0
                }
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32, f64), i32>(&store, "count")
                .expect("count func");
            (store, f)
        };

        const LT_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "count")
                        (param $ptr i32) (param $n i32) (param $t f64) (result i32)
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_lt, f_lt) = build(LT_WAT);
        let (mut s_le, f_le) = build(&LT_WAT.replace("(f64.lt", "(f64.le"));
        let (mut s_eq, f_eq) = build(&LT_WAT.replace("(f64.lt", "(f64.eq"));
        let (mut s_ne, f_ne) = build(&LT_WAT.replace("(f64.lt", "(f64.ne"));

        const T: f64 = -12.5;
        let mut got = (0i32, 0, 0, 0);
        for _ in 0..40 {
            got.0 = f_lt.call(&mut s_lt, (BASE, N as i32, T)).expect("lt");
            got.1 = f_le.call(&mut s_le, (BASE, N as i32, T)).expect("le");
            got.2 = f_eq.call(&mut s_eq, (BASE, N as i32, T)).expect("eq");
            got.3 = f_ne.call(&mut s_ne, (BASE, N as i32, T)).expect("ne");
        }
        assert_eq!(
            got.0,
            vals.iter().filter(|&&v| v < T).count() as i32,
            "f64 lt"
        );
        assert_eq!(
            got.1,
            vals.iter().filter(|&&v| v <= T).count() as i32,
            "f64 le"
        );
        assert_eq!(
            got.2,
            vals.iter().filter(|&&v| v == T).count() as i32,
            "f64 eq"
        );
        assert_eq!(
            got.3,
            vals.iter().filter(|&&v| v != T).count() as i32,
            "f64 ne"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 compare loops",
        );
    }

    /// The f32 counterpart of the compare test: counting elements with `<` / `<=`
    /// / `==` / `!=` (each as a 0/1 value) runs end-to-end on the JIT tier and
    /// matches a Rust reference, including NaN elements. Stride 4, f32 types.
    #[test]
    fn end_to_end_f32_compare_value_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // Include a NaN every few lanes so the IEEE compare semantics are tested.
        let vals: alloc::vec::Vec<f32> = (0..N)
            .map(|k| {
                if k % 5 == 0 {
                    f32::NAN
                } else {
                    (k as f32) * 0.5 - 200.0
                }
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32, f32), i32>(&store, "count")
                .expect("count func");
            (store, f)
        };

        const LT_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "count")
                        (param $ptr i32) (param $n i32) (param $t f32) (result i32)
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

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_lt, f_lt) = build(LT_WAT);
        let (mut s_le, f_le) = build(&LT_WAT.replace("(f32.lt", "(f32.le"));
        let (mut s_eq, f_eq) = build(&LT_WAT.replace("(f32.lt", "(f32.eq"));
        let (mut s_ne, f_ne) = build(&LT_WAT.replace("(f32.lt", "(f32.ne"));

        const T: f32 = -12.5;
        let mut got = (0i32, 0, 0, 0);
        for _ in 0..40 {
            got.0 = f_lt.call(&mut s_lt, (BASE, N as i32, T)).expect("lt");
            got.1 = f_le.call(&mut s_le, (BASE, N as i32, T)).expect("le");
            got.2 = f_eq.call(&mut s_eq, (BASE, N as i32, T)).expect("eq");
            got.3 = f_ne.call(&mut s_ne, (BASE, N as i32, T)).expect("ne");
        }
        assert_eq!(
            got.0,
            vals.iter().filter(|&&v| v < T).count() as i32,
            "f32 lt"
        );
        assert_eq!(
            got.1,
            vals.iter().filter(|&&v| v <= T).count() as i32,
            "f32 le"
        );
        assert_eq!(
            got.2,
            vals.iter().filter(|&&v| v == T).count() as i32,
            "f32 eq"
        );
        assert_eq!(
            got.3,
            vals.iter().filter(|&&v| v != T).count() as i32,
            "f32 ne"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 compare loops",
        );
    }

    /// An f64 loop that combines all four arithmetic ops against folded constants
    /// — `sum += ((a[i] * 2.0) - 1.0) / 4.0 + 0.5` — runs end-to-end on the JIT
    /// tier with a bit-exact result (the constants are pre-materialized into a
    /// scratch slot and the same residual ops run).
    #[test]
    fn end_to_end_f64_const_arith_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result f64)
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

        let vals: alloc::vec::Vec<f64> = (0..N).map(|k| (k as f64) * 0.123 - 61.0).collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }
        let expected = vals
            .iter()
            .fold(0.0f64, |s, &v| s + ((v * 2.0) - 1.0) / 4.0 + 0.5);

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32), f64>(&store, "sum")
            .expect("sum func");

        let mut got = 0.0;
        for _ in 0..40 {
            got = f.call(&mut store, (BASE, N as i32)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f64 const arith must be bit-exact"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 const-arith loop",
        );
    }

    /// An f64 loop accumulating the unary ops `sqrt(abs(a[i]))` plus a separately
    /// summed `floor`/`ceil`/`trunc`/`nearest` of each element runs end-to-end on
    /// the JIT tier with a bit-exact result against a Rust reference (the input
    /// includes negatives and `.5` fractions to exercise the rounding modes).
    #[test]
    fn end_to_end_f64_unary_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result f64)
                    (local $i i32) (local $sum f64) (local $x f64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $x
                                (f64.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 8)))))
                            (local.set $sum
                                (f64.add (local.get $sum)
                                    (f64.add
                                        (f64.sqrt (f64.abs (local.get $x)))
                                        (f64.add
                                            (f64.floor (local.get $x))
                                            (f64.add
                                                (f64.ceil (local.get $x))
                                                (f64.add
                                                    (f64.trunc (local.get $x))
                                                    (f64.add
                                                        (f64.nearest (local.get $x))
                                                        (f64.neg (local.get $x)))))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;

        // Negatives and exact `.5` fractions exercise abs, sign, and the
        // round-half-to-even (`nearest`) tie rule.
        let vals: alloc::vec::Vec<f64> = (0..N).map(|k| (k as f64) * 0.5 - 250.0).collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }
        // Reference uses the same canonical wasm float helpers stock wasmi runs, so
        // a bit-exact match proves the JIT plumbing (load → slot → residual → acc).
        // f64 add is not associative, so the grouping must match the WAT's
        // right-associated `f64.add` tree exactly.
        use crate::core::wasm;
        let expected = vals.iter().fold(0.0f64, |s, &x| {
            let a = wasm::f64_sqrt(wasm::f64_abs(x));
            let b = wasm::f64_floor(x);
            let c = wasm::f64_ceil(x);
            let d = wasm::f64_trunc(x);
            let e = wasm::f64_nearest(x);
            let g = wasm::f64_neg(x);
            s + (a + (b + (c + (d + (e + g)))))
        });

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32), f64>(&store, "sum")
            .expect("sum func");

        let mut got = 0.0;
        for _ in 0..40 {
            got = f.call(&mut store, (BASE, N as i32)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f64 unary ops must be bit-exact"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 unary loop",
        );
    }

    /// The f32 counterpart of the unary test: `sqrt(abs(x))` plus a summed
    /// `floor`/`ceil`/`trunc`/`nearest`/`neg` of each element, bit-exact against a
    /// Rust reference using the same wasm helpers. Stride 4, f32 types.
    #[test]
    fn end_to_end_f32_unary_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result f32)
                    (local $i i32) (local $sum f32) (local $x f32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $x
                                (f32.load
                                    (i32.add (local.get $ptr)
                                             (i32.mul (local.get $i) (i32.const 4)))))
                            (local.set $sum
                                (f32.add (local.get $sum)
                                    (f32.add
                                        (f32.sqrt (f32.abs (local.get $x)))
                                        (f32.add
                                            (f32.floor (local.get $x))
                                            (f32.add
                                                (f32.ceil (local.get $x))
                                                (f32.add
                                                    (f32.trunc (local.get $x))
                                                    (f32.add
                                                        (f32.nearest (local.get $x))
                                                        (f32.neg (local.get $x)))))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;

        let vals: alloc::vec::Vec<f32> = (0..N).map(|k| (k as f32) * 0.5 - 250.0).collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }
        // f32 add is not associative, so the reference grouping mirrors the WAT's
        // right-associated `f32.add` tree exactly.
        use crate::core::wasm;
        let expected = vals.iter().fold(0.0f32, |s, &x| {
            let a = wasm::f32_sqrt(wasm::f32_abs(x));
            let b = wasm::f32_floor(x);
            let c = wasm::f32_ceil(x);
            let d = wasm::f32_trunc(x);
            let e = wasm::f32_nearest(x);
            let g = wasm::f32_neg(x);
            s + (a + (b + (c + (d + (e + g)))))
        });

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32), f32>(&store, "sum")
            .expect("sum func");

        let mut got = 0.0f32;
        for _ in 0..40 {
            got = f.call(&mut store, (BASE, N as i32)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f32 unary ops must be bit-exact"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 unary loop",
        );
    }

    /// An f32 conversion loop — integer→f32 convert (signed/unsigned, 32/64-bit),
    /// an f32→f64→f32 promote/demote round-trip, and an i32↔f32 reinterpret
    /// round-trip — runs end-to-end on the JIT tier with a bit-exact result against
    /// a Rust reference. The i64 parameter is negative so the signed/unsigned
    /// convert paths diverge.
    #[test]
    fn end_to_end_f32_convert_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const Q: i64 = -5;

        const WAT: &str = r#"
            (module
                (func (export "run") (param $n i32) (param $q i64) (result f32)
                    (local $i i32) (local $sum f32) (local $x f32) (local $ib i32)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f32.add (local.get $sum)
                                    (f32.add (f32.convert_i32_s (local.get $i))
                                        (f32.add (f32.convert_i32_u (local.get $i))
                                            (f32.add (f32.convert_i64_s (local.get $q))
                                                (f32.convert_i64_u (local.get $q)))))))
                            (local.set $x (f32.convert_i32_s (local.get $i)))
                            (local.set $sum
                                (f32.add (local.get $sum)
                                    (f32.demote_f64 (f64.promote_f32 (local.get $x)))))
                            (local.set $ib (i32.reinterpret_f32 (local.get $x)))
                            (local.set $sum
                                (f32.add (local.get $sum)
                                    (f32.reinterpret_i32 (local.get $ib))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;

        // Reference mirrors the WAT's statement order and right-associated add tree,
        // using the same canonical wasm helpers the stock executor runs.
        use crate::core::wasm;
        let expected = {
            let mut sum = 0.0f32;
            for i in 0..N as i32 {
                let a = wasm::f32_convert_i32_s(i);
                let b = wasm::f32_convert_i32_u(i as u32);
                let c = wasm::f32_convert_i64_s(Q);
                let d = wasm::f32_convert_i64_u(Q as u64);
                sum = sum + (a + (b + (c + d)));
                let x = wasm::f32_convert_i32_s(i);
                sum = sum + wasm::f32_demote_f64(wasm::f64_promote_f32(x));
                let ib = wasm::i32_reinterpret_f32(x);
                sum = sum + wasm::f32_reinterpret_i32(ib);
            }
            sum
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<(i32, i64), f32>(&store, "run")
            .expect("run func");

        let mut got = 0.0f32;
        for _ in 0..40 {
            got = f.call(&mut store, (N as i32, Q)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f32 conversions must be bit-exact"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 conversion loop",
        );
    }

    /// An f64 clamp loop `sum += min(max(a[i], lo), hi)` (lo a parameter slot, hi
    /// a folded constant) runs end-to-end on the JIT tier with a bit-exact result.
    /// The data includes NaN and ±0.0 to exercise wasm's NaN-propagating /
    /// signed-zero min/max semantics.
    #[test]
    fn end_to_end_f64_minmax_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "clamp")
                        (param $ptr i32) (param $n i32) (param $lo f64) (result f64)
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

        // A spread spanning below-LO and above-HI (so both clamp arms fire) plus
        // signed zeros. NaN is excluded from the accumulation because it would
        // poison the running sum to NaN and mask a clamp-value mismatch; the
        // NaN-propagating path is covered by the f64-compare e2e test.
        let vals: alloc::vec::Vec<f64> = (0..N)
            .map(|k| match k % 7 {
                1 => -0.0,
                2 => 0.0,
                _ => (k as f64) * 0.5 - 250.0,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }
        // Reference uses the same canonical wasm min/max helpers stock wasmi runs;
        // f64 add must follow the WAT's `sum + clamp` association.
        use crate::core::wasm;
        const LO: f64 = -3.0;
        const HI: f64 = 100.0;
        let expected = vals
            .iter()
            .fold(0.0f64, |s, &x| s + wasm::f64_min(wasm::f64_max(x, LO), HI));

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, f64), f64>(&store, "clamp")
            .expect("clamp func");

        let mut got = 0.0;
        for _ in 0..40 {
            got = f.call(&mut store, (BASE, N as i32, LO)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f64 min/max must be bit-exact"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 clamp loop",
        );
    }

    /// A loop summing `copysign(buf[i], sign)` runs end-to-end on the JIT tier
    /// with a bit-exact result. The magnitude comes from a load (accumulator) and
    /// the sign from a parameter (slot), so wasmi emits the `_Rrs`/`_Rsr` forms;
    /// the data spans both signs and ±0.0 so the sign transfer is exercised in
    /// both directions. Runs twice with opposite sign parameters.
    #[test]
    fn end_to_end_f64_copysign_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum_copysign")
                        (param $ptr i32) (param $n i32) (param $sign f64) (result f64)
                    (local $i i32) (local $sum f64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (f64.add (local.get $sum)
                                    (f64.copysign
                                        (f64.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 8))))
                                        (local.get $sign))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;

        let vals: alloc::vec::Vec<f64> = (0..N)
            .map(|k| match k % 5 {
                1 => -0.0,
                2 => 0.0,
                3 => -(k as f64) * 0.25,
                _ => (k as f64) * 0.5 - 250.0,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }
        use crate::core::wasm;
        let expected = |sign: f64| -> f64 {
            vals.iter()
                .fold(0.0f64, |s, &x| s + wasm::f64_copysign(x, sign))
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        memory
            .write(&mut store, BASE as usize, &raw)
            .expect("write");
        let f = instance
            .get_typed_func::<(i32, i32, f64), f64>(&store, "sum_copysign")
            .expect("sum_copysign func");

        for sign in [-3.0f64, 7.0f64] {
            let mut got = 0.0;
            for _ in 0..40 {
                got = f.call(&mut store, (BASE, N as i32, sign)).expect("call");
            }
            assert_eq!(
                got.to_bits(),
                expected(sign).to_bits(),
                "f64 copysign must be bit-exact (sign={sign})"
            );
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 copysign loop",
        );
    }

    /// A loop summing all four widening int→f64 conversions runs end-to-end on the
    /// JIT tier with a bit-exact result. The i32 value `i - 500` and the i64 value
    /// `j + i` both go negative, so the signed vs unsigned conversions diverge
    /// (exercising both); the i64→f64 path needs the slot sign-extend too.
    #[test]
    fn end_to_end_f64_convert_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const WAT: &str = r#"
            (module
                (func (export "conv") (param $n i32) (param $j i64) (result f64)
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

        const N: i32 = 1000;
        const J: i64 = -1_000_000;
        // Reference uses the same canonical wasm conversion helpers stock wasmi
        // runs; the f64 add tree is right-associated to match the WAT exactly.
        use crate::core::wasm;
        let expected = (0..N).fold(0.0f64, |s, i| {
            let x = i.wrapping_sub(500);
            let y = J.wrapping_add(i as i64);
            let a = wasm::f64_convert_i32_s(x);
            let b = wasm::f64_convert_i32_u(x as u32);
            let c = wasm::f64_convert_i64_s(y);
            let d = wasm::f64_convert_i64_u(y as u64);
            s + (a + (b + (c + d)))
        });

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<(i32, i64), f64>(&store, "conv")
            .expect("conv func");

        let mut got = 0.0;
        for _ in 0..40 {
            got = f.call(&mut store, (N, J)).expect("call");
        }
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f64 conversions must be bit-exact"
        );
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 conversion loop",
        );
    }

    /// All four saturating f64→int truncations run end-to-end on the JIT tier with
    /// bit-exact results. The data spans NaN (→0), ±inf and out-of-range values
    /// (→ the integer min/max), and negatives (→0 for the unsigned forms). Each
    /// variant is a separate module summing one trunc; all stay alive at once to
    /// avoid the op-stream-pointer stale-cache alias.
    #[test]
    fn end_to_end_f64_trunc_sat_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        let vals: alloc::vec::Vec<f64> = (0..N)
            .map(|k| match k % 8 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => 1.0e300,
                4 => -1.0e300,
                5 => -3.7,
                6 => 9.9e18,
                _ => (k as f64) * 0.5 - 13.25,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }

        const I64S_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result i64)
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
        let i32s_wat = I64S_WAT
            .replace("(local $sum i64)", "(local $sum i32)")
            .replace("(result i64)", "(result i32)")
            .replace("i64.add", "i32.add")
            .replace("i64.trunc_sat_f64_s", "i32.trunc_sat_f64_s");
        let i64u_wat = I64S_WAT.replace("i64.trunc_sat_f64_s", "i64.trunc_sat_f64_u");
        let i32u_wat = i32s_wat.replace("i32.trunc_sat_f64_s", "i32.trunc_sat_f64_u");

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            (store, instance)
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_i64s, in_i64s) = build(I64S_WAT);
        let (mut s_i64u, in_i64u) = build(&i64u_wat);
        let (mut s_i32s, in_i32s) = build(&i32s_wat);
        let (mut s_i32u, in_i32u) = build(&i32u_wat);
        let f_i64s = in_i64s
            .get_typed_func::<(i32, i32), i64>(&s_i64s, "sum")
            .expect("i64s");
        let f_i64u = in_i64u
            .get_typed_func::<(i32, i32), i64>(&s_i64u, "sum")
            .expect("i64u");
        let f_i32s = in_i32s
            .get_typed_func::<(i32, i32), i32>(&s_i32s, "sum")
            .expect("i32s");
        let f_i32u = in_i32u
            .get_typed_func::<(i32, i32), i32>(&s_i32u, "sum")
            .expect("i32u");

        use crate::core::wasm;
        let exp_i64s = vals
            .iter()
            .fold(0i64, |s, &v| s.wrapping_add(wasm::i64_trunc_sat_f64_s(v)));
        let exp_i64u = vals.iter().fold(0i64, |s, &v| {
            s.wrapping_add(wasm::i64_trunc_sat_f64_u(v) as i64)
        });
        let exp_i32s = vals
            .iter()
            .fold(0i32, |s, &v| s.wrapping_add(wasm::i32_trunc_sat_f64_s(v)));
        let exp_i32u = vals.iter().fold(0i32, |s, &v| {
            s.wrapping_add(wasm::i32_trunc_sat_f64_u(v) as i32)
        });

        let (mut g64s, mut g64u, mut g32s, mut g32u) = (0i64, 0i64, 0i32, 0i32);
        for _ in 0..40 {
            g64s = f_i64s
                .call(&mut s_i64s, (BASE, N as i32))
                .expect("call i64s");
            g64u = f_i64u
                .call(&mut s_i64u, (BASE, N as i32))
                .expect("call i64u");
            g32s = f_i32s
                .call(&mut s_i32s, (BASE, N as i32))
                .expect("call i32s");
            g32u = f_i32u
                .call(&mut s_i32u, (BASE, N as i32))
                .expect("call i32u");
        }
        assert_eq!(g64s, exp_i64s, "trunc_sat_i64_s");
        assert_eq!(g64u, exp_i64u, "trunc_sat_i64_u");
        assert_eq!(g32s, exp_i32s, "trunc_sat_i32_s");
        assert_eq!(g32u, exp_i32u, "trunc_sat_i32_u");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the trunc_sat loops",
        );
    }

    /// All four trapping f64→int truncations run end-to-end on the JIT tier with
    /// bit-exact results for in-range, non-negative data (no input traps). Each
    /// variant is a separate module summing one trunc; all stay alive at once to
    /// avoid the op-stream-pointer stale-cache alias.
    #[test]
    fn end_to_end_f64_trunc_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // In range for every variant (fits i32, and non-negative for the unsigned
        // forms), so no input traps.
        let vals: alloc::vec::Vec<f64> = (0..N).map(|k| (k % 97) as f64 + 0.5).collect();
        let mut raw = alloc::vec![0u8; N * 8];
        for k in 0..N {
            raw[k * 8..k * 8 + 8].copy_from_slice(&vals[k].to_le_bytes());
        }

        const I64S_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result i64)
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
        let i32s_wat = I64S_WAT
            .replace("(local $sum i64)", "(local $sum i32)")
            .replace("(result i64)", "(result i32)")
            .replace("i64.add", "i32.add")
            .replace("i64.trunc_f64_s", "i32.trunc_f64_s");
        let i64u_wat = I64S_WAT.replace("i64.trunc_f64_s", "i64.trunc_f64_u");
        let i32u_wat = i32s_wat.replace("i32.trunc_f64_s", "i32.trunc_f64_u");

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            (store, instance)
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_i64s, in_i64s) = build(I64S_WAT);
        let (mut s_i64u, in_i64u) = build(&i64u_wat);
        let (mut s_i32s, in_i32s) = build(&i32s_wat);
        let (mut s_i32u, in_i32u) = build(&i32u_wat);
        let f_i64s = in_i64s
            .get_typed_func::<(i32, i32), i64>(&s_i64s, "sum")
            .expect("i64s");
        let f_i64u = in_i64u
            .get_typed_func::<(i32, i32), i64>(&s_i64u, "sum")
            .expect("i64u");
        let f_i32s = in_i32s
            .get_typed_func::<(i32, i32), i32>(&s_i32s, "sum")
            .expect("i32s");
        let f_i32u = in_i32u
            .get_typed_func::<(i32, i32), i32>(&s_i32u, "sum")
            .expect("i32u");

        use crate::core::wasm;
        let exp_i64s = vals.iter().fold(0i64, |s, &v| {
            s.wrapping_add(wasm::i64_trunc_f64_s(v).unwrap())
        });
        let exp_i64u = vals.iter().fold(0i64, |s, &v| {
            s.wrapping_add(wasm::i64_trunc_f64_u(v).unwrap() as i64)
        });
        let exp_i32s = vals.iter().fold(0i32, |s, &v| {
            s.wrapping_add(wasm::i32_trunc_f64_s(v).unwrap())
        });
        let exp_i32u = vals.iter().fold(0i32, |s, &v| {
            s.wrapping_add(wasm::i32_trunc_f64_u(v).unwrap() as i32)
        });

        let (mut g64s, mut g64u, mut g32s, mut g32u) = (0i64, 0i64, 0i32, 0i32);
        for _ in 0..40 {
            g64s = f_i64s
                .call(&mut s_i64s, (BASE, N as i32))
                .expect("call i64s");
            g64u = f_i64u
                .call(&mut s_i64u, (BASE, N as i32))
                .expect("call i64u");
            g32s = f_i32s
                .call(&mut s_i32s, (BASE, N as i32))
                .expect("call i32s");
            g32u = f_i32u
                .call(&mut s_i32u, (BASE, N as i32))
                .expect("call i32u");
        }
        assert_eq!(g64s, exp_i64s, "trunc_i64_s");
        assert_eq!(g64u, exp_i64u, "trunc_i64_u");
        assert_eq!(g32s, exp_i32s, "trunc_i32_s");
        assert_eq!(g32u, exp_i32u, "trunc_i32_u");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the trunc loops",
        );
    }

    /// A trapping f64→int truncation surfaces the faithful trap: NaN raises
    /// `BadConversionToInteger`, an out-of-range value raises `IntegerOverflow`.
    /// The function does no store, so recovery re-runs the stock executor; this
    /// proves the residual flags the trap rather than silently returning 0.
    #[test]
    fn end_to_end_f64_trunc_traps_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const BASE: i32 = 64;
        const SUM_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result i64)
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

        let run = |vals: &[f64]| -> crate::TrapCode {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, SUM_WAT).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            let mut raw = alloc::vec![0u8; vals.len() * 8];
            for (k, &v) in vals.iter().enumerate() {
                raw[k * 8..k * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32), i64>(&store, "sum")
                .expect("typed func");
            let err = f
                .call(&mut store, (BASE, vals.len() as i32))
                .expect_err("a trapping truncation must surface a trap");
            err.as_trap_code().expect("a wasm trap code")
        };

        // A few in-range values, then a NaN: must trap with BadConversionToInteger.
        assert_eq!(
            run(&[1.0, 2.0, 3.0, f64::NAN, 5.0]),
            crate::TrapCode::BadConversionToInteger,
        );
        // An out-of-range magnitude (> i64::MAX) must trap with IntegerOverflow.
        assert_eq!(
            run(&[1.0, 2.0, 1.0e19, 4.0]),
            crate::TrapCode::IntegerOverflow,
        );
    }

    /// When a store has already committed this run and a later truncation traps,
    /// `run_jit` raises the recorded conversion trap code directly (it cannot
    /// re-run stock without double-applying the store). The function is warmed to
    /// the JIT tier first, then fed a NaN; the trap must be `BadConversionToInteger`
    /// (not the memory `MemoryOutOfBounds` default), proving the recorded code is
    /// threaded through.
    #[test]
    fn end_to_end_f64_trunc_trap_after_store_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 100;
        const SRC: i32 = 64;
        const DST: i32 = SRC + (N as i32) * 8 + 64;
        // Per iteration: store `i` to dst[i] (commits), then accumulate
        // trunc_f64_s(src[i]) (may trap). Store precedes the trap in program order.
        const WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "g") (param $src i32) (param $dst i32) (param $n i32) (result i64)
                    (local $i i32) (local $acc i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (i64.store
                                (i32.add (local.get $dst) (i32.mul (local.get $i) (i32.const 8)))
                                (i64.extend_i32_s (local.get $i)))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (i64.trunc_f64_s
                                        (f64.load
                                            (i32.add (local.get $src)
                                                     (i32.mul (local.get $i) (i32.const 8)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $acc)))
        "#;

        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let memory = instance.get_memory(&store, "mem").expect("memory export");
        let f = instance
            .get_typed_func::<(i32, i32, i32), i64>(&store, "g")
            .expect("typed func");

        // Seed in-range src data and warm the function onto the JIT tier.
        let mut good = alloc::vec![0u8; N * 8];
        for k in 0..N {
            good[k * 8..k * 8 + 8].copy_from_slice(&((k % 50) as f64 + 0.5).to_le_bytes());
        }
        memory
            .write(&mut store, SRC as usize, &good)
            .expect("write good");
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        for _ in 0..40 {
            f.call(&mut store, (SRC, DST, N as i32)).expect("warm call");
        }
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the store+trunc loop",
        );

        // Now make src[0] a NaN: iteration 0 stores dst[0] (commits), then the
        // truncation traps. Recovery must raise the recorded conversion code.
        memory
            .write(&mut store, SRC as usize, &f64::NAN.to_le_bytes())
            .expect("write nan");
        let err = f
            .call(&mut store, (SRC, DST, N as i32))
            .expect_err("the trapping truncation must surface a trap");
        assert_eq!(
            err.as_trap_code(),
            Some(crate::TrapCode::BadConversionToInteger),
            "a committed-store run must raise the recorded conversion trap code",
        );
    }

    /// All four saturating f32→int truncations run end-to-end on the JIT tier and
    /// produce bit-exact results, including the saturating cases (NaN→0, ±inf and
    /// out-of-range magnitudes clamp to the integer bounds). Every variant is a
    /// separate module summing one trunc_sat; all stay alive at once to avoid the
    /// op-stream-pointer stale-cache alias.
    #[test]
    fn end_to_end_f32_trunc_sat_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        let vals: alloc::vec::Vec<f32> = (0..N)
            .map(|k| match k % 8 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => 1.0e30,
                4 => -1.0e30,
                5 => -3.7,
                6 => 9.9e9,
                _ => (k as f32) * 0.5 - 13.25,
            })
            .collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        const I64S_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result i64)
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
        let i64u_wat = I64S_WAT.replace("i64.trunc_sat_f32_s", "i64.trunc_sat_f32_u");
        let i32u_wat = i32s_wat.replace("i32.trunc_sat_f32_s", "i32.trunc_sat_f32_u");

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            (store, instance)
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_i64s, in_i64s) = build(I64S_WAT);
        let (mut s_i64u, in_i64u) = build(&i64u_wat);
        let (mut s_i32s, in_i32s) = build(&i32s_wat);
        let (mut s_i32u, in_i32u) = build(&i32u_wat);
        let f_i64s = in_i64s
            .get_typed_func::<(i32, i32), i64>(&s_i64s, "sum")
            .expect("i64s");
        let f_i64u = in_i64u
            .get_typed_func::<(i32, i32), i64>(&s_i64u, "sum")
            .expect("i64u");
        let f_i32s = in_i32s
            .get_typed_func::<(i32, i32), i32>(&s_i32s, "sum")
            .expect("i32s");
        let f_i32u = in_i32u
            .get_typed_func::<(i32, i32), i32>(&s_i32u, "sum")
            .expect("i32u");

        use crate::core::wasm;
        let exp_i64s = vals
            .iter()
            .fold(0i64, |s, &v| s.wrapping_add(wasm::i64_trunc_sat_f32_s(v)));
        let exp_i64u = vals.iter().fold(0i64, |s, &v| {
            s.wrapping_add(wasm::i64_trunc_sat_f32_u(v) as i64)
        });
        let exp_i32s = vals
            .iter()
            .fold(0i32, |s, &v| s.wrapping_add(wasm::i32_trunc_sat_f32_s(v)));
        let exp_i32u = vals.iter().fold(0i32, |s, &v| {
            s.wrapping_add(wasm::i32_trunc_sat_f32_u(v) as i32)
        });

        let (mut g64s, mut g64u, mut g32s, mut g32u) = (0i64, 0i64, 0i32, 0i32);
        for _ in 0..40 {
            g64s = f_i64s
                .call(&mut s_i64s, (BASE, N as i32))
                .expect("call i64s");
            g64u = f_i64u
                .call(&mut s_i64u, (BASE, N as i32))
                .expect("call i64u");
            g32s = f_i32s
                .call(&mut s_i32s, (BASE, N as i32))
                .expect("call i32s");
            g32u = f_i32u
                .call(&mut s_i32u, (BASE, N as i32))
                .expect("call i32u");
        }
        assert_eq!(g64s, exp_i64s, "trunc_sat_f32_i64_s");
        assert_eq!(g64u, exp_i64u, "trunc_sat_f32_i64_u");
        assert_eq!(g32s, exp_i32s, "trunc_sat_f32_i32_s");
        assert_eq!(g32u, exp_i32u, "trunc_sat_f32_i32_u");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 trunc_sat loops",
        );
    }

    /// All four trapping f32→int truncations run end-to-end on the JIT tier with
    /// bit-exact results for in-range, non-negative data (no input traps).
    #[test]
    fn end_to_end_f32_trunc_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: usize = 1000;
        const BASE: i32 = 64;

        // In range for every variant (fits i32, and non-negative for the unsigned
        // forms), so no input traps.
        let vals: alloc::vec::Vec<f32> = (0..N).map(|k| (k % 97) as f32 + 0.5).collect();
        let mut raw = alloc::vec![0u8; N * 4];
        for k in 0..N {
            raw[k * 4..k * 4 + 4].copy_from_slice(&vals[k].to_le_bytes());
        }

        const I64S_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result i64)
                    (local $i i32) (local $sum i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.trunc_f32_s
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
            .replace("i64.trunc_f32_s", "i32.trunc_f32_s");
        let i64u_wat = I64S_WAT.replace("i64.trunc_f32_s", "i64.trunc_f32_u");
        let i32u_wat = i32s_wat.replace("i32.trunc_f32_s", "i32.trunc_f32_u");

        let build = |wat: &str| {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, wat).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            (store, instance)
        };

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let (mut s_i64s, in_i64s) = build(I64S_WAT);
        let (mut s_i64u, in_i64u) = build(&i64u_wat);
        let (mut s_i32s, in_i32s) = build(&i32s_wat);
        let (mut s_i32u, in_i32u) = build(&i32u_wat);
        let f_i64s = in_i64s
            .get_typed_func::<(i32, i32), i64>(&s_i64s, "sum")
            .expect("i64s");
        let f_i64u = in_i64u
            .get_typed_func::<(i32, i32), i64>(&s_i64u, "sum")
            .expect("i64u");
        let f_i32s = in_i32s
            .get_typed_func::<(i32, i32), i32>(&s_i32s, "sum")
            .expect("i32s");
        let f_i32u = in_i32u
            .get_typed_func::<(i32, i32), i32>(&s_i32u, "sum")
            .expect("i32u");

        use crate::core::wasm;
        let exp_i64s = vals.iter().fold(0i64, |s, &v| {
            s.wrapping_add(wasm::i64_trunc_f32_s(v).unwrap())
        });
        let exp_i64u = vals.iter().fold(0i64, |s, &v| {
            s.wrapping_add(wasm::i64_trunc_f32_u(v).unwrap() as i64)
        });
        let exp_i32s = vals.iter().fold(0i32, |s, &v| {
            s.wrapping_add(wasm::i32_trunc_f32_s(v).unwrap())
        });
        let exp_i32u = vals.iter().fold(0i32, |s, &v| {
            s.wrapping_add(wasm::i32_trunc_f32_u(v).unwrap() as i32)
        });

        let (mut g64s, mut g64u, mut g32s, mut g32u) = (0i64, 0i64, 0i32, 0i32);
        for _ in 0..40 {
            g64s = f_i64s
                .call(&mut s_i64s, (BASE, N as i32))
                .expect("call i64s");
            g64u = f_i64u
                .call(&mut s_i64u, (BASE, N as i32))
                .expect("call i64u");
            g32s = f_i32s
                .call(&mut s_i32s, (BASE, N as i32))
                .expect("call i32s");
            g32u = f_i32u
                .call(&mut s_i32u, (BASE, N as i32))
                .expect("call i32u");
        }
        assert_eq!(g64s, exp_i64s, "trunc_f32_i64_s");
        assert_eq!(g64u, exp_i64u, "trunc_f32_i64_u");
        assert_eq!(g32s, exp_i32s, "trunc_f32_i32_s");
        assert_eq!(g32u, exp_i32u, "trunc_f32_i32_u");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 trunc loops",
        );
    }

    /// A trapping f32→int truncation surfaces the faithful trap: NaN raises
    /// `BadConversionToInteger`, an out-of-range value raises `IntegerOverflow`.
    /// The function does no store, so recovery re-runs the stock executor; this
    /// proves the residual flags the trap rather than silently returning 0.
    #[test]
    fn end_to_end_f32_trunc_traps_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const BASE: i32 = 64;
        const SUM_WAT: &str = r#"
            (module
                (memory (export "mem") 1)
                (func (export "sum") (param $ptr i32) (param $n i32) (result i64)
                    (local $i i32) (local $sum i64)
                    (block $break
                        (loop $continue
                            (br_if $break (i32.ge_s (local.get $i) (local.get $n)))
                            (local.set $sum
                                (i64.add (local.get $sum)
                                    (i64.trunc_f32_s
                                        (f32.load
                                            (i32.add (local.get $ptr)
                                                     (i32.mul (local.get $i) (i32.const 4)))))))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $continue)))
                    (local.get $sum)))
        "#;

        let run = |vals: &[f32]| -> crate::TrapCode {
            let engine = Engine::default();
            let mut store = Store::new(&engine, ());
            let module = Module::new(&engine, SUM_WAT).expect("module");
            let instance = Instance::new(&mut store, &module, &[]).expect("instance");
            let memory = instance.get_memory(&store, "mem").expect("memory export");
            let mut raw = alloc::vec![0u8; vals.len() * 4];
            for (k, &v) in vals.iter().enumerate() {
                raw[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            memory
                .write(&mut store, BASE as usize, &raw)
                .expect("write");
            let f = instance
                .get_typed_func::<(i32, i32), i64>(&store, "sum")
                .expect("typed func");
            let err = f
                .call(&mut store, (BASE, vals.len() as i32))
                .expect_err("a trapping truncation must surface a trap");
            err.as_trap_code().expect("a wasm trap code")
        };

        // A few in-range values, then a NaN: must trap with BadConversionToInteger.
        assert_eq!(
            run(&[1.0, 2.0, 3.0, f32::NAN, 5.0]),
            crate::TrapCode::BadConversionToInteger,
        );
        // An out-of-range magnitude (> i64::MAX) must trap with IntegerOverflow.
        assert_eq!(
            run(&[1.0, 2.0, 1.0e19, 4.0]),
            crate::TrapCode::IntegerOverflow,
        );
    }

    /// An integer global used as loop-carried state runs end-to-end on the JIT
    /// tier: each call resets the global to 0 (an immediate `global.set`), then the
    /// hot loop reads-modifies-writes it (`global.get` + register `global.set`), and
    /// the function returns it (`global.get`). The return value matches the closed
    /// form and the host-visible global reflects the last run, proving the residual
    /// global helpers read and write the real store through `GLOBALS_CTX`.
    #[test]
    fn end_to_end_globals_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: i32 = 1000;
        const WAT: &str = r#"
            (module
                (global $g (export "g") (mut i64) (i64.const 0))
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

        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<i32, i64>(&store, "run")
            .expect("typed func");

        // sum(0..N) = N*(N-1)/2.
        let expected = (i64::from(N) * i64::from(N - 1)) / 2;

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let mut got = 0i64;
        for _ in 0..40 {
            got = f.call(&mut store, N).expect("call");
        }
        assert_eq!(got, expected, "global-carried sum");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the global read/modify/write loop",
        );

        // The host-visible global reflects the last JIT run's committed writes.
        let g = instance.get_global(&store, "g").expect("global export");
        assert_eq!(g.get(&store).i64(), Some(expected), "host-visible global");
    }

    /// f32 and f64 globals used as loop-carried accumulators run end-to-end on the
    /// JIT tier with bit-exact results. Each call adds `convert_i32(i)` for `i` in
    /// `0..N` into the global (`global.get`+float`global.set`, no reset), so over
    /// the warm-up calls the global compounds; the Rust reference replays the exact
    /// same float additions in the same order, so the bit patterns must match. Also
    /// confirms the host-visible float global reflects the committed writes.
    #[test]
    fn end_to_end_float_globals_jit_tier() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        const N: i32 = 1000;
        const CALLS: usize = 40;
        const F64_WAT: &str = r#"
            (module
                (global $g (export "g") (mut f64) (f64.const 0))
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
        let f32_wat = F64_WAT
            .replace("mut f64", "mut f32")
            .replace("(result f64)", "(result f32)")
            .replace("f64.const 0", "f32.const 0")
            .replace("f64.add", "f32.add")
            .replace("f64.convert_i32_s", "f32.convert_i32_s");

        // --- f64 global ---
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, F64_WAT).expect("f64 module");
        let instance = Instance::new(&mut store, &module, &[]).expect("f64 instance");
        let f = instance
            .get_typed_func::<i32, f64>(&store, "run")
            .expect("f64 run");
        let mut ref64 = 0f64;
        let mut got64 = 0f64;
        for _ in 0..CALLS {
            got64 = f.call(&mut store, N).expect("f64 call");
            for i in 0..N {
                ref64 += f64::from(i);
            }
        }
        assert_eq!(got64.to_bits(), ref64.to_bits(), "f64 global accumulator");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f64 global loop",
        );
        let g64 = instance.get_global(&store, "g").expect("f64 global export");
        assert_eq!(
            g64.get(&store).f64().map(|v| f64::from(v).to_bits()),
            Some(ref64.to_bits())
        );

        // --- f32 global ---
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, &f32_wat).expect("f32 module");
        let instance = Instance::new(&mut store, &module, &[]).expect("f32 instance");
        let f = instance
            .get_typed_func::<i32, f32>(&store, "run")
            .expect("f32 run");
        let mut ref32 = 0f32;
        let mut got32 = 0f32;
        for _ in 0..CALLS {
            got32 = f.call(&mut store, N).expect("f32 call");
            for i in 0..N {
                ref32 += i as f32;
            }
        }
        assert_eq!(got32.to_bits(), ref32.to_bits(), "f32 global accumulator");
        assert!(
            KERNEL_COMPILES.load(Ordering::Relaxed) >= 1,
            "the JIT tier must have compiled the f32 global loop",
        );
        let g32 = instance.get_global(&store, "g").expect("f32 global export");
        assert_eq!(
            g32.get(&store).f32().map(|v| f32::from(v).to_bits()),
            Some(ref32.to_bits())
        );
    }

    /// A function outside the supported subset (a `mul`-by-self square, which
    /// wasmi lowers to the `I32Mul_Rr` accumulator-square op the prepass does not
    /// yet handle) is rejected by the prepass and falls back to the stock
    /// executor, still yielding the correct result — the splice does not disturb
    /// ineligible functions.
    #[test]
    fn end_to_end_ineligible_falls_back() {
        use crate::{Engine, Instance, Module, Store};

        const SQUARE_WAT: &str = r#"
            (module
                (func (export "sq") (param $x i32) (result i32)
                    (i32.mul (local.get $x) (local.get $x))))
        "#;
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, SQUARE_WAT).expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let func = instance
            .get_typed_func::<i32, i32>(&store, "sq")
            .expect("typed func");
        assert_eq!(func.call(&mut store, 7).expect("call"), 49);
        assert_eq!(func.call(&mut store, -3).expect("call"), 9);
    }

    fn bench_mode() -> &'static str {
        if std::env::var_os("WASMI_NO_MAJIT").is_some() {
            "STOCK"
        } else {
            "MAJIT"
        }
    }

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

    /// Regime 1: steady-state hot loop, per-call overhead amortized over a huge
    /// iteration count. Reports MIN ns/iter over reps (robust to throttling).
    /// Run: `--release -- --ignored --nocapture bench_steady_state`
    /// (`WASMI_NO_MAJIT=1` bypasses the tier for the stock baseline).
    #[test]
    #[ignore]
    fn bench_steady_state() {
        use crate::{Engine, Instance, Module, Store};
        use std::time::Instant;
        let mode = bench_mode();
        const REPS: usize = 9;
        const N: i64 = 100_000_000;

        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let cm = Module::new(&engine, COUNTER_WAT).unwrap();
        let ci = Instance::new(&mut store, &cm, &[]).unwrap();
        let cf = ci
            .get_typed_func::<i32, i32>(&store, "count-via-locals")
            .unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            let _ = cf.call(&mut store, N as i32).unwrap();
            best = best.min(t.elapsed().as_nanos() as f64 / N as f64);
        }
        eprintln!("[{mode}] count(n={N}) | min {best:6.3} ns/iter");

        let fm = Module::new(&engine, FIB_WAT).unwrap();
        let fi = Instance::new(&mut store, &fm, &[]).unwrap();
        let ff = fi
            .get_typed_func::<i64, i64>(&store, "fibonacci_iter")
            .unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            let _ = ff.call(&mut store, N).unwrap();
            best = best.min(t.elapsed().as_nanos() as f64 / N as f64);
        }
        eprintln!("[{mode}] fib  (n={N}) | min {best:6.3} ns/iter");
    }

    /// Regime 2: many short calls — per-call overhead NOT amortized. With the
    /// per-call-driver shape this PANICKED (recompile + thread spawn every call);
    /// the persistent driver must make it net-positive.
    /// Run: `--release -- --ignored --nocapture bench_call_overhead`
    #[test]
    #[ignore]
    fn bench_call_overhead() {
        use crate::{Engine, Instance, Module, Store};
        use std::time::Instant;
        let mode = bench_mode();
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, COUNTER_WAT).unwrap();
        let instance = Instance::new(&mut store, &module, &[]).unwrap();
        let f = instance
            .get_typed_func::<i32, i32>(&store, "count-via-locals")
            .unwrap();
        const CALLS: u32 = 2_000_000;
        const SMALL_N: i32 = 8;
        let _ = f.call(&mut store, 1).unwrap();
        let t = Instant::now();
        let mut acc = 0i64;
        for _ in 0..CALLS {
            acc += f.call(&mut store, SMALL_N).unwrap() as i64;
        }
        let dt = t.elapsed();
        assert_eq!(
            acc, 0,
            "every count(8) must return 0 (no silent corruption)"
        );
        eprintln!(
            "[{mode}] count(n={SMALL_N}) x{CALLS} -> {acc} | {:>9.3} ms | {:7.2} ns/call",
            dt.as_secs_f64() * 1e3,
            dt.as_nanos() as f64 / CALLS as f64
        );
    }

    /// The kernel actually traces + compiles the hot loop (not just interprets).
    #[test]
    fn kernel_compiles_hot_loop() {
        let _serial = serial_kernel_guard();
        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        KERNEL_GUARD_FAILS.store(0, Ordering::Relaxed);
        let mp = compile_counter();
        let slots = seed(1000, mp.num_slots);
        let result = run_kernel(&mp.words, &slots, 3, 0, 0);
        assert_eq!(result, 0);
        let compiles = KERNEL_COMPILES.load(Ordering::Relaxed);
        let deopts = KERNEL_GUARD_FAILS.load(Ordering::Relaxed);
        eprintln!("[kernel] count-via-locals(1000): compiles={compiles} guard_fails={deopts}");
        assert!(
            compiles >= 1,
            "kernel must compile the hot loop at least once"
        );
        assert!(
            deopts <= 5,
            "compiled trace should run the loop itself (≈1 deopt at exit), got {deopts}",
        );
    }

    /// The adaptive tier policy probes both tiers, then commits to whichever
    /// reported the lower per-call time.
    fn fresh_probe() -> TierPolicy {
        TierPolicy::Probe {
            jit_calls: 0,
            min_jit_ns: u64::MAX,
            stock_calls: 0,
            min_stock_ns: u64::MAX,
        }
    }

    #[test]
    fn tier_policy_commits_to_faster_tier() {
        let mut slow = fresh_probe();
        for _ in 0..PROBE_JIT_CALLS {
            assert!(matches!(slow.next_action(), TierAction::ProbeJit));
            slow.record_jit(1000);
        }
        for _ in 0..PROBE_STOCK_CALLS {
            assert!(matches!(slow.next_action(), TierAction::ProbeStock));
            slow.record_stock(50);
        }
        // Stock was faster (50 < 1000) → commit Stock, stably.
        assert!(matches!(slow.next_action(), TierAction::Stock));
        assert!(matches!(slow.next_action(), TierAction::Stock));

        let mut fast = fresh_probe();
        for _ in 0..PROBE_JIT_CALLS {
            assert!(matches!(fast.next_action(), TierAction::ProbeJit));
            fast.record_jit(50);
        }
        for _ in 0..PROBE_STOCK_CALLS {
            assert!(matches!(fast.next_action(), TierAction::ProbeStock));
            fast.record_stock(1000);
        }
        // JIT was faster (50 < 1000) → commit Jit, stably.
        assert!(matches!(fast.next_action(), TierAction::Jit));
        assert!(matches!(fast.next_action(), TierAction::Jit));
    }

    /// A function seen only a handful of times never finishes probing, so it
    /// keeps running on the JIT it started on (the giant single-call loop case).
    #[test]
    fn tier_policy_keeps_probing_rare_function() {
        let mut rare = fresh_probe();
        for _ in 0..3 {
            assert!(matches!(rare.next_action(), TierAction::ProbeJit));
            rare.record_jit(60_000_000);
        }
    }

    /// CALL_ASSEMBLER: a caller function with a loop calls a JIT-eligible
    /// callee via `call $double` (CallInternal) on every iteration. The
    /// caller is JIT-eligible (has a loop), so it runs on the kernel. The
    /// `call $double` instruction emits MINI_CALL_RESIDUAL, which invokes
    /// `call_runner_fn`. If the callee is also JIT-eligible AND has no
    /// yield/bail ops, `call_runner_fn` runs it on the CALLEE_DRIVER via
    /// `run_callee` (the CALL_ASSEMBLER path) instead of the stock executor.
    ///
    #[test]
    fn end_to_end_call_assembler_callee_jit() {
        let _serial = serial_kernel_guard();
        use crate::{Engine, Instance, Module, Store};

        KERNEL_COMPILES.store(0, Ordering::Relaxed);
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());
        // callee: double(x) = x * 2 via a trivial loop (one iteration,
        // makes the function JIT-eligible by having a back-edge).
        // caller: sum(n) = sum of double(i) for i=0..n-1 via a loop.
        // Both functions are JIT-eligible.
        let module = Module::new(
            &engine,
            r#"
            (module
                ;; double(x) = x + x, computed via a 1-iteration loop.
                ;; The loop guard uses i64.ne (slot, slot) which the prepass
                ;; handles natively (BranchI64Ne_Ss → MINI_BR_I64_NE_SS),
                ;; so $double is JIT-eligible with no yield/bail.
                (func $double (export "double") (param $x i64) (result i64)
                    (local $i i64) (local $acc i64) (local $one i64)
                    (local.set $one (i64.const 1))
                    (local.set $i (i64.const 0))
                    (local.set $acc (local.get $x))
                    (block $break
                        (loop $loop
                            (br_if $break (i64.eq (local.get $i) (local.get $one)))
                            (local.set $acc (i64.add (local.get $acc) (local.get $x)))
                            (local.set $i (i64.add (local.get $i) (local.get $one)))
                            (br $loop)
                        )
                    )
                    (local.get $acc)
                )
                (func (export "sum_doubled") (param $n i64) (result i64)
                    (local $i i64) (local $acc i64)
                    (local.set $i (i64.const 0))
                    (local.set $acc (i64.const 0))
                    (block $break
                        (loop $loop
                            (br_if $break (i64.ge_s (local.get $i) (local.get $n)))
                            (local.set $acc
                                (i64.add (local.get $acc)
                                    (call $double (local.get $i))))
                            (local.set $i (i64.add (local.get $i) (i64.const 1)))
                            (br $loop)
                        )
                    )
                    (local.get $acc)
                )
            )"#,
        )
        .expect("module");
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");

        // double(5) = 10
        let double = instance
            .get_typed_func::<i64, i64>(&store, "double")
            .expect("typed func double");
        assert_eq!(double.call(&mut store, 5).expect("double"), 10);

        // sum_doubled(n) = 2*(0+1+...+(n-1)) = n*(n-1)
        let sum_doubled = instance
            .get_typed_func::<i64, i64>(&store, "sum_doubled")
            .expect("typed func sum_doubled");
        for n in [1i64, 5, 10, 100] {
            let expected = n * (n - 1);
            let got = sum_doubled.call(&mut store, n).expect("call");
            assert_eq!(got, expected, "sum_doubled({n}) must be {expected}");
        }
    }
}
