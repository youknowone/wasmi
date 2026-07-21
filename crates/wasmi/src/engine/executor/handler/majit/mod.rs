//! Proof-of-concept meta-tracing JIT tier for wasmi, built on the in-repo
//! `majit` framework (a Rust port of RPython's tracing JIT).
//!
//! majit does not attach to wasmi's existing handler-threaded executor (that
//! loop uses raw pointers and function-pointer dispatch, outside the restricted
//! Rust subset majit can meta-trace). Instead, an eligible wasm function's
//! dispatch loop is replaced by a small mainloop authored in the traceable
//! subset, over a flattened `i64`-word program. See the plan for the full
//! design.
//!
//! ## M1 — smoke check ([`smoke`])
//!
//! A self-contained register-machine mainloop cloned from
//! `majit/examples/tinyframe`. It de-risks, before any wasmi-IR work:
//!   * the `majit` crates build as cross-workspace path deps of `crates/wasmi`,
//!   * the cranelift backend links here,
//!   * a hot loop actually traces + compiles (observed via `set_on_compile_loop`).
//!
//! (The wasm MiniProgram uses an `[i64]`-word `env`; that the `#[jit_interp]`
//! macro accepts `[i64]` rather than only `[u8]` is proven by the `i64env`
//! majit example. `smoke` itself keeps tinyframe's `[u8]` stream.)
//!
//! ## M2 — prepass ([`prepass`])
//!
//! [`prepass::prepass`] decodes an eligible function's `indirect-dispatch` op
//! stream into a flat `i64`-word [`prepass::MiniProgram`]: it maps a small op
//! subset, builds the byte-offset → word-index table that rewrites
//! branch-relative targets, detects the back-edge (negative `BranchOffset`) as
//! the loop merge point, and models wasmi's implicit `Reg<i64>` accumulator as a
//! scalar `ireg`. Any op outside the subset yields `None` (the function is
//! ineligible and the caller falls back to the stock executor).
//!
//! ## M3 — execute with fallback ([`kernel`], [`super::func`])
//!
//! [`kernel`] is the majit-traced mainloop that interprets a MiniProgram: its
//! reds are the frame slots (`[int; virt]`) and the scalar `ireg`; `greens =
//! [pc, program]`. i32 wrap-around is expressed with i64 shifts (`(x << 32) >>
//! 32`) because the tracer aborts on `as i32` / `wrapping_add`.
//!
//! `init_wasm_func_call` runs the prepass once and stores the resulting
//! `Option<MiniProgram>` on the call. `execute_until_done` (the only hot-path
//! edit) routes an eligible call to [`kernel::run_kernel`] — seeding the cell
//! array from the frame, running the loop, writing the single result back to
//! slot 0 — and otherwise runs the stock executor unchanged.
//!
//! ## M4 — a richer real `.wasm`
//!
//! The op subset is extended to cover `fibonacci_iter` (a pure i64 loop: three
//! locals, slot/const/accumulator copies, an `i64.add`, a forward `block`-break
//! branch and a loop back-edge). i64 arithmetic is a plain `+` (which traces and
//! wraps mod 2^64 in release like wasm); only i32 needs the shift trick. The
//! function runs end-to-end on the JIT tier and matches the stock `fib(n)`.

pub mod kernel;
pub mod prepass;
pub mod smoke;

/// Whether the majit JIT tier is enabled at runtime. Reads `WASMI_NO_MAJIT`
/// once: set it to bypass the tier (eligible functions run on the stock
/// executor) for apples-to-apples benchmarking from a single binary.
pub(crate) fn majit_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WASMI_NO_MAJIT").is_none())
}

/// Whether the cross-instance loop-yield tier entry is enabled. Default off:
/// this path runs a callee's loop in the kernel until its tail-call, then
/// resumes the stock executor at that offset. Across the bench suite it never
/// yields a real win (per-bench deltas sit within run-to-run noise) but it
/// regresses a tail-call loop (`int_loop`) ~18%. That ~18% is not extra work:
/// the retired-instruction count is identical with the tier on or off (~159.7M
/// both), so the loss is purely microarchitectural — resuming the stock
/// executor mid-function at the tail-call offset has worse code locality and
/// branch prediction than a fresh call entry. There is no mechanical lever to
/// recover it. The plain-call tier (`run_persistent`) is unaffected. Opt in
/// with `WASMI_MAJIT_LOOP_YIELD` to evaluate the path on a workload that might
/// benefit.
pub(crate) fn loop_yield_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WASMI_MAJIT_LOOP_YIELD").is_some())
}
