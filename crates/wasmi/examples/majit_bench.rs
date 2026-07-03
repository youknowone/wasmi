//! Micro-benchmark harness for the `majit-jit` meta-tracing tier.
//!
//! Builds a hot-loop `.wat` workload, warms it up so the tier compiles the
//! inner loop, then times a batch of calls and prints throughput. The tier is
//! toggled per process by the `WASMI_NO_MAJIT` env var (read once by the crate),
//! so a wrapper script runs this twice — with and without the var — to compare
//! the JIT tier against the stock register interpreter from one binary.
//!
//! Usage: `majit_bench <workload> <n> <calls>`
//!   workload : fib | isum | sumsq   (default: fib)
//!   n        : inner-loop trip count        (default: 2000)
//!   calls    : timed outer call count       (default: 200000)

use std::time::Instant;
use wasmi::{Engine, Instance, Module, Store};

/// Each workload is a single exported function taking one integer and running an
/// `n`-iteration loop whose body uses only tier-eligible ops. Results wrap mod
/// 2^bits exactly like wasm, so large `n` is fine (release build; debug would
/// abort on overflow by design).
fn workload_wat(name: &str) -> (&'static str, &'static str, bool) {
    // (export name, wat, is_i64)
    match name {
        // Iterative Fibonacci: pure i64 loop, three locals, add/sub, ne-exit.
        "fib" => (
            "fib",
            r#"
            (module
              (func (export "fib") (param $n i64) (result i64)
                (local $a i64) (local $b i64) (local $i i64)
                (local.set $a (i64.const 0))
                (local.set $b (i64.const 1))
                (local.set $i (local.get $n))
                (block $break
                  (br_if $break (i64.eqz (local.get $i)))
                  (loop $cont
                    (i64.add (local.get $a) (local.get $b))
                    (local.set $a (local.get $b))
                    (local.set $b)
                    (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                    (br_if $cont (i64.ne (local.get $i) (i64.const 0)))))
                (local.get $a)))
            "#,
            true,
        ),
        // Sum of 1..=n as i64: countdown loop, ne-exit back-edge, accumulate.
        "isum" => (
            "isum",
            r#"
            (module
              (func (export "isum") (param $n i64) (result i64)
                (local $i i64) (local $acc i64)
                (local.set $i (local.get $n))
                (block $break
                  (br_if $break (i64.eqz (local.get $i)))
                  (loop $cont
                    (local.set $acc (i64.add (local.get $acc) (local.get $i)))
                    (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                    (br_if $cont (i64.ne (local.get $i) (i64.const 0)))))
                (local.get $acc)))
            "#,
            true,
        ),
        // Sum of i*i for i in 0..n as i32: i32 multiply + unsigned exit compare.
        "sumsq" => (
            "sumsq",
            r#"
            (module
              (func (export "sumsq") (param $n i32) (result i32)
                (local $i i32) (local $acc i32)
                (block $break
                  (loop $cont
                    (br_if $break (i32.ge_u (local.get $i) (local.get $n)))
                    (local.set $acc
                      (i32.add (local.get $acc)
                        (i32.mul (local.get $i) (local.get $i))))
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (br $cont)))
                (local.get $acc)))
            "#,
            false,
        ),
        other => panic!("unknown workload {other:?}; pick fib|isum|sumsq"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let workload = args.next().unwrap_or_else(|| "fib".into());
    let n: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let calls: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200_000);

    let (export, wat, is_i64) = workload_wat(&workload);
    let jit = std::env::var_os("WASMI_NO_MAJIT").is_none();

    let engine = Engine::default();
    let module = Module::new(&engine, wat).expect("module");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instance");

    // Warm up past the tier's hot-count threshold so the timed region runs
    // compiled code (when the tier is enabled).
    let warmup = 50u64;
    let result = if is_i64 {
        let func = instance
            .get_typed_func::<i64, i64>(&store, export)
            .expect("typed func");
        for _ in 0..warmup {
            func.call(&mut store, n as i64).expect("warmup");
        }
        let start = Instant::now();
        let mut last = 0i64;
        for _ in 0..calls {
            last = func.call(&mut store, n as i64).expect("call");
        }
        report(&workload, jit, n, calls, start.elapsed());
        last as i128
    } else {
        let func = instance
            .get_typed_func::<i32, i32>(&store, export)
            .expect("typed func");
        for _ in 0..warmup {
            func.call(&mut store, n as i32).expect("warmup");
        }
        let start = Instant::now();
        let mut last = 0i32;
        for _ in 0..calls {
            last = func.call(&mut store, n as i32).expect("call");
        }
        report(&workload, jit, n, calls, start.elapsed());
        last as i128
    };
    // Emit the result so JIT-vs-stock runs can be diffed for correctness.
    println!("result={result}");
}

fn report(workload: &str, jit: bool, n: u64, calls: u64, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64();
    let total_iters = (n * calls) as f64;
    let ns_per_iter = elapsed.as_nanos() as f64 / total_iters;
    let mode = if jit { "majit" } else { "stock" };
    println!(
        "workload={workload} mode={mode} n={n} calls={calls} \
         elapsed_ms={:.1} ns_per_loop_iter={:.3} Miter_per_s={:.1}",
        secs * 1e3,
        ns_per_iter,
        total_iters / secs / 1e6,
    );
}
