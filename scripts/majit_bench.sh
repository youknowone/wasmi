#!/usr/bin/env bash
# Re-runnable benchmark for the `majit-jit` meta-tracing JIT tier.
#
# Builds the `majit_bench` example once (release), then runs each workload twice
# from the same binary — once with the tier live, once with `WASMI_NO_MAJIT=1`
# forcing the stock register interpreter — and prints a speedup table. The tier
# reads `WASMI_NO_MAJIT` once per process, so on/off must be separate runs.
#
# Usage: scripts/majit_bench.sh [n] [calls]
#   n     : inner-loop trip count per call   (default: 5000000)
#   calls : timed calls per workload         (default: 40)
#
# Only workloads whose hot loop the tier actually compiles are included; each run
# also asserts JIT and stock produce identical results.
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-5000000}"
CALLS="${2:-40}"
WORKLOADS=(fib isum sumsq)
BIN=target/release/examples/majit_bench

echo "building majit_bench (release)..."
if ! cargo build -q -p wasmi --features majit-jit,wat --release \
     --example majit_bench 2>build-majit_bench.log; then
  cat build-majit_bench.log >&2; exit 1
fi
rm -f build-majit_bench.log

# Pull `ns_per_loop_iter` and `result` out of one run.
field() { sed -n "s/.*$1=\([^ ]*\).*/\1/p"; }

printf '\n%-8s %14s %14s %9s   %s\n' workload majit_ns/it stock_ns/it speedup correct
printf '%.0s-' {1..64}; echo
for w in "${WORKLOADS[@]}"; do
  jit_out=$("$BIN" "$w" "$N" "$CALLS")
  stock_out=$(WASMI_NO_MAJIT=1 "$BIN" "$w" "$N" "$CALLS")
  jit_ns=$(echo "$jit_out"   | field ns_per_loop_iter)
  st_ns=$(echo  "$stock_out" | field ns_per_loop_iter)
  jit_res=$(echo "$jit_out"   | field result)
  st_res=$(echo  "$stock_out" | field result)
  speedup=$(awk "BEGIN{printf \"%.2f\", $st_ns/$jit_ns}")
  ok=$([ "$jit_res" = "$st_res" ] && echo yes || echo "NO($jit_res!=$st_res)")
  printf '%-8s %14s %14s %8sx   %s\n' "$w" "$jit_ns" "$st_ns" "$speedup" "$ok"
done
echo
echo "n=$N calls=$CALLS  (ns/it = nanoseconds per inner-loop iteration; lower is faster)"
