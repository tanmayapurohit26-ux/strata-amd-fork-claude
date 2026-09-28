#!/bin/bash
# amd/check.sh - AMD port: build the engine and its test programs for this PC's AMD GPU, then run the tests.
#
#   bash amd/check.sh                                  build + the GPU tests that need no model files
#   bash amd/check.sh /path/to/...-IQ3_XXS-00001-of-00002.gguf [/path/to/...-IQ3_S-00001-of-00002.gguf]
#                                                      the same, plus the expert test on each model file given
#
# Every test's full output goes to amd/logs/<test>.log; the summary at the end says which passed.  Nothing here
# changes your system: it only writes inside this folder (build-hip-check/ and amd/logs/).
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
ROCM=${ROCM_PATH:-/opt/rocm}
BUILD=$ROOT/build-hip-check
LOGS=$ROOT/amd/logs
mkdir -p "$LOGS"

say() { printf '%s\n' "$*"; }

# ---- 1. tools
CMAKE=$(command -v cmake || true)
[ -x "$ROOT/.venv/bin/cmake" ] && CMAKE=$ROOT/.venv/bin/cmake
NINJA=$(command -v ninja || true)
[ -x "$ROOT/.venv/bin/ninja" ] && NINJA=$ROOT/.venv/bin/ninja
if [ -z "$CMAKE" ] || [ -z "$NINJA" ]; then
  say "cmake and ninja are needed: sudo apt install cmake ninja-build  (a one-time install), then run this again"
  exit 1
fi
if [ ! -x "$ROCM/llvm/bin/clang++" ]; then
  say "ROCm's compiler was not found at $ROCM/llvm/bin/clang++ (set ROCM_PATH if ROCm lives elsewhere)"
  exit 1
fi
ARCH=$("$ROCM/bin/rocminfo" 2>/dev/null | grep -oE '^\s*Name:\s+gfx1[0-9a-f]+' | head -1 | grep -oE 'gfx[0-9a-f]+')
ARCH=${ARCH:-gfx1100}
say "GPU architecture: $ARCH"

# the pinned llama.cpp (setup.py downloads it to third_party/llama.cpp); without it CMake clones it with git
GGML_ARG=()
[ -f "$ROOT/third_party/llama.cpp/ggml/CMakeLists.txt" ] && GGML_ARG=("-DSTRATA_GGML_DIR=$ROOT/third_party/llama.cpp")

# ---- 2. build
say "Building the engine and the test programs (the first time takes 10-30 minutes) ..."
"$CMAKE" -G Ninja "-DCMAKE_MAKE_PROGRAM=$NINJA" -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON "-DCMAKE_HIP_ARCHITECTURES=$ARCH" "-DCMAKE_HIP_COMPILER=$ROCM/llvm/bin/clang++" \
  "-DCMAKE_PREFIX_PATH=$ROCM" "${GGML_ARG[@]}" > "$LOGS/build-configure.log" 2>&1 \
  || { say "CONFIGURE FAILED - see amd/logs/build-configure.log"; exit 1; }
"$CMAKE" --build "$BUILD" -j "$(( $(nproc) / 2 > 2 ? $(nproc) / 2 : 2 ))" > "$LOGS/build.log" 2>&1 \
  || { say "BUILD FAILED - see amd/logs/build.log (the first 'error:' line is the one that matters)"; exit 1; }
say "Build OK."

# ---- 3. tests
PASS=(); FAIL=()
run() {   # run <name> <command...>
  local name=$1; shift
  if (cd "$ROOT" && timeout 600 "$@") > "$LOGS/$name.log" 2>&1; then PASS+=("$name"); say "  pass  $name"
  else FAIL+=("$name"); say "  FAIL  $name   (amd/logs/$name.log)"; fi
}
say "Running the GPU tests (no model files needed) ..."
run device_selftest        "$BUILD/strata-device" --selftest
run native_qsa_score_test  "$BUILD/native_qsa_score_test"
for t in dequant_s2_parity s2_gemv_parity s2_gemv_q8_parity s_gemv_parity s_gemv_q8k_parity bf16_gemv_parity \
         shared_expert_parity gr_parity gdn_parity sampler_parity rope_parity quantize_act_parity \
         router_top10_parity elementwise_parity qsa_parity; do
  run "$t" "$BUILD/$t" --selftest
done
for t in kv_q8_parity kv_stream_parity kv_q4_parity cvec_parity platform_memory_test; do
  run "$t" "$BUILD/$t"
done
run ple_reader_test "$BUILD/ple_reader_test" --selftest

# the model's own experts: GPU vs CPU vs ggml's reference, on real rows of each file given (6 layers)
for g in "$@"; do
  name=native_expert_parity_$(basename "$g" .gguf)
  run "$name" "$BUILD/native_expert_parity" "$g"
done

say ""
say "Passed: ${#PASS[@]}   Failed: ${#FAIL[@]}"
[ ${#FAIL[@]} -gt 0 ] && say "Failed: ${FAIL[*]}"
say "Logs: amd/logs/  (send the whole folder back if anything failed)"
[ ${#FAIL[@]} -eq 0 ]
