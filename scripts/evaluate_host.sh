#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
build_dir="${repo_root}/linux/build/evaluator-host"

make -C "${repo_root}/linux" \
    BUILD_DIR=build/evaluator-host \
    CFLAGS='-O2 -fPIE -Wall -Wextra -Werror' \
    LDFLAGS='-pie' \
    -j2

cc -O2 -Wall -Wextra -Werror \
    -I"${repo_root}/linux" \
    -I"${repo_root}/linux/compat" \
    "${repo_root}/linux/tests/test_prompt_source.c" \
    "${repo_root}/linux/prompt_source.c" \
    -o "${build_dir}/test_prompt_source"
"${build_dir}/test_prompt_source"

cc -O2 -Wall -Wextra -Werror \
    -I"${repo_root}/linux" \
    -I"${repo_root}/linux/compat" \
    -I"${repo_root}/sw/baremetal" \
    "${repo_root}/linux/tests/test_lm_head_topk.c" \
    "${repo_root}/sw/baremetal/bitnet_kernels.c" \
    -lm -pthread \
    -o "${build_dir}/test_lm_head_topk"
"${build_dir}/test_lm_head_topk"

printf '%s\n' 'FAST_BITNET_HOST_EVALUATOR_PASS'
