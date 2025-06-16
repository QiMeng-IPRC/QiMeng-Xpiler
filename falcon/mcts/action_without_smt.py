import json
import random

from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split, run_loop_contraction, run_loop_fusion, run_loop_reorder,
    run_split_annotation, run_stmt_split)
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic, run_cache_process, run_code_decoration,
    run_double_buffer, run_tensorization, run_thread_binding)
from falcon.src.pre_processing.preprocessing import (run_detensorization,
                                                     run_loop_recovery)


def loop_recovery(file_name, code, source_platform, target_platform):
    final_code = run_loop_recovery(code, source_platform)
    return final_code


def stmt_split(file_name, code, source_platform, target_platform):
    return run_stmt_split(code)


def detensorization(file_name, code, source_platform, target_platform):
    final_code = run_detensorization(code, source_platform)
    return final_code


def loop_fusion(file_name, code, source_platform, target_platform):
    final_code = run_loop_fusion(code)
    return final_code


def loop_reorder(file_name, code, source_platform, target_platform):
    final_code = run_loop_reorder(code)
    return final_code


def loop_split(file_name, code, source_platform, target_platform):
    code = run_split_annotation(code)
    final_code = run_apply_split(code)
    return final_code


def loop_contraction(file_name, code, source_platform, target_platform):
    final_code = run_loop_contraction(code, None)
    return final_code


def auto_bind(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu", "cuda", "hip"]:
        return code
    final_code = run_thread_binding(code, target_platform)
    return final_code


def auto_cache(file_name, code, source_platform, target_platform):
    code = run_code_decoration(code)
    op_pragma = {}
    if target_platform == "mlu":
        op_pragma = json.load(
            open(
                "./falcon/documents/operation_bang_C_instruction_map.json", "r"
            )
        )
    code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    # If no need to cache, just return origin code
    if space_maps is None:
        return code

    cache_code = run_cache_process(code, space_maps, target_platform)
    return cache_code


def auto_tensorization(file_name, code, source_platform, target_platform):
    code = run_code_decoration(code)
    final_code = run_tensorization(code, target_platform)
    return final_code


def auto_pipeline(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu"]:
        return code
    final_code = run_double_buffer(code, target_platform)
    return final_code


actions = [
    loop_recovery,
    stmt_split,
    detensorization,
    loop_fusion,
    loop_reorder,
    loop_split,
    loop_contraction,
    auto_bind,
    auto_cache,
    auto_tensorization,
    auto_pipeline,
]

if __name__ == "__main__":
    code = """
    __global__ void __launch_bounds__(1024)
    add(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
                (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] +
                B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    source_platform = "cuda"
    target_platform = "mlu"
    file_name = "benchmark/data/cuda_code_test/add_18_128.cu"
    selected_function = random.choice(actions)
    # Invoke the randomly selected function.
    result = selected_function(
        file_name, code, source_platform, target_platform
    )

    # Output result
    print("运行结果:", result)
