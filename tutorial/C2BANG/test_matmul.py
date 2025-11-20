from falcon.smt.loop_transformation.loop_fusion import ast_loop_fusion
from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.loop_transformation.loop_split import ast_loop_split
from falcon.smt.tensorization.detensorization import ast_detensorization
from falcon.smt.thread_binding import ast_thread_binding
from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_fusion,
    run_split_annotation,
)
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic,
    run_cache_process,
    run_code_decoration,
    run_tensorization,
    run_thread_binding,
)
from falcon.unit_test import unit_test


def run_transcompile_code(file_name, source, target):
    with open(file_name, "r") as f:
        device_code = f.read()
        f.close()

    device_code = device_code.split("extern")[0]
    # preprocess
    code = run_loop_recovery(device_code, target)
    if not unit_test(file_name, device_code):
        code = ast_loop_recovery(device_code, source)

    print("[INFO]*********loop recovery: ", code)

    modi_code = run_detensorization(code, source)
    if not unit_test(file_name, modi_code):
        modi_code = ast_detensorization(code, source)

    print("[INFO]***********detensorization: ", modi_code)
    # loop transformation
    fusion_code = run_loop_fusion(modi_code)
    if not unit_test(file_name, fusion_code):
        fusion_code = ast_loop_fusion(modi_code)

    print("[INFO]***********fusion: ", fusion_code)
    code = run_split_annotation(fusion_code)
    print("[INFO]***********split annotate: ", code)
    split_code = run_apply_split(code)
    print("[INFO]***********split: ", code)
    if not unit_test(file_name, split_code):
        split_code = ast_loop_split(code)
    print("[INFO]***********split: ", split_code)
    # postprocessing
    final_code = run_thread_binding(split_code, target)
    if not unit_test(file_name, final_code)[0]:
        final_code = ast_thread_binding(split_code, target)

    code = run_code_decoration(final_code)
    print("[INFO] decorate code: ", code)
    op_pragma = {}
    if target == "mlu":
        op_pragma = json.load(
            open(
                "./falcon/documents/operation_bang_C_instruction_map.json", "r"
            )
        )
    code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    cache_code = run_cache_process(code, space_maps, target)

    if not unit_test(file_name, cache_code):
        cache_code = ast_auto_cache(code, space_maps)
    print("[INFO] cache code: ", cache_code)
    code = run_code_decoration(cache_code)
    print("[INFO] tensor_decorate code: ", code)
    final_code = run_tensorization(code, target)
    if not unit_test(file_name, final_code)[0]:
        final_code = ast_tensorization(code, space_maps)
    return final_code


if __name__ == "__main__":
    file_name = "benchmark/data/cuda_code_test/gemm_32_128_128.cu"
    code = run_transcompile_code(file_name, source="cuda", target="mlu")
    print(code)
