import json

from falcon.smt.auto_cache import ast_auto_cache
from falcon.smt.const_inline import constant_inline
from falcon.smt.loop_transformation.loop_contraction import \
    ast_loop_contraction
from falcon.smt.loop_transformation.loop_fusion import ast_loop_fusion
from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.loop_transformation.loop_reorder import ast_loop_reorder
from falcon.smt.loop_transformation.loop_split import ast_loop_split
from falcon.smt.software_pipeline import smt_double_buffer
from falcon.smt.stmt_split import ast_stmt_split
from falcon.smt.tensorization.detensorization import ast_detensorization
from falcon.smt.tensorization.tensorization import ast_tensorization
from falcon.smt.thread_binding import ast_thread_binding
from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split, run_loop_contraction, run_loop_fusion, run_loop_reorder,
    run_split_annotation, run_stmt_split)
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic, run_cache_process, run_code_decoration,
    run_double_buffer, run_tensorization, run_thread_binding)
from falcon.src.pre_processing.preprocessing import (run_detensorization,
                                                     run_loop_recovery)
from falcon.unit_test import unit_test


def loop_recovery(file_name, code, source_platform, target_platform):
    try:
        final_code = run_loop_recovery(code, source_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop recovery error")
    except Exception:
        final_code = ast_loop_recovery(code, source_platform)
    return final_code


def stmt_split(file_name, code, source_platform, target_platform):
    try:
        final_code = run_stmt_split(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("stmt split error")
    except Exception:
        final_code = ast_stmt_split(code, source_platform)
    return final_code


def detensorization(file_name, code, source_platform, target_platform):
    try:
        final_code = run_detensorization(code, source_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("detensorization error")
    except Exception:
        final_code = ast_detensorization(code, source_platform)
    return final_code


def loop_fusion(file_name, code, source_platform, target_platform):
    try:
        final_code = run_loop_fusion(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop fusion error")
    except Exception:
        final_code = ast_loop_fusion(code)
    return final_code


def loop_reorder(file_name, code, source_platform, target_platform):
    try:
        final_code = run_loop_reorder(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop_reorder error")
    except Exception:
        final_code = ast_loop_reorder(code)
    return final_code


def loop_split(file_name, code, source_platform, target_platform):
    try:
        code = run_split_annotation(code)
        final_code = run_apply_split(code)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop_split error")
    except Exception:
        final_code = ast_loop_split(code)
    return final_code


def loop_contraction(file_name, code, source_platform, target_platform):
    try:
        final_code = run_loop_contraction(code, None)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("loop_contraction error")
    except Exception:
        final_code = ast_loop_contraction(code)
    return final_code


def auto_bind(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu", "cuda", "hip"]:
        return code
    try:
        final_code = run_thread_binding(code, target_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("auto_bind error")
    except Exception:
        final_code = ast_thread_binding(code, target_platform)
    return final_code


def auto_cache(file_name, code, source_platform, target_platform):
    code = constant_inline(code)
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
    try:
        cache_code = run_cache_process(code, space_maps, target_platform)
        if not unit_test(file_name, cache_code):
            raise RuntimeError("auto_cache error")
    except Exception:
        cache_code = ast_auto_cache(code, space_maps)
    return cache_code


def auto_tensorization(file_name, code, source_platform, target_platform):
    try:
        code = run_code_decoration(code)
        final_code = run_tensorization(code, target_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("auto_tensorization error")
    except Exception:
        final_code = ast_tensorization(code, target_platform)
    return final_code


def auto_pipeline(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu"]:
        return code
    try:
        final_code = run_double_buffer(code, target_platform)
        if not unit_test(file_name, final_code)[0]:
            raise RuntimeError("auto_pipeline error")
    except Exception:
        final_code = smt_double_buffer(code)
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
    file_name = "benchmark/data/cuda_code_test/add_4_4_4_64.cu"
    from falcon.mcts.utils import open_file

    code = open_file(file_name)
    code = code.split("extern")[0]
    for action_id in [0]:
        action = actions[action_id]
        code = action(
            file_name,
            code,
            "cuda",
            "cpu",
        )
        print("[INFO]****trans code: ", code)
    from falcon.util import get_target

    target, file_type = get_target(code)
    if target == "mlu":
        new_file = "./tmp/add_4_4_4_64.mlu"
    elif target == "cuda":
        new_file = "./tmp/add_4_4_4_64.cu"
    else:
        new_file = "./tmp/add_4_4_4_64.cpp"
    print("[INFO]**********code: ", code)
    with open(new_file, "w", encoding="utf-8") as f:
        f.write(code)
    from falcon.mcts.transcompile import objective

    score = objective(new_file, target)
    print(score)
