import json

from falcon.smt.auto_cache import ast_auto_cache
from falcon.smt.thread_binding import ast_thread_binding
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic,
    run_cache_process,
    run_code_decoration,
    run_tensorization,
)


def falcon_postprocess_pipeline(code, target):
    final_code = ast_thread_binding(code, target)
    print("[INFO] final_code: ", final_code)
    # when target is "mlu" or "DLBOOST", insert tensorization process.
    if target in ["mlu", "DLBOOST"]:
        code = run_code_decoration(final_code)
        print("[INFO] decorate code: ", code)
        op_pragma = {}
        if target == "mlu":
            op_pragma = json.load(
                open(
                    "./falcon/documents/operation_bang_C_instruction_map.json",
                    "r",
                )
            )
        code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
        cache_code = run_cache_process(code, space_maps, target)

        cache_code = ast_auto_cache(code, space_maps)
        print("[INFO] cache code: ", cache_code)
        code = run_code_decoration(cache_code)
        print("[INFO] tensor_decorate code: ", code)
        final_code = run_tensorization(code, target)
        final_code = ast_auto_cache(code, space_maps)
    return final_code


if __name__ == "__main__":
    code = """
    void add_kernel(float* output, float* input1, float* input2) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 128; k++) {
                    for (int l = 0; l < 128; l++) {
                        output[i * 4 * 128 * 128 + j * 128 * 128 + k * 128 + l] = input1[i * 4 * 128 * 128 + j * 128 * 128 + k * 128 + l] + input2[i * 4 * 128 * 128 + j * 128 * 128 + k * 128 + l];
                    }
                }
            }
        }
    }
    """
    code = falcon_postprocess_pipeline(code, target="mlu")
    print(code)
