from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.tensorization.detensorization import ast_detensorization
from falcon.src.pre_processing.preprocessing import (
    run_detensorization,
    run_loop_recovery,
)
from falcon.unit_test import unit_test


def falcon_preprocess_pipeline(file_name, target):
    with open(file_name, "r") as f:
        org_code = f.read()
        f.close()

    code = run_loop_recovery(org_code, target)
    print("[INFO]******code: ", code)
    if not unit_test(file_name, code):
        code = ast_loop_recovery(org_code, target)

    if target in ["mlu"]:
        modi_code = run_detensorization(code, target)
        if not unit_test(file_name, modi_code):
            modi_code = ast_detensorization(code, target)

    return modi_code


if __name__ == "__main__":
    bang_file_name = "benchmark/data/mlu_code_test/sign_45_25.mlu"
    code = falcon_preprocess_pipeline(bang_file_name, target="mlu")
    print(code)
    print("===================================")
    cuda_file_name = "benchmark/data/cuda_code_test/add_3_3_256.cu"
    code = falcon_preprocess_pipeline(cuda_file_name, target="cuda")
    print(code)
