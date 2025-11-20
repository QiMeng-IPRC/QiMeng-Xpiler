from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.tensorization.detensorization import ast_detensorization


def falcon_preprocess_pipeline(file_name, target):
    with open(file_name, "r") as f:
        org_code = f.read()
        f.close()
    if target == "cuda":
        org_code = org_code.split("extern")[0]
    code = ast_loop_recovery(org_code, target)
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
