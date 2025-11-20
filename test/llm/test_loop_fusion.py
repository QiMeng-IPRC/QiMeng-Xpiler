import os
import re

from falcon.src.loop_transformation.pass_prompt import (
    LOOP_FUSION_DEMO,
    LOOP_FUSION_PROMPT,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-4-turbo"""
api_key = os.getenv("OPENAI_API_KEY")


def run_loop_fusion(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {LOOP_FUSION_PROMPT}
    {LOOP_FUSION_DEMO}
    Please return the output kernel function without any additional information.
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{LOOP_FUSION_PROMPT}", LOOP_FUSION_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_FUSION_DEMO}", LOOP_FUSION_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


if __name__ == "__main__":
    code = """
    extern "C" void add_kernel(float* output, float* input1, float* input2) {
        int dim1 = 4;
        int dim2 = 4;
        int dim3 = 4;
        int dim4 = 64;

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    for (int l = 0; l < dim4; l++) {
                        int index = i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l;
                        output[index] = input1[index] + input2[index];
                    }
                }
            }
        }
    }
    """
    output_code = run_loop_fusion(code)
    print(output_code)

    code = """
    extern "C" void  add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 1024; j++) {
                    T_add[((i * 1024) + j)] = (A[((i * 1024) + j)] + B[((i * 1024) + j)]);
                }
            }
        }
    }
    """
    code = run_loop_fusion(code)
    print(code)
