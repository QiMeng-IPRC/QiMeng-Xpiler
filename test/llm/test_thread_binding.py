import os
import re

from falcon.src.post_processing.post_processing_prompt import (
    THREAD_BINDING_DEMO_BANG,
    THREAD_BINDING_DEMO_CUDA,
    THREAD_BINDING_PROMPT_BANG,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-4-turbo"""
api_key = os.getenv("OPENAI_API_KEY")


def run_thread_binding(code, target):
    PROMPT = """
    {SYSTEM_PROMPT}

    {THREAD_BINDING_PROMPT}

    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    prompt_demo = None
    THREAD_BINDING_PROMPT = None
    if target == "cuda" or target == "hip":
        prompt_demo = THREAD_BINDING_DEMO_CUDA
        THREAD_BINDING_PROMPT = THREAD_BINDING_DEMO_CUDA
    elif target == "mlu":
        prompt_demo = THREAD_BINDING_DEMO_BANG
        THREAD_BINDING_PROMPT = THREAD_BINDING_PROMPT_BANG

    PROMPT = PROMPT.replace("{THREAD_BINDING_PROMPT}", THREAD_BINDING_PROMPT)
    PROMPT = PROMPT.replace("{THREAD_BINDING_DEMO}", prompt_demo)
    PROMPT = PROMPT.replace("{cpp_code}", code)
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
    output_code = run_thread_binding(code, "mlu")
    print(output_code)

    code = """
    extern "C" void  add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 1024; j++) {
                if (((i * 1024) + j) < 2309) {
                    T_add[((i * 1024) + j)] = (A[((i * 1024) + j)] + B[((i * 1024) + j)]);
                }
            }
        }
    }
    """
    code = run_thread_binding(code, target="cuda")
    print(code)
