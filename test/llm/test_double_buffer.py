import os
import re

from falcon.src.post_processing.post_processing_prompt import (
    DOUBLE_BUFFER_DEMO,
    DOUBLE_BUFFER_PROMPT,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-4-turbo"""
api_key = os.getenv("OPENAI_API_KEY")


def double_buffer(code):
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of double buffer: {DOUBLE_BUFFER_PROMPT}
    Please optimize the code snippet below #pragma with double buffer pipeline.

    {code}


    accordingt to the introduction of double buffer.

    {DOUBLE_BUFFER_DEMO}
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{DOUBLE_BUFFER_PROMPT}", DOUBLE_BUFFER_PROMPT)
    PROMPT = PROMPT.replace("{DOUBLE_BUFFER_DEMO}", DOUBLE_BUFFER_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


def run_double_buffer(code, target):
    code = double_buffer(code)
    return code


if __name__ == "__main__":
    code = """
    __mlu_global__ void kernel(float* c, float* a, float b) {
        __nram__ float a_tmp[128];
        __nram__ float c_tmp[128];
        #pragma double_buffer
        for (int i = 0; i < 64; i++) {
            __memcpy(a_tmp, a + i * 128, 128 * sizeof(float), GDRAM2NRAM);
            __bang_add_scalar(c_tmp, a_tmp, b, 128);
            __memcpy(c + i * 128, c_tmp, 128 * sizeof(float), NRAM2GDRAM);
        }
    }
    """
    code = run_double_buffer(code, target="mlu")
    print(code)
