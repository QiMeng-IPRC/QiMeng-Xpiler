import json
import os
import re

from falcon.src.pre_processing.preprocessing_prompt import (
    DETENSORIZATION_PROMPT_BANG,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-4-turbo"""
api_key = os.getenv("OPENAI_API_KEY")


def detensorization(op, code, document):
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of Detensorization: {DETENSORIZATION_PROMPT_BANG}
    Please transform the instruction {op} in following code into sequential for loop.

    {code}

    accordingt to the description of tinstruction.

    {document}

    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace(
        "{DETENSORIZATION_PROMPT_BANG}", DETENSORIZATION_PROMPT_BANG
    )
    PROMPT = PROMPT.replace("{document}", document)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{op}", op)

    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


def extract_bang_instructions(code):
    # Define a regex pattern to match instructions starting with __bang
    pattern = r"__bang\w+"
    # Find all matches in the provided code
    instructions = re.findall(pattern, code)
    return instructions


def run_detensorization(code, target):
    op_dict = json.load(open("./falcon/documents/bang_c_user_guide.json", "r"))
    instructions = extract_bang_instructions(code)
    # First, detensorize memory
    code = detensorization("__memcpy", code, op_dict["__memcpy"])
    for inst in instructions:
        code = detensorization(inst, code, op_dict[inst])
    return code


if __name__ == "__main__":
    code = """
    extern "C" __mlu_global__ void tanh(float* input0, float* active_tanh_210) {
        __nram__ float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    code = run_detensorization(code, target="mlu")
    print(code)
