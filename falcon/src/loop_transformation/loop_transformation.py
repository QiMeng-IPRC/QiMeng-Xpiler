import re

from falcon.client import invoke_llm
from falcon.simplification import simplify_code
from falcon.smt.const_inline import constant_inline
from falcon.src.loop_transformation.decorate_pragma import (
    SPLIT_PRAGMA_DEMO,
    SPLIT_PRAGMA_PROMPT,
)
from falcon.src.loop_transformation.pass_prompt import (
    LOOP_FUSION_DEMO,
    LOOP_FUSION_PROMPT,
    LOOP_REORDER_DEMO,
    LOOP_REORDER_PROMPT,
    LOOP_SPLIT_DEMO,
    LOOP_SPLIT_PROMPT,
    STMT_SPLIT_PROMPT,
    TENSOR_CONTRACTION,
    TENSOR_CONTRACTION_DEMO,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT
from falcon.stmt_simplification import ast_stmt_simplification
from falcon.util import make_full_func


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
        return simplify_code(code_content)
    return None


def run_loop_reorder(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {LOOP_REORDER_PROMPT}
    {LOOP_REORDER_DEMO}
    Please return the output kernel function without any additional information.
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{LOOP_REORDER_PROMPT}", LOOP_REORDER_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_REORDER_DEMO}", LOOP_REORDER_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


def run_split_annotation(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {SPLIT_PRAGMA_PROMPT}
    {SPLIT_PRAGMA_DEMO}
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{SPLIT_PRAGMA_PROMPT}", SPLIT_PRAGMA_PROMPT)
    PROMPT = PROMPT.replace("{SPLIT_PRAGMA_DEMO}", SPLIT_PRAGMA_DEMO)
    PROMPT = PROMPT.replace("{code}", code)

    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


def run_apply_split(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {LOOP_SPLIT_PROMPT}
    {LOOP_SPLIT_DEMO}
    Please return the output kernel function without any additional information.
    """
    if "#pragma loop_split" not in code:
        return code
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{LOOP_SPLIT_PROMPT}", LOOP_SPLIT_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_SPLIT_DEMO}", LOOP_SPLIT_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        code_content = constant_inline(code_content)
        code_content = ast_stmt_simplification(code_content)
        return code_content
    return None


def run_loop_contraction(code, target=None):
    code = simplify_code(code)
    PROMPT = """
    {SYSTEM_PROMPT}
    {TENSOR_CONTRACTION}
    {TENSOR_CONTRACTION_DEMO}
    Please return the output kernel function without any additional information.
    Input code: {code}
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{TENSOR_CONTRACTION}", TENSOR_CONTRACTION)
    PROMPT = PROMPT.replace(
        "{TENSOR_CONTRACTION_DEMO}", TENSOR_CONTRACTION_DEMO
    )
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        code_content = constant_inline(code_content)
        code_content = ast_stmt_simplification(code_content)
        code_content = make_full_func(code_content, target)
        return code_content
    return None


def run_stmt_split(code):
    code = simplify_code(code)
    PROMPT = """
    {SYSTEM_PROMPT}
    {STMT_SPLIT_PROMPT}

    Please return the output kernel function without any additional information.
    Input code: {code}
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{STMT_SPLIT_PROMPT}", STMT_SPLIT_PROMPT)
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        code_content = constant_inline(code_content)
        code_content = make_full_func(code_content)
        return code_content
    return None
