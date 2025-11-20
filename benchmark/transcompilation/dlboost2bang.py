import os
import re

from falcon.client import invoke_llm


def run_transcompile(code):
    PROMPT = """Please rewrite the following DLBoost-optimized code into BANG C code for NPU acceleration.
    Ensure the converted code retains the same functionality and is optimized for BANG's parallel processing capabilities.

    Requirements:

    1. Replace DLBoost-specific logic with BANG kernels and NPU-compatible operations;
    2. Optimize the code for efficient memory usage and parallel execution on BANG-enabled NPUs;
    3. Provide comments explaining how the BANG implementation corresponds to the original DLBoost functionality;
    4. Maintain the input-output interfaces and preserve the original computational intent.
    Original DLBoost Code:
    {code}
    """
    PROMPT = PROMPT.replace("{code}", code)
    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


if __name__ == "__main__":
    files = glob.glob("benchmark/data/dlboost_code_test/*.cpp")
    for file in tqdm(files):
        base_name = os.path.basename(file)
        with open(file, "r") as f:
            source = f.read()
            f.close()

        target_code = run_transcompile(source)
        file_name = os.path.join(
            "benchmark/transcompilation/dlboost/bang", base_name
        )
        with open(file_name, mode="w") as f:
            f.write(target_code)
            f.close()
