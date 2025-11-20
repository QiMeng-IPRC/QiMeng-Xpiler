import os
import re

from falcon.client import invoke_llm


def run_transcompile(code):
    PROMPT = """Please rewrite the following CUDA C code to utilize VNNI (Vector Neural Network Instructions) for optimal performance. Ensure the code leverages VNNI for matrix computations and neural network operations.

    Requirements:

    1. Replace GPU acceleration logic with VNNI-compatible SIMD instructions;
    2. Optimize the code to fully exploit VNNI's capabilities in integer operations and tensor processing;
    3. Add appropriate comments explaining the VNNI-specific optimizations and logic;
    4. Retain the original functionality and input-output interfaces of the code.
    Original CUDA C Code:
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
    files = glob.glob("benchmark/data/cuda_code_test/*.cu")
    for file in tqdm(files):
        base_name = os.path.basename(file)
        with open(file, "r") as f:
            source = f.read()
            f.close()

        target_code = run_transcompile(source)
        file_name = os.path.join(
            "benchmark/transcompilation/cuda/dlboost", base_name
        )
        with open(file_name, mode="w") as f:
            f.write(target_code)
            f.close()
