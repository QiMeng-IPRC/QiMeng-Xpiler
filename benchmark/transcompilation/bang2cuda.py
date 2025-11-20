import os
import re

from falcon.client import invoke_llm


def run_transcompile(code):
    PROMPT = """Please rewrite the following BANG-optimized code into CUDA C code for GPU acceleration.
    Ensure the converted code retains the same functionality and is optimized for CUDA's parallel processing capabilities.

    Requirements:

    1. Replace BANG-specific logic with CUDA kernels and GPU-compatible operations;
    2. Optimize the code for efficient memory usage and parallel execution on CUDA-enabled GPUs;
    3. Provide comments explaining how the CUDA implementation corresponds to the original BANG functionality;
    4. Maintain the input-output interfaces and preserve the original computational intent.
    Original BANG Code:
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
    files = glob.glob("benchmark/data/bang_code_test/*.cpp")
    for file in tqdm(files):
        base_name = os.path.basename(file)
        with open(file, "r") as f:
            source = f.read()
            f.close()

        target_code = run_transcompile(source)
        file_name = os.path.join(
            "benchmark/transcompilation/bang/cuda", base_name
        )
        with open(file_name, mode="w") as f:
            f.write(target_code)
            f.close()
