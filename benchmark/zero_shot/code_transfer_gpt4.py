import argparse
import os
import re
import sys

from benchmark.zero_shot.zero_shot_prompt import (
    CPU_TO_CUDA_PROMPT,
    CPU_TO_HIP_PROMPT,
    CPU_TO_MLU_PROMPT,
    CUDA_TO_AMD_PROMPT,
    CUDA_TO_CPU_PROMPT,
    CUDA_TO_MLU_PROMPT,
    HIP_TO_CPU_PROMPT,
    HIP_TO_CUDA_PROMPT,
    HIP_TO_MLU_PROMPT,
    MLU_TO_CPU_PROMPT,
    MLU_TO_CUDA_PROMPT,
    MLU_TO_HIP_PROMPT,
)
from falcon.client import invoke_llm

ext_map = {
    "cuda": "cu",
    "hip": "hip",
    "cpu": "cpp",
    "mlu": "mlu",
}


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.stderr.write(
        "Error: please set the OPENAI_API_KEY environment variable\n"
    )
    sys.exit(1)
openai.api_key = api_key

# Map (source_platform, dest_platform) to prompt templates
PROMPT_MAP = {
    ("cuda", "cpu"): CUDA_TO_CPU_PROMPT,
    ("cuda", "hip"): CUDA_TO_AMD_PROMPT,
    ("cuda", "mlu"): CUDA_TO_MLU_PROMPT,
    ("hip", "cpu"): HIP_TO_CPU_PROMPT,
    ("hip", "cuda"): HIP_TO_CUDA_PROMPT,
    ("hip", "mlu"): HIP_TO_MLU_PROMPT,
    ("cpu", "cuda"): CPU_TO_CUDA_PROMPT,
    ("cpu", "hip"): CPU_TO_HIP_PROMPT,
    ("cpu", "mlu"): CPU_TO_MLU_PROMPT,
    ("mlu", "cpu"): MLU_TO_CPU_PROMPT,
    ("mlu", "cuda"): MLU_TO_CUDA_PROMPT,
    ("mlu", "hip"): MLU_TO_HIP_PROMPT,
}


def code_transform(input_code: str, prompt_template: str) -> str:
    """Call the GPT-4 API to perform code translation."""
    prompt = prompt_template.format(input_code=input_code)
    output = invoke_llm(prompt)
    match = re.search(r"```cpp(.*?)```", output, re.DOTALL)
    if not match:
        print("No C++ code block found.")
        return False

    cpp_code = match.group(1).strip()
    return cpp_code


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot code translation between CUDA/CPU/hip/MLU"
    )
    parser.add_argument(
        "source_file", help="Path to the source code file to translate"
    )
    parser.add_argument(
        "source_platform",
        choices=["cuda", "hip", "cpu", "mlu"],
        help="Source platform (cuda | hip | cpu | mlu)",
    )
    parser.add_argument(
        "dest_platform",
        choices=["cuda", "hip", "cpu", "mlu"],
        help="Destination platform (cuda | hip | cpu | mlu)",
    )
    args = parser.parse_args()

    direction = (args.source_platform, args.dest_platform)
    if direction not in PROMPT_MAP:
        print(
            f"Error: unsupported translation direction {direction}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read the source code
    with open(args.source_file, "r", encoding="utf-8") as f:
        input_code = f.read()

    if args.source_platform in ["cuda", "hip"]:
        input_code = input_code.split("extern")[0]

    # Perform translation
    prompt_template = PROMPT_MAP[direction]
    translated_code = code_transform(input_code, prompt_template)

    # Write output to directory named "<source>_<dest>"
    output_dir = f"{args.source_platform}_{args.dest_platform}"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(args.source_file)
    output_path = os.path.join(output_dir, base_name)
    file_stem = os.path.splitext(os.path.basename(args.source_file))[0]
    target_ext = ext_map[args.dest_platform]
    output_path = os.path.join(output_dir, f"{file_stem}.{target_ext}")

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(translated_code)

    print(f"Translated code saved to: {output_path}")


if __name__ == "__main__":
    main()
