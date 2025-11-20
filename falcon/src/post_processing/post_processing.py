import json
import re

from falcon.client import invoke_llm
from falcon.src.post_processing.post_processing_prompt import (
    CACHE_READ_DEMO,
    CACHE_READ_PROMPT,
    CACHE_WRITE_DEMO,
    CACHE_WRITE_PROMPT,
    DECORATION_PROMPT,
    DOUBLE_BUFFER_DEMO,
    DOUBLE_BUFFER_PROMPT,
    TENSORIZATION_PROMPT,
    THREAD_BINDING_DEMO_BANG,
    THREAD_BINDING_DEMO_CUDA,
    THREAD_BINDING_PROMPT_BANG,
    THREAD_BINDING_PROMPT_CUDA,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT
from falcon.util import make_full_func


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
        THREAD_BINDING_PROMPT = THREAD_BINDING_PROMPT_CUDA
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
        return make_full_func(code_content, target)
    return None


def get_operation_content(code):
    # Define the regex pattern to match the content inside parentheses
    # following #pragma intrinsic
    pattern = r"#pragma\s+operation\(([^)]+)\)"
    # Find all matches in the given code (there might be multiple pragmas)
    matches = re.findall(pattern, code)
    return matches


def get_input_operand(pragma):
    inputs = pragma.split("input[")[1].split("]")[0]
    input_list = inputs.split(", ")
    return input_list


def get_output_operand(pragma):
    outputs = pragma.split("output[")[1].split("]")[0]
    output_list = outputs.split(", ")
    return output_list


def replace_operation_with_intrinsic(code, op_pragma):
    if not op_pragma:
        return code, None
    # Get the list of operations from the code
    op_list = get_operation_content(code)
    space_maps = []
    # Iterate over each operation found in the code
    for op in op_list:
        op_name = op.split("(")[0]
        pragma_pattern = re.escape(f"#pragma operation({op})")
        if op_name not in op_pragma:
            raise KeyError(f"Operation '{op_name}' not found in op_pragma.")

        # Handle input/output existence for op and op_pragma[op_name]
        if "input[" in op:
            input_operands = get_input_operand(op)
        else:
            input_operands = []
        if "output[" in op:
            output_operands = get_output_operand(op)
        else:
            output_operands = []

        pragma_val = op_pragma[op_name]
        if "input[" in pragma_val:
            input_spaces = get_input_operand(pragma_val)
        else:
            input_spaces = []
        if "output[" in pragma_val:
            output_spaces = get_output_operand(pragma_val)
        else:
            output_spaces = []

        if len(input_operands) != len(input_spaces):
            raise ValueError(
                f"Input operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(input_operands)} operands vs {len(input_spaces)} spaces)."
            )
        if len(output_operands) != len(output_spaces):
            raise ValueError(
                f"Output operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(output_operands)} operands vs {len(output_spaces)} spaces)."
            )
        input_map = {operand: space for operand, space in zip(input_operands, input_spaces)}
        output_map = {operand: space for operand, space in zip(output_operands, output_spaces)}
        space_map = {"input": input_map, "output": output_map}
        code = re.sub(pragma_pattern, pragma_val, code)
        space_maps.append(space_map)
    return code, space_maps


def get_intrinsic_content(code):
    # Define the regex pattern to match the content inside parentheses
    # following #pragma intrinsic
    pattern = r"#pragma\s+intrinsic\(([^)]+)\)"
    # Find all matches in the given code (there might be multiple pragmas)
    matches = re.findall(pattern, code)
    return matches


def get_input_memory_spaces(pragma):
    inputs = pragma.split("input[")[1].split("]")[0]
    input_list = inputs.split(", ")
    return input_list


def get_output_memory_spaces(pragma):
    outputs = pragma.split("output[")[1].split("]")[0]
    output_list = outputs.split(", ")
    return output_list


def generate_cache_read_prompt(buffer, space, code):
    PROMPT = """
    {SYSTEM_PROMPT}

    {CACHE_READ_PROMPT}

    {CACHE_READ_DEMO}
    Please return the output kernel function without any additional information.
    """
    space_map = {"nram": "__nram__", "wram": "__wram__"}
    NAMESPACE = space_map[space.lower()]

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_READ_PROMPT}", CACHE_READ_PROMPT)
    PROMPT = PROMPT.replace("{buffer}", buffer)
    PROMPT = PROMPT.replace("{CACHE_READ_DEMO}", CACHE_READ_DEMO)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{CODE}", code)
    PROMPT = PROMPT.replace("{NAMESPACE}", NAMESPACE)
    return PROMPT


def generate_cache_write_prompt(buffer, space, code):
    assert space, "memory space cannot be empty"
    PROMPT = """
    {SYSTEM_PROMPT}
    {CACHE_WRITE_PROMPT}
    {CACHE_WRITE_DEMO}
    Please return the output kernel function without any additional information.
    """
    NAMESPACE = "__nram__"
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_WRITE_PROMPT}", CACHE_WRITE_PROMPT)
    PROMPT = PROMPT.replace("{buffer}", buffer)
    PROMPT = PROMPT.replace("{CACHE_WRITE_DEMO}", CACHE_WRITE_DEMO)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{CODE}", code)
    PROMPT = PROMPT.replace("{NAMESPACE}", NAMESPACE)
    return PROMPT


def run_cache_process(code, space_maps, target):
    # Get the list of intrinsics from the code
    intrinsic_list = get_intrinsic_content(code)
    # Ensure the intrinsic lists and spaces have matching lengths
    if len(intrinsic_list) != len(space_maps):
        raise ValueError(
            f"intrinsics and memory spaces length mismatch for operation"
            f"({len(intrinsic_list)} intrinsics vs {len(space_maps)} spaces)."
        )
    # Iterate over each intrinsic found in the code
    for op, space_map in zip(intrinsic_list, space_maps):
        for key, value in space_map["input"].items():
            cache_read_prompt = generate_cache_read_prompt(key, value, code)
            content = invoke_llm(cache_read_prompt)
            match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
            code = match.group(1) if match else code
        for key, value in space_map["output"].items():
            cache_write_prompt = generate_cache_write_prompt(key, value, code)
            content = invoke_llm(cache_write_prompt)
            match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
            code = match.group(1) if match else code
    return make_full_func(code, target)


def tensorization(op, code, document):
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of Tensorization: {TENSORIZATION_PROMPT}
    Please tensorize the sequential code of {op} below the #pragma operation in {code}
    accordingt to the introduction of tensorized intrinsic.
    {document}
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", TENSORIZATION_PROMPT)
    PROMPT = PROMPT.replace("{document}", document)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{op}", op)

    content = invoke_llm(PROMPT)
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return make_full_func(code_content)
    return None


def get_operation_words(pragma_line):
    # Modify the pattern to capture everything inside parentheses, allowing
    # spaces and underscores
    pattern = r"#pragma\s+operation\((\w+)"
    # Find all matches in the given string
    matches = re.findall(pattern, pragma_line)
    # Return the list of matches (it will be empty if no matches are found)
    return matches


def run_tensorization(code, target):
    op_list = get_operation_words(code)
    if target == "mlu":
        op_dict = json.load(open("./falcon/documents/bang_c_op_map.json", "r"))
        for op in op_list:
            if "memory" in op:
                op_document = """
                __memcpy(void *dst, const void *src, unsigned int size, mluMemcpyDirection_t dir)
                Copies <size> bytes data from source address <src> to destination address <dst>. The copy direction is specified by <dir>.

                Parameters
                [out] dst: The address of destination area.

                [in] src: The address of source area.

                [in] size: The number of bytes to be copied.

                [in] dir: The copy direction.

                Usage Examples 1:
                // before:
                #pragma operation(memory(input[output_nram], output[output]))
                for (int i = 0; i < 512; ++i) {
                    for (int j = 0; j < 512; ++j) {
                        output[i * 512 + j] = output_nram[i * 512 + j];
                    }
                }

                // after:
                __memcpy(output, output_nram, 512 * 512 * 4, NRAM2GDRAM);

                Usage Examples 2:
                __nram__ float output_nram[512 * 512];
                // before:
                #pragma operation(memory(input[output], output[output_nram]))
                for (int i = 0; i < 512; ++i) {
                    for (int j = 0; j < 512; ++j) {
                        output_nram[i * 512 + j] = output[i * 512 + j];
                    }
                }

                // after:
                __memcpy(output_nram, output, 512 * 512 * 4, GDRAM2NRAM);

                Usage Examples 3:
                __nram__ half output_nram[512 * 512];
                // before:
                #pragma operation(memory(input[output], output[output_nram]))
                for (int i = 0; i < 512; ++i) {
                    for (int j = 0; j < 512; ++j) {
                        output_nram[i * 512 + j] = output[i * 512 + j];
                    }
                }

                // after:
                __memcpy(output_nram, output, 512 * 512 * 2, GDRAM2NRAM);

                 Usage Examples 4:
                // before:
                #pragma operation(memory(input[output], output[output_wram]))
                for (int i = 0; i < 512; ++i) {
                    for (int j = 0; j < 512; ++j) {
                        output_wram[i * 512 + j] = output[i * 512 + j];
                    }
                }

                // after:
                __memcpy(output_wram, output, 512 * 512 * 4, GDRAM2WRAM);

                """
            else:
                op_document = op_dict[op]
            code = tensorization(op, code, op_document)
    elif target in ["cuda", "hip"]:
        if "matmul" not in op_list:
            return code
    return code


def run_code_decoration(code):
    PROMPT = DECORATION_PROMPT.replace("{cpp_code}", code)
    content = invoke_llm(PROMPT)

    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


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


def post_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, cuda) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.

    :return: Transformed code after applying the two transformations."""
    code = run_thread_binding(code, target)

    # when target is "mlu" or "DLBOOST", insert tensorization process.
    if target in ["mlu", "DLBOOST"]:
        code = run_code_decoration(code)
        op_pragma = {}
        if target == "mlu":
            op_pragma = json.load(
                open(
                    "./falcon/documents/operation_bang_C_instruction_map.json",
                    "r",
                )
            )
        code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
        code = run_cache_process(code, space_maps, target)
        code = run_code_decoration(code)
        code = run_tensorization(code, target)
    return code


if __name__ == "__main__":
    code = """
     extern "C" __mlu_global__ void add(float *input1, float *input2, float *output)
    {
    __nram__ float input1_Nram[256];
    __nram__ float input2_Nram[256];
    __nram__ float output_Nram[256];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
        #pragma operation(memory(input[input1], output[input1_Nram]))
        for (int k = 0; k < 256; k++)
        {
            int index = (((i * 3) * 256) + (j * 256)) + k;
            input1_Nram[k] = input1[index];
        }

        #pragma operation(memory(input[input2], output[input2_Nram]))
        for (int k = 0; k < 256; k++)
        {
            int index = (((i * 3) * 256) + (j * 256)) + k;
            input2_Nram[k] = input2[index];
        }

        #pragma operation(add(input[input1_Nram, input2_Nram], output[output_Nram]))
        for (int k = 0; k < 256; k++)
        {
            output_Nram[k] = input1_Nram[k] + input2_Nram[k]; // Use both cached versions
        }

        #pragma operation(memory(input[output_Nram], output[output]))
        for (int k = 0; k < 256; k++)
        {
            int index = (((i * 3) * 256) + (j * 256)) + k;
            output[index] = output_Nram[k]; // Write cached result to original buffer
        }
        }
    }
    }
    """
    target = "mlu"
    code = run_tensorization(code, target)
    print("[INFO]***********code: ", code)
