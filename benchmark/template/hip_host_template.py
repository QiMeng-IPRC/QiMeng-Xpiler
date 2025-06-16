import re
from string import Template


def infer_grid_dim_from_kernel(kernel_code: str, thread_num: int) -> str:
    # Derive the dimensions used by blockIdx and threadIdx.
    use_block_x = bool(re.search(r"\bblockIdx\.x\b", kernel_code))
    use_block_y = bool(re.search(r"\bblockIdx\.y\b", kernel_code))
    use_block_z = bool(re.search(r"\bblockIdx\.z\b", kernel_code))

    use_thread_x = bool(re.search(r"\bthreadIdx\.x\b", kernel_code))
    use_thread_y = bool(re.search(r"\bthreadIdx\.y\b", kernel_code))
    use_thread_z = bool(re.search(r"\bthreadIdx\.z\b", kernel_code))

    # Set numBlocks
    numBlocks_x = 256 if use_block_x else 1
    numBlocks_y = 256 if use_block_y else 1
    numBlocks_z = 256 if use_block_z else 1
    numblocks_define = (
        f"dim3 numBlocks({numBlocks_x}, {numBlocks_y}, {numBlocks_z});"
    )

    # Set the blockSize.
    blockSize_x = thread_num if use_thread_x else 1
    blockSize_y = 1024 if use_thread_y else 1
    blockSize_z = 1024 if use_thread_z else 1
    blocksize_define = (
        f"dim3 blockSize({blockSize_x}, {blockSize_y}, {blockSize_z});"
    )
    return numblocks_define, blocksize_define


def create_hip_func(file_name, op_type="ewise"):
    with open(file_name, "r") as f:
        original_function = f.read()

    # Regular expression to extract function signature
    function_signature_pattern = r"__global__ void\s*(?:__launch_bounds__\((\d+)(?:,\s*\d+)?\))?\s*(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find HIP kernel signature.")

    thread_num = match.group(1)
    kernel_name = match.group(2)
    param_list_str = match.group(3)

    if thread_num is None:
        thread_num = "1024"  # Default value

    params = [param.strip() for param in param_list_str.split(",")]
    params = [var.replace("__restrict__ ", "").strip() for var in params]
    params = [var.replace("const ", "").strip() for var in params]
    param_list = ", ".join(params)
    param_names = [
        param.split()[-1].replace("*", "").replace("__restrict__", "").strip()
        for param in params
    ]

    device_vars = [f"{name}_hip" for name in param_names]

    # Memory allocation and copy operations
    device_memory_alloc = []
    memcpy = []
    size = None
    if op_type == "ewise":
        size = "size"
        for param in params:
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_hip;\n")
            device_memory_alloc.append(
                f"hipMalloc((void**)&{name}_hip, {size} * sizeof(float));\n"
            )

        for param in params[:-1]:
            name = param.split("*")[1]
            memcpy.append(
                f"hipMemcpy({name}_hip, {name}, {size} * sizeof(float), hipMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"hipMemcpy({name}, {name}_hip, {size} * sizeof(float), hipMemcpyDeviceToHost);\n"
    elif op_type == "pool":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_hip;\n")
            device_memory_alloc.append(
                f"hipMalloc((void**)&{name}_hip, {size[i]} * sizeof(float));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            memcpy.append(
                f"hipMemcpy({name}_hip, {name}, {size[i]} * sizeof(float), hipMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"hipMemcpy({name}, {name}_hip, {size[-1]} * sizeof(float), hipMemcpyDeviceToHost);\n"
    elif op_type == "matmul":
        size = ["size1", "size2", "size3"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            device_memory_alloc.append(param + "_hip;\n")
            device_memory_alloc.append(
                f"hipMalloc((void**)&{name}_hip, {size[i]} * sizeof({dtype}));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"hipMemcpy({name}_hip, {name}, {size[i]} * sizeof({dtype}), hipMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"hipMemcpy({name}, {name}_hip, size3 * sizeof({dtype}), hipMemcpyDeviceToHost);\n"

    elif op_type == "layer_norm":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_hip;\n")
            if i == 1 or i == 2:
                device_memory_alloc.append(
                    f"hipMalloc((void**)&{name}_hip, size2 * sizeof(float));\n"
                )
            else:
                device_memory_alloc.append(
                    f"hipMalloc((void**)&{name}_hip, size1 * sizeof(float));\n"
                )
        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            if i == 1 or i == 2:
                memcpy.append(
                    f"hipMemcpy({name}_hip, {name}, size2 * sizeof(float), hipMemcpyHostToDevice);\n"
                )
            else:
                memcpy.append(
                    f"hipMemcpy({name}_hip, {name}, size1 * sizeof(float), hipMemcpyHostToDevice);\n"
                )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"hipMemcpy({name}, {name}_hip, size1 * sizeof(float), hipMemcpyDeviceToHost);\n"
    elif op_type == "deformable":
        size = ["size1", "size2", "size3", "size4", "size5"]
        print("[INFO]params: ", params)
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            device_memory_alloc.append(param + "_hip;\n")
            device_memory_alloc.append(
                f"hipMalloc((void**)&{name}_hip, {size[i]} * sizeof({dtype}));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"hipMemcpy({name}_hip, {name}, {size[i]} * sizeof({dtype}), hipMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"hipMemcpy({name}, {name}_hip, size3 * sizeof({dtype}), hipMemcpyDeviceToHost);\n"

    # Infer grid dimensions from kernel code
    original_function = original_function.replace('extern "C"', "")
    numblocks_define, blocksize_define = infer_grid_dim_from_kernel(
        original_function
    )
    if isinstance(size, list):
        size_list = ", ".join(
            arg for arg in ["int " + string for string in size]
        )
    else:
        size_list = "int size"
    # Create host function template
    host_func_template = Template(
        """
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <hip_runtime.h>
#include <hip.h>
#include <cmath>
#include <mma.h>
#include <hip_fp16.h>
#include <stdlib.h>

using namespace nvhip;

// Original Kernel
${original_function}

extern "C" void ${kernel_name}_kernel(${param_list}, ${size_list}) {
    ${memcpy_alloc_list}
    ${memcpy_htod}

    ${blockSize_define}
    ${numblocks_define}
    ${kernel_name}<<<numBlocks, blockSize>>>(${called_param_list});

    ${memcpy_dtoh}
    ${hip_free}
}
"""
    )
    memcpy_alloc_list = "    ".join(alloc for alloc in device_memory_alloc)
    new_code = host_func_template.substitute(
        kernel_name=kernel_name,
        original_function=original_function.strip(),
        param_list=param_list,
        memcpy_htod="\n    ".join(memcpy),
        thread_num=thread_num,
        numblocks_define=numblocks_define,
        blocksize_define=blocksize_define,
        called_param_list=", ".join(device_vars),
        memcpy_dtoh=memcpy_back,
        hip_free="\n    ".join(
            [f"hipFree({dev}_hip);" for dev in param_names]
        ),
        size_list=size_list,
        memcpy_alloc_list=memcpy_alloc_list,
    )

    output_file = file_name.replace(".hip", "_bak.hip")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file


if __name__ == "__main__":
    create_hip_perf_func("benchmark/data/hip_code_test/sign_5_128.hip")
