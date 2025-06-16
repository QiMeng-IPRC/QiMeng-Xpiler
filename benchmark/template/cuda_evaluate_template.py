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

    # Set the blockSize
    blockSize_x = thread_num if use_thread_x else 1
    blockSize_y = 1024 if use_thread_y else 1
    blockSize_z = 1024 if use_thread_z else 1
    blocksize_define = (
        f"dim3 blockSize({blockSize_x}, {blockSize_y}, {blockSize_z});"
    )
    return numblocks_define, blocksize_define


def create_cuda_perf_func(file_name, op_type="ewise"):
    with open(file_name, "r") as f:
        original_function = f.read()

    # Regular expression to extract function signature
    function_signature_pattern = r"__global__ void\s*(?:__launch_bounds__\((\d+)(?:,\s*\d+)?\))?\s*(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find CUDA kernel signature.")

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

    device_vars = [f"{name}_cuda" for name in param_names]

    # Memory allocation and copy operations
    device_memory_alloc = []
    memcpy = []
    size = None
    if op_type == "ewise":
        size = "size"
        for param in params:
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size} * sizeof(float));\n"
            )

        for param in params[:-1]:
            name = param.split("*")[1]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size} * sizeof(float), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, {size} * sizeof(float), cudaMemcpyDeviceToHost);\n"
    elif op_type == "pool":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof(float));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof(float), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, {size[-1]} * sizeof(float), cudaMemcpyDeviceToHost);\n"
    elif op_type == "matmul":
        size = ["size1", "size2", "size3"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof({dtype}));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof({dtype}), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size3 * sizeof({dtype}), cudaMemcpyDeviceToHost);\n"

    elif op_type == "layer_norm":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            if i == 1 or i == 2:
                device_memory_alloc.append(
                    f"cudaMalloc((void**)&{name}_cuda, size2 * sizeof(float));\n"
                )
            else:
                device_memory_alloc.append(
                    f"cudaMalloc((void**)&{name}_cuda, size1 * sizeof(float));\n"
                )
        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            if i == 1 or i == 2:
                memcpy.append(
                    f"cudaMemcpy({name}_cuda, {name}, size2 * sizeof(float), cudaMemcpyHostToDevice);\n"
                )
            else:
                memcpy.append(
                    f"cudaMemcpy({name}_cuda, {name}, size1 * sizeof(float), cudaMemcpyHostToDevice);\n"
                )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size1 * sizeof(float), cudaMemcpyDeviceToHost);\n"
    elif op_type == "deformable":
        size = ["size1", "size2", "size3", "size4", "size5"]
        print("[INFO]params: ", params)
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof({dtype}));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof({dtype}), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size3 * sizeof({dtype}), cudaMemcpyDeviceToHost);\n"

    original_function = original_function.replace('extern "C"', "")
    # Infer grid dimensions from kernel code
    numblocks_define, blocksize_define = infer_grid_dim_from_kernel(
        original_function, thread_num
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
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdlib.h>

using namespace nvcuda;

// Original Kernel
${original_function}

extern "C" float timed_${kernel_name}_kernel(${param_list}, ${size_list}) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    ${memcpy_alloc_list}
    ${memcpy_htod}

    ${blockSize_define}
    ${numblocks_define}
    float timeTotal = 0;
    // Warmup
    for (int i = 0; i < 10; ++i) {
        ${kernel_name}<<<numBlocks, blockSize>>>(${called_param_list});
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        ${kernel_name}<<<numBlocks, blockSize>>>(${called_param_list});
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&timeTotal, start, end);
    float ms_time = timeTotal / 100;
    printf("Total Time: %.3f ms\\n", ms_time);

    ${memcpy_dtoh}
    ${cuda_free}

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms_time;
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
        blockSize_define=blocksize_define,
        called_param_list=", ".join(device_vars),
        memcpy_dtoh=memcpy_back,
        cuda_free="\n    ".join(
            [f"cudaFree({dev}_cuda);" for dev in param_names]
        ),
        size_list=size_list,
        memcpy_alloc_list=memcpy_alloc_list,
    )

    output_file = file_name.replace(".cu", "_bak.cu")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file


if __name__ == "__main__":
    create_cuda_perf_func("benchmark/data/cuda_code_test/sign_5_128.cu")
