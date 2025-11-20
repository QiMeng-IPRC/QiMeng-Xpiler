import csv
import ctypes
import glob
import os
import re

import numpy as np
import torch

from benchmark.utils import avgpool_np, maxpool_np, minpool_np
from benchmark.utils import run_hip_compilation as run_compilation
from benchmark.utils import sumpool_np


def create_hip_perf_func(hip_file):
    """读取HIP文件，插入事件以获取内核执行时间，并写回文件."""
    with open(hip_file, "r") as f:
        content = f.read()

    # The regular expression matches lines with kernel calls, such as add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);
    # TODO(michael): solve the case if the kernel change into another line
    # Match lines containing <<<...>>>
    kernel_call_pattern = r"(\w+<<<.*?>>>\(.*?\);)"

    # Find all matching rows.
    matches = re.findall(kernel_call_pattern, content)
    for match in matches:
        # Insert warm-up and event-related code.
        event_code = f"""
// Warm-up the kernel by running it 10 times
for (int i = 0; i < 10; i++) {{
    {match}
}}

// Timing the kernel execution
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);
hipEventRecord(start, 0);

for (int i = 0; i < 1000; i++) {{
    {match}
}}
hipEventRecord(stop, 0);
hipEventSynchronize(stop);
float milliseconds = 0;
hipEventElapsedTime(&milliseconds, start, stop);
milliseconds = milliseconds / 1000.0f;
printf("Kernel execution time: %f ms\\n", milliseconds);
hipEventDestroy(start);
hipEventDestroy(stop);
        """

        # Insert time calculation code after the kernel call.
        with open(
            os.path.join(os.getcwd(), "benchmark/macro/hip_macro.txt"), "r"
        ) as f:
            macro = f.read()
        content = content.replace('extern "C" void', 'extern "C" float')
        modified_kernel = macro + content.replace(match, event_code)

        # Insert "return milliseconds;" at the end of the main function.
        # Allow function parameters
        main_pattern = r'extern "C" float \w+\s*\(.*?\)\s*\{'
        main_match = re.search(
            main_pattern, modified_kernel, re.DOTALL
        )  # Use re.DOTALL

        if main_match:
            # Locate the end of the main function to replace the last `}`.
            last_brace_index = modified_kernel.rfind("}")
            if last_brace_index != -1:
                modified_kernel = (
                    modified_kernel[:last_brace_index]
                    + "return milliseconds;\n}"
                    + modified_kernel[last_brace_index + 1 :]
                )
        else:
            print("未找到主函数定义。请检查函数名和签名。")
            return

        # Write the modified content back to the original file.
        with open(hip_file.replace(".hip", "_bak.hip"), "w") as f:
            f.write(modified_kernel)


def perf_unary(name, shape, function, dtype="float32"):
    # Define the function's parameters and return types.

    function.restype = ctypes.c_float
    # Create the input array.
    input_array = np.random.uniform(size=shape).astype(dtype)

    # Create the output array.
    output_array = np.zeros_like(input_array)

    # Convert the input and output arrays to C pointer types.
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if name == "rmsnorm":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(input_ptr, output_ptr, *shape)
    elif name == "softmax":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            input_ptr, output_ptr, np.prod(shape[:-1]), shape[-1]
        )
    else:
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        # Calling a C function
        elapsed_time = function(input_ptr, output_ptr, np.prod(shape))
    return elapsed_time


def perf_layernorm(shape1, shape2, function, dtype="float32"):
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # Create the input array.
    input_array = np.random.uniform(size=shape1).astype(dtype)
    gamma = np.random.uniform(size=shape2).astype(dtype)
    beta = np.random.uniform(size=shape2).astype(dtype)
    # Create the output array.
    output_array = np.zeros_like(input_array)

    # Convert the input and output arrays into C pointer types.
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gamma_ptr = gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    beta_ptr = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Invoke the C function
    elapsed_time = function(
        input_ptr, gamma_ptr, beta_ptr, output_ptr, *shape1
    )
    return elapsed_time


def perf_binary(name, shape_A, shape_B, shape_C, function, dtype="float32"):
    A = np.random.rand(*shape_A).astype("float32")
    B = np.random.rand(*shape_B).astype("float32")
    shape = [*shape_A, shape_B[-1]]
    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    function.restype = ctypes.c_float
    # Call the function with the matrices and dimensions
    result_ctypes = np.zeros(shape_C, dtype=np.float32)
    output_ptr = result_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if name == "add":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        elapsed_time = function(A_ptr, B_ptr, output_ptr, np.prod(shape_A))
    elif name in ["gemm", "bmm"]:
        # Convert the matrices to contiguous memory for ctypes
        A = np.random.rand(*shape_A).astype("float16")
        B = np.random.rand(*shape_B).astype("float16")
        shape = [*shape_A, shape_B[-1]]
        # Convert the matrices to contiguous memory for ctypes
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            A_ptr,
            B_ptr,
            output_ptr,
            *shape,
        )
    elif name in ["conv1d"]:
        # Convert the matrices to contiguous memory for ctypes
        A = np.random.rand(*shape_A).astype("float32")
        B = np.random.rand(*shape_B).astype("float32")
        # Convert the matrices to contiguous memory for ctypes
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            A_ptr, B_ptr, output_ptr, shape_A[0], shape_A[0] - shape_B[0] + 1
        )
    elif name in ["gemv"]:
        # Convert the matrices to contiguous memory for ctypes
        A = np.random.rand(*shape_A).astype("float32")
        B = np.random.rand(*shape_B).astype("float32")
        # Convert the matrices to contiguous memory for ctypes
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            A_ptr, B_ptr, output_ptr, shape_A[0], shape_A[1]
        )
    return elapsed_time


def perf_conv2d(name, shape_A, shape_B, shape_C, stride, function):
    function.restype = ctypes.c_float
    data_np = torch.rand(shape_A)
    kernel_np = torch.rand(shape_B)
    function.restype = ctypes.c_float
    # Convert the matrices to contiguous memory for ctypes
    input_ptr = data_np.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel_np.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    result_ctypes = torch.zeros(shape_C)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    if name == "conv2d":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            input_ptr,
            kernel_ptr,
            output_ptr,
            shape_A[0],
            shape_A[1],
            shape_A[3],
            shape_B[0],
            shape_B[1],
            stride,
        )
    elif name == "conv2dnchw":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            input_ptr,
            kernel_ptr,
            output_ptr,
            shape_A[0],
            shape_A[2],
            shape_A[1],
            shape_B[0],
            shape_B[2],
            stride,
        )
    elif name == "depthwiseconv":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            input_ptr,
            kernel_ptr,
            output_ptr,
            *stride,
        )
    return elapsed_time


def perf_deformable(shape, function):
    N, M, D = shape[:3]
    Lq, L, P = shape[3:]
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
    )
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True
    )

    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = ctypes.c_float

    # Create the output array.
    output_array = np.zeros(
        (
            value.shape[0],
            sampling_locations.shape[1],
            value.shape[2] * value.shape[3],
        ),
        "float32",
    )

    # Convert the input and output arrays into C pointer types.
    value_ptr = value.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    shapes_ptr = (
        shapes.int().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    sampling_locations_ptr = sampling_locations.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    level_start_index_ptr = (
        level_start_index.int()
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    attention_weights_ptr = attention_weights.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Call a C function
    elapsed_time = function(
        value_ptr,
        shapes_ptr,
        level_start_index_ptr,
        sampling_locations_ptr,
        attention_weights_ptr,
        output_ptr,
    )
    return elapsed_time


def perf_pooling(name, shape, kernel, stride, function, dtype="float32"):
    input_array = torch.rand(*shape)
    # Calculate the result using numpy for comparison
    if name == "maxpool":
        output_np = maxpool_np(input_array, kernel + stride)
    elif name == "minpool":
        output_np = minpool_np(input_array, kernel + stride)
    elif name == "avgpool":
        output_np = avgpool_np(input_array, kernel + stride)
    elif name == "sumpool":
        output_np = sumpool_np(input_array, kernel + stride)
    output_array = torch.zeros(output_np.shape)
    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    np.prod(shape)
    np.prod(output_np.shape)
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # Call the function with the matrices and dimensions
    elapsed_time = function(
        input_ptr,
        output_ptr,
        shape[0],
        shape[3],
        shape[1],
        kernel[0],
        stride[0],
    )
    return elapsed_time


def perf_scaled_dot_product_attention(shape, function, dtype="float32"):
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # Create the input array.
    input_array_1 = np.random.uniform(size=shape).astype(dtype)
    input_array_2 = np.random.uniform(size=shape).astype(dtype)
    input_array_3 = np.random.uniform(size=shape).astype(dtype)
    # Create the output array.
    output_array = np.zeros_like(input_array_1)

    # Convert the input and output arrays into C pointer types.
    input_ptr_1 = input_array_1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_2 = input_array_2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_3 = input_array_3.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Invoke the C function.
    elapsed_time = function(
        input_ptr_1, input_ptr_2, input_ptr_3, output_ptr, *shape
    )
    return elapsed_time


def perf_pipeline(hip_file):
    create_hip_perf_func(hip_file)
    backup_hip_file = hip_file.replace(".hip", "_bak.hip")
    so_name = hip_file.replace(".hip", ".so")
    success, output = run_compilation(so_name, backup_hip_file)


def benchmark(hip_file):
    execution_time = 0
    base_name = os.path.basename(hip_file)
    name = base_name.split("_")[0]
    perf_pipeline(hip_file)
    lib = ctypes.CDLL(hip_file.replace(".hip", ".so"))
    function = getattr(lib, name + "_kernel")
    if name == "add":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_binary(name, shape, shape, shape, function)

    elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
        shape = base_name.split("_")[1:5]
        shape = [int(intg) for intg in shape]
        kernel_stride = base_name.split(".")[0].split("_")[5:]
        kernel_stride = [int(intg) for intg in kernel_stride]
        execution_time = perf_pooling(
            name, shape, kernel_stride[:2], kernel_stride[2:], function
        )

    elif name == "bmm":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
        shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
        shape_B = [batch_size, matrix_dim_k, matrix_dim_j]
        shape_C = [batch_size, matrix_dim_i, matrix_dim_k]
        execution_time = perf_binary(name, shape_A, shape_B, shape_C, function)

    elif name == "gemm":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        shape_A = [1, shape[0], shape[1]]
        shape_B = [1, shape[2], shape[1]]
        shape_C = [1, shape[0], shape[2]]
        execution_time = perf_binary(name, shape_A, shape_B, shape_C, function)

    elif name in ["sign", "relu", "sigmoid", "softmax", "rmsnorm", "gelu"]:
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_unary(name, shape, function)

    elif name == "conv2d":
        data_shape = base_name.split("_")[1:5]
        data_shape = [int(intg) for intg in data_shape]

        kernel_shape = base_name.split("_")[5:9]
        kernel_shape = [int(intg) for intg in kernel_shape]
        stride_h = stride_w = int(base_name.split("_")[9])
        pad_h = pad_w = int(base_name.split("_")[10].replace(".hip", ""))

        batch_size, input_height, input_width, input_channel = data_shape
        output_channel, kernel_height, kernel_width, _ = kernel_shape
        out_height = int(
            (input_height + np.sum(pad_h) - kernel_height) / stride_h + 1
        )
        out_width = int(
            (input_width + np.sum(pad_w) - kernel_width) / stride_w + 1
        )
        output_shape = [batch_size, out_height, out_width, output_channel]
        execution_time = perf_conv2d(
            name, data_shape, kernel_shape, output_shape, stride_h, function
        )

    elif name == "conv2dnchw":
        data_shape = base_name.split("_")[1:5]
        data_shape = [int(intg) for intg in data_shape]
        kernel_shape = base_name.split("_")[5:9]
        kernel_shape = [int(intg) for intg in kernel_shape]
        stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
        pad = int(base_name.split(".")[0].split("_")[10])
        batch_size, input_channel, input_height, input_width = data_shape
        output_channel, _, kernel_height, kernel_width = kernel_shape
        out_height = int(
            (input_height + np.sum(pad) - kernel_height) / stride_h + 1
        )
        out_width = int(
            (input_width + np.sum(pad) - kernel_width) / stride_w + 1
        )
        output_shape = [batch_size, output_channel, out_height, out_width]
        execution_time = perf_conv2d(
            name, data_shape, kernel_shape, output_shape, stride_h, function
        )

    elif name == "gemv":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        kernel_shape = [shape[1]]
        output_shape = [shape[0]]
        execution_time = perf_binary(
            name, shape, kernel_shape, output_shape, function
        )

    elif name == "conv1d":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        shape = [shape[1]]
        kernel_shape = [3]
        output_shape = [shape[0]]
        execution_time = perf_binary(
            name, shape, kernel_shape, output_shape, function
        )

    elif name == "depthwiseconv":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        input_height, kernel_size, input_channels = (
            shape[0],
            shape[1],
            shape[2],
        )
        data_shape = [input_height, input_height, input_channels]
        kernel_shape = [kernel_size, kernel_size, input_channels]
        # Calculate the output tensor shape
        output_height = input_height - kernel_size + 1
        output_width = input_height - kernel_size + 1
        output_shape = [output_height, output_width, input_channels]
        execution_time = perf_conv2d(
            name, data_shape, kernel_shape, output_shape, shape, function
        )

    elif name == "deformable":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_deformable(shape, function)

    elif name == "mha":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        perf_scaled_dot_product_attention(shape, function)

    elif name == "layernorm":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_layernorm(shape, [shape[-1]], function)

    else:
        print("Undefined file: ", hip_file)
        return 0

    os.remove(hip_file.replace(".hip", "_bak.hip"))
    os.remove(hip_file.replace(".hip", ".so"))
    return execution_time


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/hip_code_test/*.hip")
    )

    table = []
    times = []
    table.append(files)
    for file in files:
        execution_time = benchmark(file)
        times.append(execution_time)

    table.append(times)

    # Transpose data
    transposed_data = list(zip(*table))

    # Add a title row.
    header = ["file", "time(ms)"]
    transposed_data.insert(0, header)

    # Save as CSV file
    with open("benchmark/perf/hip_output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(transposed_data)
