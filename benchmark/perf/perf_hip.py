import argparse
import ctypes
import os

import numpy as np
import torch

from benchmark.template.hip_evaluate_template import create_hip_perf_func
from benchmark.utils import avgpool_np, conv2d_nchw, maxpool_np, minpool_np
from benchmark.utils import run_hip_compilation as run_compilation
from benchmark.utils import sumpool_np


def perf_unary(shape, function, dtype="float32"):
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # Create the input array.
    input_array = np.random.uniform(size=shape).astype(dtype)

    # Create the output array.
    output_array = np.zeros_like(input_array)

    # Convert the input and output arrays into C pointer types.
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
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
    ]
    function.restype = ctypes.c_float
    # Create the input array.
    input_array = np.random.uniform(size=shape1).astype(dtype)
    gamma = np.random.uniform(size=shape2).astype(dtype)
    beta = np.random.uniform(size=shape2).astype(dtype)
    # Create the output array.
    output_array = np.zeros_like(input_array)

    # Convert the input and output arrays to C pointer types.
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gamma_ptr = gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    beta_ptr = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Invoke the C function
    elapsed_time = function(
        input_ptr,
        gamma_ptr,
        beta_ptr,
        output_ptr,
        np.prod(shape1),
        np.prod(shape2),
    )
    return elapsed_time


def perf_binary(name, shape_A, shape_B, shape_C, function, dtype="float32"):
    A = np.random.rand(*shape_A).astype("float32")
    B = np.random.rand(*shape_B).astype("float32")

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
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
    elif name in [
        "gemm",
        "gemv",
        "bmm",
        "conv2d",
        "conv1d",
        "depthwiseconv",
        "conv2dnchw",
    ]:
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            A_ptr,
            B_ptr,
            output_ptr,
            np.prod(shape_A),
            np.prod(shape_B),
            np.prod(shape_C),
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
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
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

    # Convert the input and output arrays to C pointer types.
    value_ptr = value.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    shapes_ptr = (
        shapes.int().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    sampling_locations_ptr = sampling_locations.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    attention_weights_ptr = attention_weights.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Invoke the C function
    elapsed_time = function(
        value_ptr,
        shapes_ptr,
        sampling_locations_ptr,
        attention_weights_ptr,
        output_ptr,
        np.prod(value.shape),
        np.prod(shapes.shape),
        np.prod(sampling_locations.shape),
        np.prod(attention_weights.shape),
        np.prod(output_array.shape),
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
    size1 = np.prod(shape)
    size2 = np.prod(output_np.shape)
    # Define the function parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # Call the function with the matrices and dimensions
    elapsed_time = function(input_ptr, output_ptr, size1, size2)
    return elapsed_time


def perf_scaled_dot_product_attention(shape, function, dtype="float32"):
    # Define the function's parameters and return types.
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # Create the input array.
    input_array_1 = np.random.uniform(size=shape).astype(dtype)
    input_array_2 = np.random.uniform(size=shape).astype(dtype)
    input_array_3 = np.random.uniform(size=shape).astype(dtype)
    # Create the output array.
    output_array = np.zeros_like(input_array_1)

    # Convert the input and output arrays to C pointer types.
    input_ptr_1 = input_array_1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_2 = input_array_2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_3 = input_array_3.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Invoke the C function
    elapsed_time = function(
        input_ptr_1, input_ptr_2, input_ptr_3, output_ptr, np.prod(shape)
    )
    return elapsed_time


def perf_pipeline(file_name, op_type="ewise"):
    create_hip_perf_func(file_name, op_type)
    backup_file_name = file_name.replace(".hip", "_bak.hip")
    so_name = file_name.replace(".hip", "hip.so")
    success, output = run_compilation(so_name, backup_file_name)


def benchmark(file_name):
    execution_time = 0
    base_name = os.path.basename(file_name)
    name = base_name.split("_")[0]
    if name == "add":
        perf_pipeline(file_name, "ewise")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_binary(name, shape, shape, shape, function)

    elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
        perf_pipeline(file_name, "pool")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shape = base_name.split("_")[1:5]
        shape = [int(intg) for intg in shape]
        kernel_stride = base_name.split(".")[0].split("_")[5:]
        kernel_stride = [int(intg) for intg in kernel_stride]
        execution_time = perf_pooling(
            name, shape, kernel_stride[:2], kernel_stride[2:], function
        )

    elif name == "bmm":
        perf_pipeline(file_name, "matmul")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
        shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
        shape_B = [batch_size, matrix_dim_k, matrix_dim_j]
        shape_C = [batch_size, matrix_dim_i, matrix_dim_k]
        execution_time = perf_binary(name, shape_A, shape_B, shape_C, function)

    elif name == "gemm":
        perf_pipeline(file_name, "matmul")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        shape_A = [1, shape[0], shape[1]]
        shape_B = [1, shape[1], shape[2]]
        shape_C = [1, shape[0], shape[2]]
        execution_time = perf_binary(name, shape_A, shape_B, shape_C, function)

    elif name in ["sign", "relu", "sigmoid", "softmax", "rmsnorm", "gelu"]:
        perf_pipeline(file_name, "ewise")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_unary(shape, function)

    elif name == "conv2d":
        perf_pipeline(file_name, "matmul")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
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
        execution_time = perf_binary(
            name, data_shape, kernel_shape, output_shape, function
        )

    elif name == "conv2dnchw":
        perf_pipeline(file_name, "matmul")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        data_shape = base_name.split("_")[1:5]
        data_shape = [int(intg) for intg in data_shape]
        kernel_shape = base_name.split("_")[5:9]
        kernel_shape = [int(intg) for intg in kernel_shape]
        stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
        pad = int(base_name.split(".")[0].split("_")[10])

        # generate data
        data_np = torch.rand(data_shape)
        # cpu compute
        result_cpu = conv2d_nchw(
            data_np,
            kernel_shape[1],
            kernel_shape[0],
            kernel_shape[2],
            stride_h,
            pad,
        )
        execution_time = perf_binary(
            name, data_shape, kernel_shape, result_cpu.shape, function
        )

    elif name == "gemv":
        perf_pipeline(file_name, "matmul")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        kernel_shape = [shape[1]]
        output_shape = [shape[0]]
        execution_time = perf_binary(
            name, shape, kernel_shape, output_shape, function
        )

    elif name == "conv1d":
        perf_pipeline(file_name, "matmul")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        shape = [shape[1]]
        kernel_shape = [3]
        output_shape = [shape[0]]
        execution_time = perf_binary(
            name, shape, kernel_shape, output_shape, function
        )

    elif name == "depthwiseconv":
        perf_pipeline(file_name, "matmul")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_depthwiseconv_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        input_height, kernel_size, input_channels = (
            shape[0],
            shape[1],
            shape[2],
        )
        shape = [input_height, input_height, input_channels]
        kernel_shape = [kernel_size, kernel_size, input_channels]
        # Calculate the output tensor shape
        output_height = input_height - kernel_size + 1
        output_width = input_height - kernel_size + 1
        output_shape = [output_height, output_width, input_channels]
        execution_time = perf_binary(
            name, shape, kernel_shape, output_shape, function
        )

    elif name == "deformable":
        perf_pipeline(file_name, "deformable")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_deformable(shape, function)

    # elif name == "mha":
    #     perf_pipeline(file_name, "ewise")
    #     lib = ctypes.CDLL(os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so")))
    #     function = getattr(lib, "timed_" + name + "_kernel")
    #     shapes = base_name.split(".")[0]
    #     shape = [int(intg) for intg in shapes.split("_")[1:]]
    #     perf_scaled_dot_product_attention(shape, function)

    elif name == "layernorm":
        perf_pipeline(file_name, "layer_norm")
        lib = ctypes.CDLL(
            os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so"))
        )
        function = getattr(lib, "timed_" + name + "_kernel")
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_layernorm(shape, [shape[-1]], function)

    else:
        print("Undefined file: ", file_name)
        return 0

    os.remove(os.path.join(os.getcwd(), file_name.replace(".hip", "_bak.hip")))
    os.remove(os.path.join(os.getcwd(), file_name.replace(".hip", "hip.so")))
    return execution_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the transcompile benchmark"
    )
    parser.add_argument(
        "--file_name",
        "-f",
        required=True,
        help="Path to the input HIP file to benchmark",
    )
    args = parser.parse_args()
    execution_time = benchmark(file_name=args.file_name)
    print(f"Execution time: {execution_time:.4f} ms")
