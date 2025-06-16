import csv
import glob
import os
import timeit

import torch
import torch.nn.functional as F


@torch.no_grad()
def deformable_attention_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pytorch implementation of deformable attention from
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split(
        [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_,
        # D_, H_, W_
        value_l_ = (
            value_list[lid_]
            .flatten(2)
            .transpose(1, 2)
            .reshape(N_ * M_, D_, H_, W_)
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = (
            sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        )
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    output = (
        (
            torch.stack(sampling_value_list, dim=-2).flatten(-2)
            * attention_weights
        )
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def perf_elementwise(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(shape, device=device)
    y = torch.randn(shape, device=device)

    def test_add():
        torch.add(x, y)

    def test_sign():
        torch.sign(x)

    op_name = name.split("_")[0]
    if op_name == "add":
        # Use timeit to perform multiple measurements, setting the number of executions to 100.
        execution_time = timeit.timeit(test_add, number=100)
        print(f"{name} execution time: {execution_time * 10} ms")

    elif op_name == "sign":
        execution_time = timeit.timeit(test_sign, number=100)
        print(f"{name} execution time: {execution_time * 10} ms")
    return execution_time * 10


def perf_pooling(name, shape, kernel, stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a random input tensor.
    x = torch.randn(shape, device=device)
    # Define the MaxPooling operation.
    op_name = name.split("_")[0]

    pool = torch.nn.AvgPool2d(kernel_size=kernel, stride=stride)
    if op_name == "maxpool":
        pool = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride)

    elif op_name == "minpool":

        class MinPool2d(torch.nn.Module):
            def __init__(self, kernel_size, stride=None, padding=0):
                super(MinPool2d, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding

            def forward(self, x):
                # Inverted input
                x_neg = -x
                # Perform maximum pooling.
                x_maxpool = F.max_pool2d(
                    x_neg,
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
                # Reversed result again
                return -x_maxpool

        # Using a custom MinPool2d
        pool = MinPool2d(kernel_size=kernel, stride=stride)

    elif op_name == "sumpool":

        class SumPool2d(torch.nn.Module):
            def __init__(self, kernel_size, stride=None, padding=0):
                super(SumPool2d, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding

            def forward(self, x):
                # Use average pooling.
                x_avgpool = F.avg_pool2d(
                    x,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
                # Multiply by the size of the pooling window to achieve the effect of sum pooling.
                return x_avgpool * (self.kernel_size**2)

        # Using a custom SumPool2d
        pool = SumPool2d(kernel_size=kernel, stride=stride)

    def test_pool():
        pool(x)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_pool, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")
    return execution_time * 10


def perf_bmm(name, shape_A, shape_B):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a random tensor.
    A = torch.randn(shape_A, dtype=torch.float16, device=device)
    B = torch.randn(shape_B, dtype=torch.float16, device=device)

    def test_gemm():
        # Perform matrix multiplication operation (GEMM).
        torch.matmul(A, B)
        # Ensure CUDA operations are completed.

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_gemm, number=100)
    return execution_time * 10


def perf_activation(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(shape, device=device)
    name.split("_")[0]

    activation = torch.nn.ReLU()
    if name == "sigmoid":
        activation = torch.nn.Sigmoid()
    elif name == "gelu":
        activation = torch.nn.GELU()
    elif name == "softmax":
        activation = torch.nn.Softmax(dim=len(shape))

    def test_activation():
        activation(x)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_activation, number=100)
    return execution_time * 10


def perf_conv2d_nchw(
    name, shape, in_channels, out_channels, kernel, stride, padding
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(shape, device=device)
    # Define the convolutional layer.
    conv_layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
    ).to(device)

    def test_conv2d():
        conv_layer(input_tensor)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_conv2d, number=100)
    return execution_time * 10


def perf_conv2d_nhwc(
    name, shape, in_channels, out_channels, kernel, stride, padding
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_nhwc = torch.randn(shape, device=device)
    weight_hwio = torch.randn(
        [out_channels, kernel, kernel, shape[3]], device=device
    )

    def test_conv2d():
        # Convert the input from NHWC to NCHW.
        input_nchw = input_nhwc.permute(0, 3, 1, 2)

        # Convert the kernel from HWIO (H, W, in_channels, out_channels) format to PyTorch's OIHW format.
        weight_oihw = weight_hwio.permute(0, 3, 1, 2)

        # Perform convolution operations using the transformed convolution kernel and input.
        output_nchw = F.conv2d(
            input_nchw, weight_oihw, stride=stride, padding=padding
        )

        # Convert the output from NCHW back to NHWC.
        output_nchw.permute(0, 3, 1, 2)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_conv2d, number=100)
    return execution_time * 10


def perf_gemv(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create random matrix and vector M, N = 1000, 1000.
    matrix = torch.randn(shape, device=device)
    vector = torch.randn(shape[1], device=device)

    def test_gemv():
        torch.matmul(matrix, vector)
        # Alternatively, use matrix @ vector.

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_gemv, number=100)
    return execution_time * 10


def perf_conv1d(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the input tensor.
    input_tensor = torch.randn(1, shape[1], device=device)
    # Define the convolutional layer.
    conv_layer = torch.nn.Conv1d(
        in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0
    ).to(device)

    def test_conv1d():
        conv_layer(input_tensor)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_conv1d, number=100)
    return execution_time * 10


def perf_depthwise_conv2d(name, shape, kernel_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_hwio = torch.randn(shape, device=device)
    weight_fdio = torch.randn(
        [kernel_size, kernel_size, shape[2]], device=device
    )

    def test_depthwise_conv2d():
        # The input is (height, width, in_depth); by adding a batch dimension, it becomes (1, height, width, in_depth).
        input_nchw = input_hwio.unsqueeze(0).permute(
            0, 3, 1, 2
        )  # Convert to (1, in_depth, height, width)

        # The convolution kernel is (fd, fd, in_depth) and needs to be converted to (in_depth, 1, fd, fd).
        in_depth = weight_fdio.shape[2]
        weight_iodf = weight_fdio.permute(2, 0, 1).unsqueeze(
            1
        )  # Convert to (in_depth, 1, fd, fd)

        # Use depthwise separable convolution.
        output_nchw = F.conv2d(input_nchw, weight_iodf, groups=in_depth)

        # Convert the output format from (1, in_depth, new_height, new_width) to (new_height,
        # new_width, in_depth)
        output_nchw.squeeze(0).permute(1, 2, 0)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_depthwise_conv2d, number=100)
    return execution_time * 10


def perf_layernorm(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the input tensor.
    input_tensor = torch.randn(shape, device=device)
    # Define the LayerNorm layer.
    layer_norm = torch.nn.LayerNorm(shape[-1]).to(device)

    def test_layernorm():
        layer_norm(input_tensor)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_layernorm, number=100)
    return execution_time * 10


def perf_rmsnorm(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the input tensor.
    input_tensor = torch.randn(shape, device=device)
    # Define the RMSNorm layer.
    rmsnorm = torch.nn.RMSNorm(shape).to(device)

    def test_rmsnorm():
        rmsnorm(input_tensor)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_rmsnorm, number=100)
    return execution_time * 10


def perf_deformable(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, M, D = shape[:3]
    Lq, L, P = shape[3:]
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]],
        dtype=torch.long,
        device=device,
    )
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D, device=device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2, device=device)
    attention_weights = torch.rand(N, Lq, M, L, P, device=device) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True
    )

    def test_deformable():
        deformable_attention_pytorch(
            value,
            shapes,
            sampling_locations,
            attention_weights,
        )
        # necessary because kernel launches are async

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(test_deformable, number=100)
    return execution_time * 10


def perf_scaled_dot_product_attention(name, shape):
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optionally use the context manager to ensure one of the fused kernels is
    # run
    query = torch.rand(shape)
    key = torch.rand(shape)
    value = torch.rand(shape)

    def test_scaled_dot_product_attention():
        F.scaled_dot_product_attention(query, key, value)

    # Use timeit to perform multiple measurements, setting the number of executions to 100.
    execution_time = timeit.timeit(
        test_scaled_dot_product_attention, number=100
    )
    return execution_time * 10


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/cpp_code_test/*.cpp")
    )
    counter = 0
    execution_time = 0
    table = []
    times = []
    for file in files:
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "add" or name == "sign":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_elementwise(base_name, shape)

        elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
            shape = base_name.split("_")[1:5]
            shape = [int(intg) for intg in shape]
            kernel_stride = base_name.split(".")[0].split("_")[5:]
            kernel_stride = [int(intg) for intg in kernel_stride]
            execution_time = perf_pooling(
                base_name, shape, kernel_stride[0], kernel_stride[2]
            )

        elif name == "bmm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
            shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
            shape_B = [batch_size, matrix_dim_j, matrix_dim_k]
            execution_time = perf_bmm(name, shape_A, shape_B)

        elif name == "gemm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            shape_A = [shape[0], shape[1]]
            shape_B = [shape[1], shape[2]]
            execution_time = perf_bmm(name, shape_A, shape_B)

        elif name in ["relu", "sigmoid", "gelu", "softmax"]:
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_activation(base_name, shape)

        elif name == "conv2dnchw":
            data_shape = base_name.split("_")[1:5]
            data_shape = [int(intg) for intg in data_shape]
            kernel_shape = base_name.split("_")[5:9]
            kernel_shape = [int(intg) for intg in kernel_shape]
            stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
            pad = int(base_name.split(".")[0].split("_")[10])

            execution_time = perf_conv2d_nchw(
                base_name,
                data_shape,
                kernel_shape[1],
                kernel_shape[0],
                kernel_shape[2],
                stride_h,
                pad,
            )

        elif name == "gemv":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_gemv(base_name, shape)

        elif name == "conv2d":
            data_shape = base_name.split("_")[1:5]
            data_shape = [int(intg) for intg in data_shape]
            kernel_shape = base_name.split("_")[5:9]
            kernel_shape = [int(intg) for intg in kernel_shape]
            stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
            pad = int(base_name.split(".")[0].split("_")[10])

            execution_time = perf_conv2d_nhwc(
                base_name,
                data_shape,
                kernel_shape[1],
                kernel_shape[0],
                kernel_shape[2],
                stride_h,
                pad,
            )
        elif name == "conv1d":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_conv1d(base_name, shape)

        elif name == "depthwiseconv":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            input_height, kernel_size, input_channels = (
                shape[0],
                shape[1],
                shape[2],
            )
            shape = [input_height, input_height, input_channels]
            execution_time = perf_depthwise_conv2d(
                base_name, shape, kernel_size
            )

        elif name == "layernorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_layernorm(base_name, shape)

        elif name == "rmsnorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_rmsnorm(base_name, shape)
        elif name == "deformable":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_deformable(base_name, shape)

        elif name == "mha":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_scaled_dot_product_attention(
                base_name, shape
            )
        times.append(execution_time)

    table.append(files)
    table.append(times)

    # Transpose data
    transposed_data = list(zip(*table))

    # Add a title row.
    header = ["file", "time(ms)"]
    transposed_data.insert(0, header)

    # Save as CSV file.
    with open("benchmark/perf/oneAPI_output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(transposed_data)
