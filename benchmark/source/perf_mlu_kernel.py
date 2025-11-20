import glob
import os

import torch
import torch.nn.functional as F

# torch.set_float32_matmul_precision('medium')
device = torch.device("mlu")
# Create a random tensor with a data type of float16.


times = []
files = glob.glob(
    os.path.join(os.getcwd(), "benchmark/data/dlboost_code_test/mha*.cpp")
)
for file in files:
    base_name = os.path.basename(file)
    print(base_name)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Optionally use the context manager to ensure one of the fused kernels is
    # run
    query = torch.rand(shape, dtype=torch.float16, device=device)
    key = torch.rand(shape, dtype=torch.float16, device=device)
    value = torch.rand(shape, dtype=torch.float16, device=device)

    def test_scaled_dot_product_attention():
        F.scaled_dot_product_attention(query, key, value)

    for _ in range(100):
        test_scaled_dot_product_attention()
    start = torch.mlu.Event(enable_timing=True)
    end = torch.mlu.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.MLU,
        ),
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./time"),
    ) as p:
        for _ in range(nb_iters):
            test_scaled_dot_product_attention()
    end.record()
    torch.mlu.current_stream().synchronize()

    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if (
            avg.key == "aten::scaled_dot_product_attention"
        ):  # Find the operation you need based on the key.
            time = avg.mlu_time
    print(f"eval {base_name}, {time}")
    times.append(time)
print(times)


files = glob.glob(
    os.path.join(os.getcwd(), "benchmark/data/dlboost_code_test/gemm*.cpp")
)
for file in files:
    base_name = os.path.basename(file)
    print(base_name)
    shape_A = [shape[0], shape[1]]
    shape_B = [shape[1], shape[2]]

    A = torch.randn(shape_A, dtype=torch.float16, device=device)
    B = torch.randn(shape_B, dtype=torch.float16, device=device)
    use_amp = True

    REPEAT = 1000
    for _ in range(100):  # warm up
        C = torch.matmul(A, B)

    torch.mlu.synchronize()
    start_event = torch.mlu.Event(enable_timing=True)
    end_event = torch.mlu.Event(enable_timing=True)

    start_event.record()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.MLU,
        ),
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./time"),
    ) as p:
        for _ in range(REPEAT):
            C = torch.matmul(A, B)
    end_event.record()
    torch.mlu.current_stream().synchronize()
    total_e2e_time = start_event.elapsed_time(end_event)
    print(p.key_averages().table(sort_by="self_mlu_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if (
            avg.key == "aten::matmul"
        ):  # Find the operation you need based on the key.
            time = avg.mlu_time

    times.append(time)
print(times)

files = glob.glob(
    os.path.join(os.getcwd(), "benchmark/data/dlboost_code_test/bmm*.cpp")
)
for file in files:
    base_name = os.path.basename(file)
    print(base_name)
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
    shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
    shape_B = [batch_size, matrix_dim_j, matrix_dim_k]
    A = torch.randn(shape_A, dtype=torch.float16, device=device)
    B = torch.randn(shape_B, dtype=torch.float16, device=device)
    use_amp = True

    REPEAT = 1000
    for _ in range(100):  # warm up
        C = torch.matmul(A, B)

    torch.mlu.synchronize()
    start_event = torch.mlu.Event(enable_timing=True)
    end_event = torch.mlu.Event(enable_timing=True)

    start_event.record()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.MLU,
        ),
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./time"),
    ) as p:
        for _ in range(REPEAT):
            C = torch.matmul(A, B)
    end_event.record()
    torch.mlu.current_stream().synchronize()
    total_e2e_time = start_event.elapsed_time(end_event)
    print(p.key_averages().table(sort_by="self_mlu_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if (
            avg.key == "aten::matmul"
        ):  # Find the operation you need based on the key.
            time = avg.mlu_time

    times.append(time)
print(times)
