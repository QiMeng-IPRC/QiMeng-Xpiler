import glob
import os

import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")
# Create a random tensor with a data type of float16.
# shapes = [
# [128, 128, 128],
# [128, 256, 512],
# [256, 256, 256],
# [512, 512, 512],
# [128, 512, 256],
# [256, 512, 256],
# [512, 512, 256],
# [256, 128, 256],
# ]
times = []
files = glob.glob(
    os.path.join(os.getcwd(), "benchmark/data/dlboost_code_test/mha*.cpp")
)
for file in files:
    base_name = os.path.basename(file)
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

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./time"),
    ) as p:
        for _ in range(1000):
            test_scaled_dot_product_attention()
    end_event.record()
    torch.cuda.current_stream().synchronize()
    print(base_name)
    total_e2e_time = start_event.elapsed_time(end_event)
    print(p.key_averages().table(sort_by="self_cpu_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == "aten::matmul":  # Find the operation you need based on the key.
            time = avg.cuda_time

    times.append(time)
print(times)
