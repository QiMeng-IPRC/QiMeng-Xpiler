import glob
import os
import timeit

import torch

# torch.set_float32_matmul_precision('medium')
device = torch.device("cpu")
# Create a random tensor with a data type of float32.


times = []
files = glob.glob(
    os.path.join(os.getcwd(), "benchmark/data/mlu_code_test/bmm*.mlu")
)
for file in files:
    base_name = os.path.basename(file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]

    # # Optionally use the context manager to ensure one of the fused kernels is run
    # query = torch.rand(shape, dtype=torch.float32, device=device)
    # key = torch.rand(shape, dtype=torch.float32, device=device)
    # value = torch.rand(shape, dtype=torch.float32, device=device)

    # def test_scaled_dot_product_attention():
    #     output = F.scaled_dot_product_attention(query, key, value)
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
    shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
    shape_B = [batch_size, matrix_dim_j, matrix_dim_k]
    A = torch.randn(shape_A, dtype=torch.float32, device=device)
    B = torch.randn(shape_B, dtype=torch.float32, device=device)
    use_amp = True

    def test_bmm():
        torch.matmul(A, B)

    # Use timeit to perform multiple measurements, setting the number of
    # executions to 100.
    execution_time = timeit.timeit(test_bmm, number=100)
    print(f"{base_name} execution time: {execution_time * 10} ms")
