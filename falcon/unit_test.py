import logging
import os
import shutil
import subprocess

from falcon.util import get_target

# Configure the log
logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log format.
)

test_file_map = {
    "deformable": "benchmark/evaluation/{target}_test/test_deformable_attention.py",
    "layernorm": "benchmark/evaluation/{target}_test/test_layer_norm_cuda.py",
    "mha": "benchmark/evaluation/{target}_test/test_mha_cuda.py",
    "rmsnorm": "benchmark/evaluation/{target}_test/test_rms_norm_cuda.py",
    "gemm": "benchmark/evaluation/{target}_test/test_gemm.py",
    "gemv": "benchmark/evaluation/{target}_test/test_gemv.py",
    "bmm": "benchmark/evaluation/{target}_test/test_bmm.py",
    "conv1d": "benchmark/evaluation/{target}_test/test_conv1d.py",
    "conv2d": "benchmark/evaluation/{target}_test/test_conv2d.py",
    "conv2dnchw": "benchmark/evaluation/{target}_test/test_conv2d.py",
    "depthwiseconv": "benchmark/evaluation/{target}_test/test_depthwiseconv.py",
    "add": "benchmark/evaluation/{target}_test/test_add.py",
    "sign": "benchmark/evaluation/{target}_test/test_sign.py",
    "avgpool": "benchmark/evaluation/{target}_test/test_avgpool.py",
    "maxpool": "benchmark/evaluation/{target}_test/test_maxpool.py",
    "minpool": "benchmark/evaluation/{target}_test/test_minpool.py",
    "sumpool": "benchmark/evaluation/{target}_test/test_sumpool.py",
    "relu": "benchmark/evaluation/{target}_test/test_relu.py",
    "sigmoid": "benchmark/evaluation/{target}_test/test_sigmoid.py",
    "gelu": "benchmark/evaluation/{target}_test/test_gelu.py",
    "softmax": "benchmark/evaluation/{target}_test/test_softmax.py",
}


def run_test(file_name, test_file):
    try:
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=400,
        )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output


def unit_test(file_name, code):
    if code is None:
        return False

    # Create a temporary directory
    tmp_dir = "tmps"
    os.makedirs(tmp_dir, exist_ok=True)

    # Remove the extension.
    filename_no_ext, _ = os.path.splitext(file_name)
    # Determine the file type and set the target.
    target, file_type = get_target(code)
    # "Generate target file name"
    filename = filename_no_ext + file_type
    # Extract the operation name and generate the test file path.
    op_name = os.path.basename(filename_no_ext).split("_")[0]

    if target == "cpu":
        code = 'extern "C" ' + code if "extern" not in code else code
    tmp_file_name = os.path.join(tmp_dir, os.path.basename(filename))
    with open(tmp_file_name, mode="w") as f:
        f.write(code)
    if target == "cpu":
        target = "dlboost"
    test_file = test_file_map.get(op_name, "").format(target=target)

    # Run the test.
    success, output = run_test(tmp_file_name, test_file)
    logging.info(output)
    shutil.rmtree(tmp_dir)
    return success, output
