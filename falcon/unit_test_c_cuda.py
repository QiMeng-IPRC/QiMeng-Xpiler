import json
import re
import subprocess
import tempfile
from pathlib import Path
import ctypes
import numpy as np

from falcon.client import invoke_llm
from falcon.unit_test_prompt import (
    CPP_INPUT_PROMPT,
    CUDA_INPUT_PROMPT,
)

from falcon.util import get_target

# =====================================================
# LLM utilities
# =====================================================

def load_prompt(prompt: str, code: str, cpp_plan_json=None) -> str:
    prompt = prompt.replace("{{CODE}}", code)
    if cpp_plan_json is not None:
        prompt = prompt.replace("{{CPP_PLAN_JSON}}", cpp_plan_json)
    return prompt


def extract_json(text: str) -> dict:
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")

    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])

    raise ValueError("Unbalanced JSON")


def make_plan(code: str, prompt: str, cpp_plan_json=None) -> dict:
    content = invoke_llm(load_prompt(prompt, code, cpp_plan_json))
    return extract_json(content)


# =====================================================
# Compile & run utilities
# =====================================================

DTYPE_TO_NP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
}

DTYPE_TO_C = {
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}


def compile_cpp(code: str, out_dir: str, so_name="kernel.so") -> Path:
    src = Path(out_dir) / "kernel.cpp"
    so = Path(out_dir) / so_name
    src.write_text(code)
    subprocess.run(
        ["g++", "-O2", "-shared", "-fPIC", src, "-o", so],
        check=True,
    )
    return so


def compile_cuda(code: str, out_dir: str, so_name="kernel.so") -> Path:
    src = Path(out_dir) / "kernel.cu"
    so = Path(out_dir) / so_name
    src.write_text(code)
    subprocess.run(
        ["nvcc", "-O2", "--shared", "-Xcompiler", "-fPIC", src, "-o", so],
        check=True,
    )
    return so


def run_with_plan(so: Path, plan: dict, shared_inputs=None, function_override=None):
    lib = ctypes.CDLL(str(so))
    fn_name = function_override or plan["function"]
    fn = getattr(lib, fn_name)

    arrays, scalars = {}, {}

    for arg in plan["args"]:
        name, kind = arg["name"], arg["kind"]
        if kind in ("input", "output"):
            shape = [int(x) for x in arg["shape"]]
            n = int(np.prod(shape))
            dtype = DTYPE_TO_NP[arg["dtype"]]
            if kind == "input":
                data = (
                    shared_inputs[name].astype(dtype, copy=True)
                    if shared_inputs and name in shared_inputs
                    else np.random.rand(n).astype(dtype)
                )
            else:
                data = np.empty(n, dtype=dtype)
            arrays[name] = data.reshape(shape)
        else:
            scalars[name] = arg["value"]

    argtypes, call_args = [], []
    for arg in plan["args"]:
        name, kind = arg["name"], arg["kind"]
        if kind in ("input", "output"):
            c_t = DTYPE_TO_C[arg["dtype"]]
            arr = np.ascontiguousarray(arrays[name])
            arrays[name] = arr
            argtypes.append(ctypes.POINTER(c_t))
            call_args.append(arr.ctypes.data_as(ctypes.POINTER(c_t)))
        else:
            c_t = DTYPE_TO_C.get(arg.get("dtype"), ctypes.c_int32)
            argtypes.append(c_t)
            call_args.append(c_t(scalars[name]))

    fn.argtypes = argtypes
    fn.restype = None
    fn(*call_args)

    inputs = {a["name"]: arrays[a["name"]] for a in plan["args"] if a["kind"] == "input"}
    outputs = {a["name"]: arrays[a["name"]] for a in plan["args"] if a["kind"] == "output"}
    return inputs, outputs


# =====================================================
# UnitTest class
# =====================================================

class CudaOpUnitTest:
    def __init__(self, cpp_ref_code: str):
        self.ref_tmp = tempfile.TemporaryDirectory()
        self.ref_so = compile_cpp(cpp_ref_code, self.ref_tmp.name, "ref.so")
        self.cpp_plan = make_plan(cpp_ref_code, CPP_INPUT_PROMPT)

        self.plans = {"cpu": self.cpp_plan}
        self.tmpdirs = []

    def compile_target(self, code: str, target: str) -> Path:
        td = tempfile.TemporaryDirectory()
        self.tmpdirs.append(td)

        if target == "cpu":
            return compile_cpp(code, td.name, "target_cpu.so")
        if target == "cuda":
            return compile_cuda(code, td.name, "target_cuda.so")
        if target == "hip":
            raise NotImplementedError("HIP support not enabled yet")

        raise ValueError(f"Unknown target: {target}")

    def unit_test(self, code: str, target=None, rtol=1e-4, atol=1e-5):
        result = {
            "target": None,
            "status": "unknown",
            "outputs": {},
            "errors": None,
        }

        try:
            tgt, _ = get_target(code, target)
            result["target"] = tgt

            shared_inputs, ref_out = run_with_plan(self.ref_so, self.cpp_plan)

            if tgt == "cpu":
                plan = self.cpp_plan
            else:
                if tgt not in self.plans:
                    self.plans[tgt] = make_plan(
                        code,
                        CUDA_INPUT_PROMPT,
                        cpp_plan_json=json.dumps(self.cpp_plan),
                    )
                plan = self.plans[tgt]

            tgt_so = self.compile_target(code, tgt)

            if tgt == "cpu":
                _, tgt_out = run_with_plan(
                    tgt_so,
                    plan,
                    shared_inputs=shared_inputs,
                    function_override=self.cpp_plan["function"],
                )
            else:
                _, tgt_out = run_with_plan(
                    tgt_so,
                    plan,
                    shared_inputs=shared_inputs,
                )

            ok_all = True
            for name, ref_val in ref_out.items():
                tgt_val = tgt_out.get(name)
                ok = tgt_val is not None and np.allclose(ref_val, tgt_val, rtol, atol)
                result["outputs"][name] = {
                    "success": bool(ok),
                    "max_abs_diff": float(np.max(np.abs(ref_val - tgt_val))) if not ok else 0.0,
                }
                if not ok:
                    ok_all = False

            result["status"] = "success" if ok_all else "mismatch"
            return ok_all, result

        except Exception as e:
            result["status"] = "error"
            result["errors"] = str(e)
            return False, result


# =====================================================
# Main
# =====================================================

def main():
    cpp_ref = r"""
extern "C" void conv1d(float *input, float *kernel, float *output) {
  for (int i = 0; i < 5; i++) {
    output[i] = 0;
    for (int j = 0; j < 3; j++) {
      output[i] += input[i + j] * kernel[j];
    }
  }
}
"""

    cuda_code = r"""
#include <cuda_runtime.h>

__global__ void conv1d(float *input, float *kernel, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 5) {
    output[idx] = 0;
    for (int j = 0; j < 3; j++) {
      output[idx] += input[idx + j] * kernel[j];
    }
  }
}

extern "C" void conv1d_kernel(float *input, float *kernel, float *output,
                             int input_size, int output_size) {
  float *d_input, *d_kernel, *d_output;
  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_kernel, (input_size - output_size + 1) * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));

  cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, (input_size - output_size + 1) * sizeof(float),
             cudaMemcpyHostToDevice);

  conv1d<<<1, 32>>>(d_input, d_kernel, d_output);
  cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
}
"""

    tester = CudaOpUnitTest(cpp_ref)
    ok, info = tester.unit_test(cuda_code)
    print(ok)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
