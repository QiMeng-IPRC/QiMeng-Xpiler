import json
import re
import subprocess
import tempfile
from pathlib import Path
import ctypes
import numpy as np
import time
import traceback
from falcon.client import invoke_llm
from falcon.unit_test_prompt import (
    CPP_INPUT_PROMPT,
    CUDA_INPUT_PROMPT,
)
from typing import Dict, Any, List
from falcon.util import get_target

# =====================================================
# LLM utilities
# =====================================================



DTYPE_TO_CUDA_C = {
    "float32": "float",
    "float64": "double",
    "int32": "int",
    "int64": "long long",
}

DTYPE_TO_SIZEOF = {
    "float32": "sizeof(float)",
    "float64": "sizeof(double)",
    "int32": "sizeof(int)",
    "int64": "sizeof(long long)",
}

def _strip_extern_c(code: str) -> str:
    # Avoid double extern "C" issues if user code already contains it
    return re.sub(r'extern\s*"C"\s*', "", code)

def _expr_prod(dims: List[Any]) -> str:
    # dims are strings or ints
    if not dims:
        return "1"
    parts = []
    for d in dims:
        if isinstance(d, int):
            parts.append(str(d))
        else:
            d = str(d).strip()
            parts.append(d)
    return " * ".join(parts) if parts else "1"

def _ensure_launch(plan: Dict[str, Any]) -> Dict[str, Any]:
    # Provide conservative defaults if missing
    launch = plan.get("launch") or {}
    block = launch.get("block")
    grid = launch.get("grid")

    if not block or not isinstance(block, list) or len(block) != 3:
        # default 1D
        block = [256, 1, 1]
    if not grid or not isinstance(grid, list) or len(grid) != 3:
        # default uses size1
        grid = ["ceil_div(size1, %d)" % int(block[0]), "1", "1"]

    return {"block": block, "grid": grid}

def create_cuda_timed_host_from_plan(code: str, plan: Dict[str, Any]) -> str:
    kernel_name = plan.get("kernel_name")
    args = plan.get("args", [])
    if not kernel_name or not isinstance(args, list) or not args:
        raise ValueError("plan must contain kernel_name and non-empty args list")

    # launch config
    launch = _ensure_launch(plan)
    block = launch["block"]
    grid = launch["grid"]

    # clean kernel code + detect whether blockIdx.* used
    cleaned_kernel_code = _strip_extern_c(code).strip()
    uses_blockidx_x = ("blockIdx.x" in cleaned_kernel_code)
    uses_blockidx_y = ("blockIdx.y" in cleaned_kernel_code)
    uses_blockidx_z = ("blockIdx.z" in cleaned_kernel_code)

    ptr_args = [a for a in args if a.get("kind") in ("input", "output")]
    scalar_args = [a for a in args if a.get("kind") == "scalar"]
    if not ptr_args:
        raise ValueError("plan.args must contain at least one pointer arg (input/output)")

    timed_name = f"timed_{kernel_name}_kernel"

    # wrapper signature: ptrs + scalars + warmup/repeat
    sig_parts = []
    for a in ptr_args:
        name = a["name"]
        dtype = a["dtype"]
        ctype = DTYPE_TO_CUDA_C.get(dtype)
        if ctype is None:
            raise ValueError(f"Unsupported dtype for pointer arg {name}: {dtype}")
        sig_parts.append(f"{ctype} *{name}")

    for a in scalar_args:
        name = a["name"]
        dtype = a.get("dtype", "int32")
        if dtype not in ("int32", "int64"):
            dtype = "int32"
        sig_parts.append(f"{DTYPE_TO_CUDA_C[dtype]} {name}")

    sig_parts.append("int warmup")
    sig_parts.append("int repeat")
    wrapper_sig = ", ".join(sig_parts)

    # device malloc
    dev_decl = []
    dev_malloc = []
    for a in ptr_args:
        name = a["name"]
        dtype = a["dtype"]
        ctype = DTYPE_TO_CUDA_C[dtype]
        sizeof_expr = DTYPE_TO_SIZEOF[dtype]
        shape = a.get("shape") or [1]
        elem_expr = _expr_prod(shape)
        bytes_expr = f"({elem_expr}) * {sizeof_expr}"
        dev_decl.append(f"{ctype} *d_{name} = nullptr;")
        dev_malloc.append(f"CUDA_CHECK(cudaMalloc((void**)&d_{name}, {bytes_expr}));")

    # H2D inputs
    h2d = []
    for a in ptr_args:
        if a.get("kind") != "input":
            continue
        name = a["name"]
        dtype = a["dtype"]
        sizeof_expr = DTYPE_TO_SIZEOF[dtype]
        shape = a.get("shape") or [1]
        elem_expr = _expr_prod(shape)
        bytes_expr = f"({elem_expr}) * {sizeof_expr}"
        h2d.append(f"CUDA_CHECK(cudaMemcpy(d_{name}, {name}, {bytes_expr}, cudaMemcpyHostToDevice));")

    # kernel call args: by default only device pointers in plan order
    kernel_call_ptrs = [f"d_{a['name']}" for a in ptr_args]
    kernel_params = plan.get("kernel_params")
    if isinstance(kernel_params, list) and kernel_params:
        ptr_names = {a["name"] for a in ptr_args}
        call_args = []
        for pname in kernel_params:
            if pname in ptr_names:
                call_args.append(f"d_{pname}")
            else:
                call_args.append(pname)
        kernel_call = ", ".join(call_args)
    else:
        kernel_call = ", ".join(kernel_call_ptrs)

    # launch override ONLY for threadIdx-only kernels (no blockIdx.*)
    bx, by, bz = block
    gx, gy, gz = grid
    work_items = plan.get("work_items", None)
    if work_items is not None:
        try:
            wi = int(work_items)
        except Exception:
            wi = None
        if wi is not None and wi > 0 and (not uses_blockidx_x) and (not uses_blockidx_y) and (not uses_blockidx_z):
            if wi <= 1024:
                bx, by, bz = wi, 1, 1
                gx, gy, gz = 1, 1, 1
            else:
                bx, by, bz = 256, 1, 1
                gx, gy, gz = f"ceil_div({wi}, {bx})", 1, 1

    launch_setup = [
        f"dim3 blockDim_({int(bx)}, {int(by)}, {int(bz)});",
        f"dim3 gridDim_({gx}, {gy}, {gz});",
    ]

    # frees
    frees = [f"CUDA_CHECK(cudaFree(d_{a['name']}));" for a in ptr_args]

    return f"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {{ \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) {{ \\
        printf("CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
    }} \\
}} while(0)
#endif

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {{
    return (a + b - 1) / b;
}}
#define ceil_div(a,b) ceil_div_int((int)(a),(int)(b))

// ========================
// Original CUDA code
// ========================
{cleaned_kernel_code}

// ========================
// Auto-generated kernel-only timing wrapper
// ========================
extern "C" float {timed_name}({wrapper_sig}) {{
    {" ".join(dev_decl)}
    {" ".join(dev_malloc)}
    {" ".join(h2d)}

    {" ".join(launch_setup)}

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    // Warmup
    for (int i = 0; i < warmup; ++i) {{
        {kernel_name}<<<gridDim_, blockDim_>>>({kernel_call});
    }}
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < repeat; ++i) {{
        {kernel_name}<<<gridDim_, blockDim_>>>({kernel_call});
    }}
    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
    float avg_ms = (repeat > 0) ? (total_ms / (float)repeat) : 0.0f;

    {" ".join(frees)}

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    return avg_ms;
}}
""".strip()

def _numel_from_shape(shape: List[Any]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)

def _fill_work_items_from_cpp_plan(plan: Dict[str, Any], cpp_plan: Dict[str, Any]) -> None:
    """
    Set plan["work_items"] from the first output tensor in cpp_plan.
    Used only as a helper for threadIdx-only kernels when plan already contains work_items or is missing it.
    """
    out_arg = next((a for a in cpp_plan.get("args", []) if a.get("kind") == "output"), None)
    if not out_arg:
        return
    shape = out_arg.get("shape")
    if not shape:
        return
    plan["work_items"] = _numel_from_shape(shape)


def create_cuda_host_from_plan(code: str, plan: Dict[str, Any]) -> str:
    """
    Generate a compilable CUDA .cu source by adding an extern "C" host wrapper
    according to `plan`. The wrapper:
      - allocates device buffers for pointer args,
      - memcpy inputs Host->Device,
      - launches the __global__ kernel,
      - memcpy outputs Device->Host,
      - frees device buffers,
      - performs basic error checks.

    Notes:
    - plan["args"] pointer shapes should be fully concrete ints (recommended, from CPP plan).
    - plan["work_items"] is OPTIONAL:
        * If kernel does NOT use blockIdx.x, we may force a safe single-block launch
          for threadIdx-only kernels (or 1D grid for wi>1024).
        * If kernel uses blockIdx.x, we NEVER override plan.launch using work_items.
    - Scalars in plan["args"] are included in wrapper signature. By default, we do NOT
      pass scalars into the kernel unless plan["kernel_params"] specifies an explicit order.
    """

    # ---------- validate plan ----------
    kernel_name = plan.get("kernel_name")
    wrapper_name = plan.get("function")
    args = plan.get("args", [])
    if not kernel_name or not wrapper_name or not isinstance(args, list) or not args:
        raise ValueError("plan must contain kernel_name, function, and non-empty args list")

    # ---------- normalize launch ----------
    launch = _ensure_launch(plan)
    block = launch["block"]
    grid = launch["grid"]

    # ---------- clean code + detect indexing ----------
    cleaned_kernel_code = _strip_extern_c(code).strip()
    uses_blockidx_x = ("blockIdx.x" in cleaned_kernel_code)
    uses_blockidx_y = ("blockIdx.y" in cleaned_kernel_code)
    uses_blockidx_z = ("blockIdx.z" in cleaned_kernel_code)

    # ---------- collect args ----------
    ptr_args = [a for a in args if a.get("kind") in ("input", "output")]
    scalar_args = [a for a in args if a.get("kind") == "scalar"]

    if not ptr_args:
        raise ValueError("plan.args must contain at least one pointer arg (input/output)")

    # ---------- wrapper signature ----------
    # extern "C" void <wrapper>(<host pointers>, <scalars...>)
    sig_parts = []
    for a in ptr_args:
        name = a["name"]
        dtype = a["dtype"]
        ctype = DTYPE_TO_CUDA_C.get(dtype)
        if ctype is None:
            raise ValueError(f"Unsupported dtype for pointer arg {name}: {dtype}")
        sig_parts.append(f"{ctype} *{name}")

    for a in scalar_args:
        name = a["name"]
        dtype = a.get("dtype", "int32")
        if dtype not in ("int32", "int64"):
            dtype = "int32"
        ctype = DTYPE_TO_CUDA_C[dtype]
        sig_parts.append(f"{ctype} {name}")

    wrapper_sig = ", ".join(sig_parts)

    # ---------- device declarations + malloc ----------
    dev_decl = []
    dev_malloc = []
    for a in ptr_args:
        name = a["name"]
        dtype = a["dtype"]
        ctype = DTYPE_TO_CUDA_C[dtype]
        sizeof_expr = DTYPE_TO_SIZEOF[dtype]

        shape = a.get("shape") or [1]
        elem_expr = _expr_prod(shape)  # "1 * 15 * 64"
        bytes_expr = f"({elem_expr}) * {sizeof_expr}"

        dev_name = f"d_{name}"
        dev_decl.append(f"{ctype} *{dev_name} = nullptr;")
        dev_malloc.append(f"CUDA_CHECK(cudaMalloc((void**)&{dev_name}, {bytes_expr}));")

    # ---------- memcpy H2D for inputs ----------
    h2d = []
    for a in ptr_args:
        if a.get("kind") != "input":
            continue
        name = a["name"]
        dtype = a["dtype"]
        sizeof_expr = DTYPE_TO_SIZEOF[dtype]
        shape = a.get("shape") or [1]
        elem_expr = _expr_prod(shape)
        bytes_expr = f"({elem_expr}) * {sizeof_expr}"
        h2d.append(f"CUDA_CHECK(cudaMemcpy(d_{name}, {name}, {bytes_expr}, cudaMemcpyHostToDevice));")

    # ---------- kernel call args ----------
    # Default: pass only device pointers in plan order.
    kernel_call_ptrs = [f"d_{a['name']}" for a in ptr_args]

    kernel_params = plan.get("kernel_params")
    if isinstance(kernel_params, list) and kernel_params:
        # kernel_params is an explicit ordered list of names (pointers and/or scalars).
        # Pointers will be passed as device pointers; scalars passed as-is.
        call_args = []
        ptr_names = {a["name"] for a in ptr_args}
        for pname in kernel_params:
            if pname in ptr_names:
                call_args.append(f"d_{pname}")
            else:
                call_args.append(pname)
        kernel_call = ", ".join(call_args)
    else:
        kernel_call = ", ".join(kernel_call_ptrs)

    # ---------- launch config ----------
    bx, by, bz = block
    gx, gy, gz = grid

    work_items = plan.get("work_items", None)
    if work_items is not None:
        try:
            wi = int(work_items)
        except Exception:
            wi = None

        # Only override when kernel does NOT use blockIdx.* at all.
        # (If kernel uses blockIdx.x/y/z, overriding would break correctness.)
        if wi is not None and wi > 0 and (not uses_blockidx_x) and (not uses_blockidx_y) and (not uses_blockidx_z):
            if wi <= 1024:
                # Safe single-block launch for threadIdx-only kernels
                bx, by, bz = wi, 1, 1
                gx, gy, gz = 1, 1, 1
            else:
                # 1D grid covering wi items
                bx, by, bz = 256, 1, 1
                gx, gy, gz = f"ceil_div({wi}, {bx})", 1, 1

    launch_lines = [
        f"dim3 blockDim_({int(bx)}, {int(by)}, {int(bz)});",
        f"dim3 gridDim_({gx}, {gy}, {gz});",
        f"{kernel_name}<<<gridDim_, blockDim_>>>({kernel_call});",
        "CUDA_CHECK(cudaGetLastError());",
        # Ensure E2E correctness (kernel finished before copying back)
        "CUDA_CHECK(cudaDeviceSynchronize());",
    ]

    # ---------- memcpy D2H for outputs ----------
    d2h = []
    for a in ptr_args:
        if a.get("kind") != "output":
            continue
        name = a["name"]
        dtype = a["dtype"]
        sizeof_expr = DTYPE_TO_SIZEOF[dtype]
        shape = a.get("shape") or [1]
        elem_expr = _expr_prod(shape)
        bytes_expr = f"({elem_expr}) * {sizeof_expr}"
        d2h.append(f"CUDA_CHECK(cudaMemcpy({name}, d_{name}, {bytes_expr}, cudaMemcpyDeviceToHost));")

    # ---------- free ----------
    frees = [f"CUDA_CHECK(cudaFree(d_{a['name']}));" for a in ptr_args]

    # ---------- compose .cu ----------
    # NOTE: keep ceil_div macro for plans that emit "ceil_div(...)" in grid expressions
    out = f"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {{ \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) {{ \\
        printf("CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
    }} \\
}} while(0)
#endif

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {{
    return (a + b - 1) / b;
}}
#define ceil_div(a,b) ceil_div_int((int)(a),(int)(b))

// ========================
// Original CUDA code
// ========================
{cleaned_kernel_code}

// ========================
// Auto-generated host wrapper
// ========================
extern "C" void {wrapper_name}({wrapper_sig}) {{
    // Device allocations
    {" ".join(dev_decl)}
    {" ".join(dev_malloc)}

    // Copy inputs H2D
    {" ".join(h2d)}

    // Launch kernel
    {" ".join(launch_lines)}

    // Copy outputs D2H
    {" ".join(d2h)}

    // Free device memory
    {" ".join(frees)}
}}
"""
    return out.strip()



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
    
            # 1) CPU: ensure C ABI export
            if tgt == "cpu":
                code = 'extern "C" ' + code if "extern" not in code else code
    
            # 2) Build shared inputs by running reference CPU impl once
            shared_inputs, ref_out = run_with_plan(self.ref_so, self.cpp_plan)
    
            # 3) Prepare plan + (for CUDA) generate host wrapper code
            if tgt == "cpu":
                plan = self.cpp_plan
                final_code = code
            else:
                # Always regenerate plan for CUDA to reflect current code
                plan = make_plan(
                    code,
                    CUDA_INPUT_PROMPT,
                    cpp_plan_json=json.dumps(self.cpp_plan),
                )
                
                # 保险：强制用 cpp_plan 的 tensor args 覆盖（双保险，防止 LLM 偶尔不听话）
                cpu_tensor_args = [a for a in self.cpp_plan["args"] if a["kind"] in ("input", "output")]
                plan["args"] = cpu_tensor_args + [a for a in plan["args"] if a["kind"] == "scalar"]
                out_arg = next((a for a in self.cpp_plan["args"] if a["kind"] == "output"), None)
                if out_arg:
                    plan["work_items"] = int(np.prod(out_arg["shape"]))
                                
                # 自动生成 host wrapper（extern "C" <kernel>_kernel）
                final_code = create_cuda_host_from_plan(code, plan)
            print(plan)


            # 4) Compile target
            tgt_so = self.compile_target(final_code, tgt)
    
            # 5) Run target
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
    
            # 6) Compare outputs
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


    def perf(self, code: str, target=None, warmup: int = 10, repeat: int = 100, outer_repeat: int = 10):
        result = {"target": None, "status": "unknown", "kernel_ms": {}, "errors": None}
        try:
            tgt, _ = get_target(code, target)
            result["target"] = tgt
            if tgt != "cuda":
                raise ValueError("kernel-only perf currently supports only CUDA target")
    
            # 1) run ref once to build shared inputs (host arrays)
            shared_inputs, _ = run_with_plan(self.ref_so, self.cpp_plan)
    
            # 2) make plan (CUDA) and force pointer args from cpp_plan
            plan = make_plan(code, CUDA_INPUT_PROMPT, cpp_plan_json=json.dumps(self.cpp_plan))
            cpu_tensor_args = [a for a in self.cpp_plan["args"] if a["kind"] in ("input", "output")]
            plan["args"] = cpu_tensor_args + [a for a in plan.get("args", []) if a.get("kind") == "scalar"]
    
            # optional robust work_items fill (good for threadIdx-only kernels)
            _fill_work_items_from_cpp_plan(plan, self.cpp_plan)
    
            # 3) point function to timed symbol and generate timed wrapper code
            plan["function"] = f"timed_{plan['kernel_name']}_kernel"
            final_code = create_cuda_timed_host_from_plan(code, plan)
    
            # 4) compile
            so = self.compile_target(final_code, "cuda")
    
            # 5) ctypes call timed wrapper: returns float avg_ms
            lib = ctypes.CDLL(str(so))
            fn = getattr(lib, plan["function"])
            fn.restype = ctypes.c_float
    
            # build call args from plan (pointer args from shared_inputs, outputs allocated)
            arrays, scalars = {}, {}
            for arg in plan["args"]:
                name, kind = arg["name"], arg["kind"]
                if kind in ("input", "output"):
                    shape = [int(x) for x in arg["shape"]]
                    n = int(np.prod(shape))
                    dtype = DTYPE_TO_NP[arg["dtype"]]
                    if kind == "input":
                        data = shared_inputs[name].astype(dtype, copy=True)
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
    
            # append warmup/repeat to signature
            argtypes += [ctypes.c_int32, ctypes.c_int32]
            call_args += [ctypes.c_int32(int(warmup)), ctypes.c_int32(int(repeat))]
            fn.argtypes = argtypes
    
            # outer repeats to get distribution (each call returns avg over `repeat`)
            times = []
            for _ in range(max(1, int(outer_repeat))):
                ms = float(fn(*call_args))
                times.append(ms)
    
            arr = np.array(times, dtype=np.float64)
            result["kernel_ms"] = {
                "outer_repeat": int(outer_repeat),
                "warmup": int(warmup),
                "repeat": int(repeat),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
            result["status"] = "success"
            return True, result
    
        except Exception as e:
            traceback.print_exc()
            result["status"] = "error"
            result["errors"] = str(e)
            return False, result


# =====================================================
# Main
# =====================================================

def main():
    cpp_ref = r"""
extern "C" void add(float *input1, float *input2, float *output) {
  int dim1 = 1;
  int dim2 = 15;
  int dim3 = 64;

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      for (int k = 0; k < dim3; k++) {
        int index = i * dim2 * dim3 + j * dim3 + k;
        output[index] = input1[index] + input2[index];
      }
    }
  }
}
"""

    cuda_code = r"""
__global__ void add(float *input1, float *input2, float *output) {
    int dim2 = 15;
    int dim3 = 64;

    int j = blockIdx.x;     // Map blockIdx.x to the second dimension (j loop)
    int k = threadIdx.x;    // Map threadIdx.x to the third dimension (k loop)

    if (j < dim2 && k < dim3) {
        int index = (j * dim3) + k;  // Flattened index calculation
        output[index] = input1[index] + input2[index];
    }
} 
"""

    tester = CudaOpUnitTest(cpp_ref)
    ok, info = tester.unit_test(cuda_code)
    print(ok)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
