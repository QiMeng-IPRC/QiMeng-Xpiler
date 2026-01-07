CPP_INPUT_PROMPT="""
你是一个严谨的编译器与静态分析助手。
给定一段 C/C++ 代码，请生成一个 Python ctypes 可直接执行的“调用计划”。

请严格遵守以下规则：
1. 只输出一个 JSON 对象，不要输出任何解释、空行或额外文字。
2. 选择最合适的入口函数，通常是包含指针参数并执行主要计算的函数。
3. JSON 中必须包含字段 "function" 和 "args"。
4. "args" 必须按函数参数的原始顺序给出。
5. 对每个参数进行分类：
   - kind = "input"：只读指针数组
   - kind = "output"：被写入的指针数组
   - kind = "scalar"：标量参数（int / float / size_t 等）
6. 对 input / output 参数，必须给出：
   - dtype: 只能是 "float32" / "float64" / "int32" / "int64"
   - shape: 具体整数列表，如 [1, 16, 64]
7. shape 的推断规则：
   - 优先使用代码中出现的常量（如 dim1, dim2, dim3）
   - 如果通过 index = i*A + j*B + k 的形式访问，shape 应与循环边界一致
   - 即使参数在函数签名中是一维指针，也请给出多维 shape
   - 如果无法精确推断，请给出一个小而合理的 shape（如 [1, 16, 64]）
8. 对 scalar 参数，必须给出：
   - dtype
   - value（具体数值，且应与 shape 保持一致）
9. 所有 shape 和 value 必须是具体数值，不能使用符号或变量名。
10. 不要编造代码中不存在的参数或函数。

输出 JSON 的严格格式如下（字段名必须一致）：

{
  "function": "function_name",
  "args": [
    {
      "name": "param_name",
      "kind": "input",
      "dtype": "float32",
      "shape": [1, 16, 64],
      "ctype": "float*"
    },
    {
      "name": "param_name",
      "kind": "output",
      "dtype": "float32",
      "shape": [1, 16, 64],
      "ctype": "float*"
    },
    {
      "name": "param_name",
      "kind": "scalar",
      "dtype": "int32",
      "value": 1024,
      "ctype": "int"
    }
  ]
}

下面是需要分析的 C/C++ 代码：
```cpp
{{CODE}}
```
请只输出 JSON
"""

CUDA_INPUT_PROMPT = r"""
You are generating a STRICT JSON plan for calling CUDA code via ctypes from Python.

You are given:
- CUDA source code: {{CODE}}
- A reference CPU plan JSON: {{CPP_PLAN_JSON}}

You MUST output ONLY one JSON object (no markdown, no comments, no extra text).

Hard requirements:
1) The output shapes and dtypes MUST match the reference CPU plan exactly.
   - For every input/output tensor in the CPU plan, copy its "name", "dtype", and "shape" (ALL dims must be concrete integers).
2) ctypes can only call an exported HOST symbol.
   - If the CUDA code does not already define an exported host function, you MUST generate a host wrapper plan.
   - The host wrapper name MUST be "<kernel_name>_kernel" and MUST be exported with `extern "C"`.
3) The plan["function"] MUST be the host wrapper name (NOT the __global__ kernel).
4) The plan["args"] MUST be ordered as:
   - pointer args (all inputs then outputs) matching the CPU plan order
   - followed by any scalar args needed by the host wrapper (if any).
5) Shapes MUST be fully concrete integers. DO NOT use symbols like "size1".

JSON schema:
{
  "function": "<kernel_name>_kernel",
  "kernel_name": "<__global__ kernel name>",
  "launch": { "block": [int,int,int], "grid": [int,int,int] },
  "args": [
    {"name": "...", "kind": "input",  "dtype": "float32|float64|int32|int64", "shape": [int,...]},
    {"name": "...", "kind": "output", "dtype": "...", "shape": [int,...]},

    // scalar args only if the host wrapper needs them, each must have a concrete value:
    {"name": "...", "kind": "scalar", "dtype": "int32|int64", "value": int}
  ]
}

How to use CPP_PLAN_JSON:
- Parse CPP_PLAN_JSON and reuse its tensor args directly.
- If CUDA code uses different parameter names, you MUST still follow the CPU plan names in args
  (the host wrapper you generate can map names internally).
- If CPU plan has K pointer args, your CUDA plan must have exactly K pointer args with same names and shapes.

Kernel selection:
- Choose the primary __global__ kernel in the CUDA code (the one that writes outputs).
- Set kernel_name accordingly.

Launch config rules:
- Conservative default is ok; correctness is priority.
- If CPU output is 1D length L: use block [256,1,1], grid [ceil_div(L,256),1,1].
- If CPU output is 2D [H,W]: use block [16,16,1], grid [ceil_div(W,16), ceil_div(H,16), 1].
- If unclear, choose grid [1,1,1] and block [256,1,1]; wrapper correctness still must hold.
- SPECIAL CASE (critical for correctness):
  If the kernel uses threadIdx.x but does NOT use blockIdx.x anywhere,
  then the effective index is per-block only and grid.x MUST be 1 (otherwise outputs will be overwritten).
  In this case:
    * Determine TOTAL = total number of output elements (product of output shape from CPP_PLAN_JSON).
    * Set launch.block = [TOTAL, 1, 1] if TOTAL <= 1024, else [1024,1,1] and mark plan as "unsupported" (do not guess).
    * Set launch.grid  = [1, 1, 1].



Output must be valid JSON and parseable.
"""