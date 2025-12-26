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

CUDA_INPUT_PROMPT="""
你是一个严谨的编译器与静态分析助手。

已知 C++ 代码的调用计划如下（JSON 格式）：
{{CPP_PLAN_JSON}}

现在给定一段 CUDA C/C++ 代码：
```cpp
{{CODE}}
请生成一个 Python ctypes 可直接执行的“调用计划”，要求输入输出参数名称和上面 C++ plan 完全一致（即 input/output 名称必须与 C++ plan 对应），如果 CUDA 多了 scalar 参数，可以直接添加，不需要改输入输出名称。

规则：

只输出一个 JSON 对象，不要输出任何解释、空行或额外文字。

选择最合适的入口函数（通常是包含指针参数并执行主要计算的函数）。

JSON 中必须包含字段 "function" 和 "args"。

"args" 按函数参数原始顺序给出。

对每个参数分类：

kind = "input"：只读指针数组

kind = "output"：被写入的指针数组

kind = "scalar"：标量参数（int / float / size_t 等）

对 input / output 参数：

dtype: "float32"/"float64"/"int32"/"int64"

shape: 尽量与 C++ plan 保持一致，如果需要 reshape 可调整，但名字必须一致

对 scalar 参数：

dtype

value

所有 shape 和 value 必须是具体数值

不要修改 C++ plan 中的输入输出名字

输出 JSON 严格格式如下：

{
"function": "function_name",
"args": [
{
"name": "param_name", # 必须与 C++ plan 对齐
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
"name": "scalar_param",
"kind": "scalar",
"dtype": "int32",
"value": 1024,
"ctype": "int"
}
]
}

请只输出 JSON
"""
