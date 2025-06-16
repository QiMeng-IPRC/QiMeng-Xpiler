import os
import re

import openai

model_name = """gpt-4-turbo"""
api_key = os.getenv("OPENAI_API_KEY")

inline_prompt = """

```
You are an expert in C code optimization and transformation. Your task is to receive a piece of C code and perform "Buffer Inlining" optimization on it.

**Optimization Goal:**
Identify and eliminate local intermediate buffers (typically fixed-size arrays, like `float buffer[...]`) used for temporary storage of element-wise or small-neighborhood calculation results within function bodies.

**Patterns to Recognize:**
This pattern typically appears inside loops, where data is read from one (or more) input pointers into a local buffer, operated on within the buffer, and then written from the buffer to an output pointer. For example:

1.  **Simple Operation (e.g., tanh):**
    * Read input to buffer: `buffer[i] = input[offset + i];`
    * Operate on buffer element: `buffer[i] = operation(buffer[i]);`
    * Write result from buffer to output: `output[offset + i] = buffer[i];`
    * **Expected Inlined Result:** `output[offset + i] = operation(input[offset + i]);`

2.  **Multi-input Operation (e.g., Add):**
    * Read input 1 to a part of the buffer: `buffer[index1] = input1[offset1 + i];`
    * Read input 2 to another part of the buffer: `buffer[index2] = input2[offset2 + i];`
    * Operate within the buffer: `buffer[index3] = operation(buffer[index4], buffer[index5]);` (e.g., `buffer[i] += buffer[i+constant]`)
    * Write result from buffer to output: `output[offset3 + i] = buffer[index6];`
    * **Expected Inlined Result:** `output[offset3 + i] = operation(input1[offset1 + i], input2[offset2 + i]);` (Indices and offsets here need to correspond correctly)

**Specific Examples:**

Please strictly follow the transformation relationship between the original and expected inlined code in the examples below. Pay special attention to how the loop variable `i`, pointer type casting `(float *)`, and offset calculations `(((int) clusterId) * ...) + (((int) coreId) * ...)` are preserved and combined in the transformed code.

**Example 1 (tanh):**

* **Original Code:**
    ```c
    void tanh_op(float *input0, float *active_tanh_210)
    {
    for (int clusterId = 0; clusterId < 4; ++clusterId)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
        float input0_local_nram[640]; // Local buffer
        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = (((float *) input0) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i];
            (((float *) input0_local_nram) + 0)[i] = tanh((((float *) input0_local_nram) + 0)[i]);
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i] = (((float *) input0_local_nram) + 0)[i];
        }
        }
    }
    }
    ```

* **Expected Inlined Code:**
    ```c
    void tanh_op(float *input0, float *active_tanh_210)
    {
    for (int clusterId = 0; clusterId < 4; ++clusterId)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
        // float input0_local_nram[640]; // Removed
        for (int i = 0; i < 640; i++)
        {
            // Compute tanh directly and write to output
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i] = tanh((((float *) input0) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i]);
        }
        }
    }
    }
    ```

**Example 2 (add):**

* **Original Code:**
    ```c
    void add_op(float *lhs, float *rhs, float *add_1605)
    {
        float lhs_local_nram[512]; // Local buffer
        if (((clusterId * 4) + coreId) < 9)
        {
            for (int i = 0; i < 256; ++i)
            {
            lhs_local_nram[i] = lhs[((clusterId * 1024) + (coreId * 256)) + i];
            lhs_local_nram[256 + i] = rhs[((clusterId * 1024) + (coreId * 256)) + i];
            lhs_local_nram[i] += lhs_local_nram[256 + i]; // Operation within buffer
            add_1605[((clusterId * 1024) + (coreId * 256)) + i] = lhs_local_nram[i];
            }
        }
    }
    ```

* **Expected Inlined Code:**
    ```c
    void add_op(float *lhs, float *rhs, float *add_1605)
    {
        // float lhs_local_nram[512]; // Removed
        if (((clusterId * 4) + coreId) < 9)
        {
            for (int i = 0; i < 256; ++i)
            {
                // Add lhs and rhs elements directly and write to add_1605
                add_1605[((clusterId * 1024) + (coreId * 256)) + i] = lhs[((clusterId * 1024) + (coreId * 256)) + i] + rhs[((clusterId * 1024) + (coreId * 256)) + i];
            }
        }
    }
    ```

**Your Task Instructions:**

When you receive a piece of C code, please:

1.  Carefully analyze the function definitions within the code.
2.  Look for local array declarations (especially float arrays) which might be intermediate buffers.
3.  Inside loops, look for sequences of statements that match the "Patterns to Recognize" described above, i.e., reading from input pointers into a local buffer, operating on elements indexed by the loop variable within the buffer, and then writing the result from the buffer back to an output pointer.
4.  If a pattern matching these sequences is found, replace them with a single statement that reads directly from the input pointer(s), performs the operation, and writes directly to the output pointer.
5.  Remove the declaration of the corresponding local buffer variable.
6.  Keep all code that does not match the pattern or is outside the pattern (including outer loops, conditional statements, other variable declarations, etc.) unchanged.
7.  **Output ONLY** the complete modified C code. Do not include any explanatory text, Markdown formatting (except for the code block), or additional notes. If no optimization opportunities matching the pattern are found in the code, output the original code as is.

**Now, please apply the buffer inlining optimization to the following C code:**
{input_code}
"""


def ast_buffer_inline(code):
    inline_prompt_filled = inline_prompt.replace("{input_code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": inline_prompt_filled}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


if __name__ == "__main__":

    # Enter code
    code = """
    void tanh(float *input0, float *active_tanh_210)
    {
    for (int clusterId = 0; clusterId < 4; ++clusterId)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
        float input0_local_nram[640];
        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = (((float *) input0) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i];
            (((float *) input0_local_nram) + 0)[i] = tanh((((float *) input0_local_nram) + 0)[i]);
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i] = (((float *) input0_local_nram) + 0)[i];
        }
        }
    }
    }
    """
    code = ast_buffer_inline(code)
    print(code)

    code = """
    void add(float *lhs, float *rhs, float *add_1605)
    {
        float lhs_local_nram[512];
        if (((clusterId * 4) + coreId) < 9)
        {
            for (int i = 0; i < 256; ++i)
            {
            lhs_local_nram[i] = lhs[((clusterId * 1024) + (coreId * 256)) + i];
            lhs_local_nram[256 + i] = rhs[((clusterId * 1024) + (coreId * 256)) + i];
            lhs_local_nram[i] += lhs_local_nram[256 + i];
            add_1605[((clusterId * 1024) + (coreId * 256)) + i] = lhs_local_nram[i];
            }

        }
    }
    """
    code = ast_buffer_inline(code)
    print(code)
