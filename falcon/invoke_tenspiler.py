import re

from pycparser import c_ast, c_parser

# --- Utility Function to Strip Comments ---


def strip_comments(code):
    """Removes C-style and C++-style comments from a string."""
    code = re.sub(r"//.*", "", code)

    def remove_c_style_comment(match):
        return ""

    code = re.sub(r"/\*.*?\*/", remove_c_style_comment, code, flags=re.DOTALL)
    return code.strip()


# --- 1. AST Visitor (No change) ---


class LoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.found_for_loop = False
        self.core_expression = ""

    def visit_For(self, node):
        self.found_for_loop = True
        if isinstance(node.stmt, c_ast.Compound):
            for inner_node in node.stmt.block_items:
                if isinstance(inner_node, c_ast.Assignment) or isinstance(
                    inner_node, c_ast.UnaryOp
                ):
                    if not self.core_expression:
                        self.core_expression = self._get_expression_string(
                            inner_node
                        )

    def _get_expression_string(self, node):
        if isinstance(node, c_ast.ID):
            return node.name
        elif isinstance(node, c_ast.Constant):
            return node.value
        elif isinstance(node, c_ast.ArrayRef):
            return f"{self._get_expression_string(node.name)}[{self._get_expression_string(node.subscript)}]"
        elif isinstance(node, c_ast.BinaryOp):
            return f"({self._get_expression_string(node.left)} {node.op} {self._get_expression_string(node.right)})"
        elif isinstance(node, c_ast.Assignment):
            return f"{self._get_expression_string(node.lvalue)} {node.op} {self._get_expression_string(node.rvalue)}"
        elif isinstance(node, c_ast.FuncCall):
            return f"{self._get_expression_string(node.name)}(...)"
        else:
            return ""


# --- 2. Core Conversion Logic (Modified) ---


def generate_tenspiler_cc_file_pycparser(input_code_snippet):
    # Step A: Extract Pragma information
    pragma_match = re.search(
        r"#\s*pragma\s*operation\((.*?)\)", input_code_snippet, re.DOTALL
    )

    if not pragma_match:
        # print("Error: #pragma operation(...) directive not found.")
        return None, None, None, None  # Return None for all new fields

    pragma_content = pragma_match.group(1).strip()

    # Parse Pragma Content
    op_name_match = re.match(r"(\w+)\s*\(", pragma_content)
    operation_name = (
        op_name_match.group(1).strip() if op_name_match else "unknown_op"
    )
    input_match = re.search(r"input\s*\[\s*(.*?)\s*\]", pragma_content)
    input_tensors = (
        [t.strip() for t in input_match.group(1).split(",")]
        if input_match
        else []
    )
    output_match = re.search(r"output\s*\[\s*(.*?)\s*\]", pragma_content)
    output_tensors = (
        [t.strip() for t in output_match.group(1).split(",")]
        if output_match
        else []
    )

    # --- Code Preprocessing and AST Parsing (Steps B & C) ---
    clean_code = strip_comments(input_code_snippet)
    preprocessed_code = re.sub(r"#\s*pragma.*", "", clean_code)
    preprocessed_code = re.sub(
        r"\s*([0-9]+\.[0-9]+)f", r"\1", preprocessed_code
    )
    preprocessed_code = preprocessed_code.replace("++", "+= 1")
    preprocessed_code = re.sub(r"\b__nram__\b", "", preprocessed_code)
    preprocessed_code = re.sub(r"\b__wram__\b", "", preprocessed_code)
    preprocessed_code = re.sub(r"\b__gram__\b", "", preprocessed_code)

    parser = c_parser.CParser()
    try:
        ast = parser.parse(preprocessed_code)
    except Exception as e:
        print(f"Error: pycparser failed to parse the code.\nDetails: {e}")
        return None, None, None, None

    visitor = LoopVisitor()
    visitor.visit(ast)

    # --- Code Generation (Step D) ---
    op_name_lower = operation_name.lower()

    # Generate C++ vector code (omitted details, same as before)
    param_list = [f"std::vector<float> {t_name}" for t_name in input_tensors]

    t_x = input_tensors[0] if len(input_tensors) >= 1 else "tensor_x"
    t_y = input_tensors[1] if len(input_tensors) >= 2 else "tensor_y"
    t_out = output_tensors[0] if len(output_tensors) >= 1 else "out"

    # Simplified Function Body Generation for Tenspiler input
    if op_name_lower == "matmul":
        func_body = f"""
            std::vector<float> {t_out};
            int m = {t_x}.size();
            for (int i = 0; i < m; i++) {{
                {t_out}.push_back({t_x}[i] * {t_y}[i]);
            }}
            return {t_out};
        """
    elif op_name_lower in ["copy", "memory"]:
        # === FIX: Ensure the output variable is declared and filled from the input ===
        func_body = f"""
            std::vector<float> {t_out};
            int m = {t_x}.size();
            for (int i = 0; i < m; i++) {{
                {t_out}.push_back({t_x}[i]);
            }}
            return {t_out};
        """
    else:
        func_body = f"""std::vector<float> {t_out}; return {t_out};"""

    cpp_code = f"""#include <vector>\nusing namespace std;\n// ORIGINAL_PRAGMA: # pragma operation({pragma_content})\nvector<float> {op_name_lower}({", ".join(param_list)}) {{{func_body}}}"""

    # Return all necessary data for the stitching step
    return operation_name, input_tensors, output_tensors, cpp_code.strip()


# --- New Function: Tensor Intrinsic Generation (Simulation) ---


def generate_tensor_intrinsic(op_name, input_tensors, output_tensors):
    """Simulates the Tenspiler output: a hardware-specific tensor intrinsic call."""
    op_lower = op_name.lower()

    if op_lower == "matmul":
        if len(input_tensors) >= 2 and len(output_tensors) >= 1:
            return f"__matmul({output_tensors[0]}, {input_tensors[0]}, {input_tensors[1]});"
        else:
            return "// ERROR: MATMUL tensors missing."

    elif op_lower in ["copy", "memory"]:
        # Format: __memcpy(Dest, Src, size);
        if len(input_tensors) >= 1 and len(output_tensors) >= 1:
            return f"__memcpy({output_tensors[0]}, {input_tensors[0]}, TENSOR_SIZE);"
        else:
            return "// ERROR: MEMORY tensors missing."

    else:
        return f"// WARNING: Tensor Op '{op_name}' not defined."


# --- New Function: Stitching (Replacement) ---


def stitch_code(original_source, tensor_intrinsic):
    """Replaces the Pragma and serial loop block with the tensor intrinsic."""
    pattern = re.compile(
        r"(#\s*pragma\s*operation\(.*?\)\s*)"  # Group 1: Pragma line
        r"(\s*(//.*?\n)*\s*)?"  # Optional: Comments/whitespace
        r"(for\s*\(.*?\)\s*\{"  # Start of outer loop
        r".*?"  # Content (lazy match)
        # End of outer loop (匹配到 } 并包含其后的空格/换行符)
        r"\}\s*)",
        re.DOTALL,
    )

    func_match = re.search(
        r"(\w+\s+\w+\s*\([^)]*\)\s*\{)(.*?)(\}\s*)$",
        original_source,
        re.DOTALL,
    )

    if func_match:
        func_start_code = func_match.group(1)  # e.g., 'void func() {'
        func_body_code = func_match.group(2)  # The content inside the function

        replacement = "    " + tensor_intrinsic

        new_body = pattern.sub(replacement, func_body_code, count=1)

        final_code = func_start_code + new_body.strip() + "\n}"
        return final_code.strip()
    else:
        print(
            "-> Error: Could not parse function structure for safe stitching."
        )
        # Fallback to the original less safe replacement
        replacement = "    " + tensor_intrinsic
        return pattern.sub(replacement, original_source, count=1).strip()


# ----------------------------------------------
# --- Main Execution Flow ---
# ----------------------------------------------


def process_and_stitch_tensor_code(input_code):
    # Step 1: Analyze and Generate Tenspiler Input (File Generation Phase)
    op_name, inputs, outputs, cc_code = generate_tenspiler_cc_file_pycparser(
        input_code
    )

    if not cc_code:
        return None, None, None

    # Step 2: Simulate Tenspiler Output (Intrinsic Generation Phase)
    tensor_intrinsic = generate_tensor_intrinsic(op_name, inputs, outputs)

    # Step 3: Stitch the Intrinsic back into the Original Code
    stitched_code = stitch_code(input_code, tensor_intrinsic)

    return cc_code, tensor_intrinsic, stitched_code


if __name__ == "__main__":

    # -----------------------------------------------------------------
    # --- Example 1: MatMul (Matrix Multiplication) ---
    # -----------------------------------------------------------------
    matmul_full_code = """
    void matmul_kernel_fragment() {
        float A_nram[512];
        float B_wram[32768];
        float C_nram[64];
        int clusterId, coreId;

        # pragma operation(matmul(input[A_nram, B_wram], output[C_nram]))
        for (int col=0; col < 64; col ++) {
            C_nram[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
            for (int i=0; i < 512; i ++) {
                C_nram[col] += A_nram[i] * B_wram[i * 64 + col];
            }
        }
    }
    """

    print("=== 1. Processing MatMul Code (Serial -> Tensorized) ===")
    cc_code_m, intrinsic_code_m, stitched_code_m = (
        process_and_stitch_tensor_code(matmul_full_code)
    )

    if stitched_code_m:
        print("\n[MatMul Tenspiler Input File (.cc)]")
        print("-" * 40)
        print(cc_code_m)
        print("-" * 40)

        print("\n[MatMul Final Stitched Kernel Code]")
        print("The serial loop is replaced by the intrinsic.")
        print("=" * 40)
        print(stitched_code_m)
        print("=" * 40)

    print("\n\n" + "#" * 50 + "\n\n")

    # -----------------------------------------------------------------
    # --- Example 2: Memory Copy (memcpy) ---
    # --- FIX APPLIED: Corrected func_body logic in generate_tenspiler_cc_file_pycparser ---
    # -----------------------------------------------------------------
    memory_full_code = """
    void memory_kernel_fragment() {
        float C_global[32768]; // Destination (Output)
        float C_nram[64];      // Source (Input)
        int clusterId, coreId;

        # pragma operation(memory(input[C_nram], output[C_global]))
        for (int col=0; col < 64; col ++) {
            C_global[(clusterId * 4 + coreId) * 64 + col] = C_nram[col];
        }
    }
    """

    print("=== 2. Processing Memory Copy Code (Serial -> Tensorized) ===")
    cc_code_mem, intrinsic_code_mem, stitched_code_mem = (
        process_and_stitch_tensor_code(memory_full_code)
    )

    if stitched_code_mem:
        print("\n[Memory Copy Tenspiler Input File (.cc)]")
        print("-" * 40)
        print(cc_code_mem)
        print("-" * 40)

        print("\n[Memory Copy Final Stitched Kernel Code]")
        print("The serial loop is replaced by the intrinsic.")
        print("=" * 40)
        print(stitched_code_mem)
        print("=" * 40)
