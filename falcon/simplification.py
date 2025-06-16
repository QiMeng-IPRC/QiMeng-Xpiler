import re

from pycparser import c_ast

from falcon.util import (NodeTransformer, generate_code, parse_code_ast,
                         remove_target_prefix)


class SimplifyConstants(NodeTransformer):
    def __init__(self):
        self.for_loop_map = {}

    def visit_For(self, node):
        if isinstance(node.cond.right, c_ast.Constant):
            self.for_loop_map[node.init.decls[0].name] = node.cond.right.value

        # Check if the conditional expression in the for loop can be
        # simplified.
        if isinstance(node.cond, c_ast.BinaryOp) and node.cond.op == "<":
            if isinstance(node.cond.right, c_ast.BinaryOp):
                # Attempt to fold constant expressions, such as 256 * 1024.
                if isinstance(
                    node.cond.right.left, c_ast.Constant
                ) and isinstance(node.cond.right.right, c_ast.Constant):
                    result = self.visit_BinaryOp(node.cond.right)
                    node.cond.right = result

        # Handle the if statement within the for loop by attempting to merge it
        # into the condition of the for loop.
        if (
            isinstance(node.stmt, c_ast.Compound)
            and len(node.stmt.block_items) == 1
        ):
            if isinstance(node.stmt.block_items[0], c_ast.If):
                if_node = node.stmt.block_items[0]
                if (
                    isinstance(if_node.cond, c_ast.BinaryOp)
                    and if_node.cond.op == "<"
                    and isinstance(if_node.cond.left, c_ast.ID)
                    and if_node.cond.left.name == node.init.decls[0].name
                    and isinstance(if_node.cond.right, c_ast.Constant)
                    and isinstance(node.cond.right, c_ast.Constant)
                    and int(if_node.cond.right.value)
                    <= int(node.cond.right.value)
                ):
                    # Merge the if condition into the for loop.
                    node.cond.right = if_node.cond.right
                    node.stmt = if_node.iftrue

        return self.generic_visit(node)

    def visit_BinaryOp(self, node):
        # First, recursively simplify the left and right child nodes.
        if isinstance(node.left, c_ast.BinaryOp):
            node.left = self.visit_BinaryOp(node.left)
        if isinstance(node.right, c_ast.BinaryOp):
            node.right = self.visit_BinaryOp(node.right)

        # Check if it is a binary operator that can be simplified.
        if node.op in ("*", "+", "-", "/"):
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                left_val = int(node.left.value)
                right_val = int(node.right.value)
                if node.op == "*":
                    result = left_val * right_val
                elif node.op == "+":
                    result = left_val + right_val
                elif node.op == "-":
                    result = left_val - right_val
                elif node.op == "/":
                    result = (
                        left_val // right_val if right_val != 0 else 0
                    )  # Avoid division by zero.
                return c_ast.Constant("int", value=str(result))

        return self.generic_visit(node)

    def visit_If(self, node):
        # Visit the if statement and try to check if it can be optimized.
        if (
            isinstance(node.cond, c_ast.BinaryOp)
            and node.cond.op == "<"
            and isinstance(node.cond.left, c_ast.ID)
            and node.cond.left.name in self.for_loop_map
            and self.for_loop_map[node.cond.left.name] == node.cond.right.value
        ):
            return node.iftrue.block_items
        return self.generic_visit(node)

    def visit_Cast(self, node):
        # Attempt to remove unnecessary type conversions.
        if (
            isinstance(node.to_type.type.type, c_ast.IdentifierType)
            and node.to_type.type.type.names[0] == "int"
            and isinstance(node.expr, c_ast.ID)
        ):
            return node.expr
        return self.generic_visit(node)


def simplify_code(source_code):
    # Remove all C/C++ style comments.
    source_code = re.sub(r"//.*?\n|/\*.*?\*/", "", source_code, flags=re.S)
    source_code = remove_target_prefix(source_code)
    # Analyze C code
    ast = parse_code_ast(source_code)
    # Create a custom accessor instance.
    visitor = SimplifyConstants()
    # Visit the AST for constant folding.
    visitor.visit(ast)
    # Generate simplified C code.
    return generate_code(ast)


if __name__ == "__main__":
    c_code = """
    void add(float *A, float *B, float *T_add)
    {
    for (int i_j_fuse = 0; i_j_fuse < 256 * 1024; ++i_j_fuse)
    {
        if (i_j_fuse < 4096)
        {
        T_add[i_j_fuse] = A[i_j_fuse] + B[i_j_fuse];
        }
    }
    }
    """
    code = simplify_code(c_code)
    print(code)

    c_code = """
    void sign(float *input0, float *active_sign_147)
    {
        for (int clusterId = 0; clusterId < 4; ++clusterId)
        {
            for (int coreId = 0; coreId < 4; ++coreId)
            {
                float input0_local_nram[25];
                for (int i0_outer_outer_outer = 0; i0_outer_outer_outer < 3; ++i0_outer_outer_outer)
                {
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        int src_offset = ((i0_outer_outer_outer * 400) + (((int) clusterId) * 100)) + (((int) coreId) * 25);
                        int dst_offset = 0;
                        for (int i = 0; i < 25; ++i)
                        {
                            input0_local_nram[dst_offset + i] = input0[src_offset + i];
                        }
                    }
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        // Detensorizing the __bang_active_sign
                        for (int i = 0; i < 25; ++i)
                        {
                            if (input0_local_nram[i] >= 0)
                                input0_local_nram[i] = 1.0f;
                            else
                                input0_local_nram[i] = -1.0f;
                        }
                    }
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        int src_offset = 0;
                        int dst_offset = ((i0_outer_outer_outer * 400) + (((int) clusterId) * 100)) + (((int) coreId) * 25);
                        for (int i = 0; i < 25; ++i)
                        {
                            active_sign_147[dst_offset + i] = input0_local_nram[src_offset + i];
                        }
                    }
                }
            }
        }
    }
    """
    code = simplify_code(c_code)
    print(code)

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
    for (int blockIdxx_threadIdxx_fused = 0; blockIdxx_threadIdxx_fused < 262144; ++blockIdxx_threadIdxx_fused)
    {
        if (blockIdxx_threadIdxx_fused < 4096)
        {
        T_add[blockIdxx_threadIdxx_fused] = A[blockIdxx_threadIdxx_fused] + B[blockIdxx_threadIdxx_fused];
        }
    }
    }
    """
    code = simplify_code(c_code)
    print(code)

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
    for (int blockIdxx = 0; blockIdxx < 256; ++blockIdxx)
    {
        for (int threadIdxx = 0; threadIdxx < 1024; ++threadIdxx)
        {
        if (((blockIdxx * 1024) + threadIdxx) < 4096)
        {
            T_add[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] = A[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] + B[(((int) blockIdxx) * 1024) + ((int) threadIdxx)];
        }
        }

    }
    }
    """
    code = simplify_code(c_code)
    print(code)

    cuda_code = """
    softmax(float* A, float* T_softmax_norm) {
        if (threadIdx.x < 5) {
            int rowStart = threadIdx.x * 128;

            float maxVal = A[rowStart];
            for (int i = 1; i < 128; ++i) {
                if (A[rowStart + i] > maxVal) {
                    maxVal = A[rowStart + i];
                }
            }

            float denom = 0.0f;
            for (int i = 0; i < 128; ++i) {
                T_softmax_norm[rowStart + i] = expf(A[rowStart + i] - maxVal);
                denom += T_softmax_norm[rowStart + i];
            }

            for (int i = 0; i < 128; ++i) {
                T_softmax_norm[rowStart + i] /= denom;
            }
        }
    }
    """
    converted_code = simplify_code(cuda_code)
    print(converted_code)
