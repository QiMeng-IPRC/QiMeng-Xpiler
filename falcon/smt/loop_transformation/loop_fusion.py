from pycparser import c_ast

from falcon.simplification import simplify_code
from falcon.util import NodeTransformer, generate_code, parse_code_ast


class LoopNestFusionVisitor(NodeTransformer):
    def __init__(self):
        self.outer_var = None
        self.inner_var = None
        self.inner_bound = None

    def visit_For(self, node):
        # Check whether the current for loop contains nested for loops
        if (
            isinstance(node.stmt, c_ast.Compound)
            and len(node.stmt.block_items) == 1
        ):
            nested_loop = node.stmt.block_items[0]
            if isinstance(nested_loop, c_ast.For) and isinstance(
                node.next, c_ast.UnaryOp
            ):
                # Merge nested for loops
                fused_loop = self.fuse_loops(node, nested_loop)
                node.init = fused_loop.init
                node.cond = fused_loop.cond
                node.next = fused_loop.next
                node.stmt = fused_loop.stmt

                self.outer_var = node.init.decls[0].name
                self.inner_var = nested_loop.init.decls[0].name
                self.inner_bound = nested_loop.cond.right.value

                # Update the initialization part, and change the variable name
                # to i_j_fused
                fused_name = f"{node.init.decls[0].name}_{nested_loop.init.decls[0].name}_fused"
                node.init.decls[0].name = fused_name
                node.init.decls[0].type.declname = fused_name
                node.cond.left.name = fused_name
                node.next.expr.name = fused_name
                return self.generic_visit(node)

        return self.generic_visit(node)

    def visit_BinaryOp(self, node):
        if node.op == "+":
            left = node.left
            right = node.right
            if (
                isinstance(right, c_ast.ID)
                and right.name == self.inner_var
                and isinstance(left, c_ast.BinaryOp)
                and left.op == "*"
                and isinstance(left.right, c_ast.Constant)
                and left.right.value == self.inner_bound
                and isinstance(left.left, c_ast.ID)
                and left.left.name == self.outer_var
            ):
                # Replace with a fused variable
                fused_index = c_ast.ID(
                    name=f"{self.outer_var}_{self.inner_var}_fused"
                )
                return fused_index
        return self.generic_visit(node)

    def visit_ArrayRef(self, node):
        if (
            isinstance(node.name, c_ast.BinaryOp)
            and node.subscript.name == self.inner_var
        ):
            right = node.name.right
            if (
                isinstance(right, c_ast.BinaryOp)
                and right.left.expr.name == self.outer_var
                and right.op == "*"
                and isinstance(right.right, c_ast.Constant)
                and right.right.value == self.inner_bound
            ):
                fused_index = c_ast.ID(
                    name=f"{self.outer_var}_{self.inner_var}_fused"
                )
                node.subscript = fused_index
                node.name = node.name.left
            return node
        return self.generic_visit(node)

    def fuse_loops(self, outer_loop, inner_loop):
        # Create new loop variables and boundaries, and merge the two loops.
        fused_init = outer_loop.init
        fused_cond = c_ast.BinaryOp(
            op="<",
            left=c_ast.ID(name=fused_init.decls[0].name),
            right=c_ast.Constant(
                type="int",
                value=str(
                    int(outer_loop.cond.right.value)
                    * int(inner_loop.cond.right.value)
                ),
            ),
        )
        fused_next = outer_loop.next
        # Adjust the body of the nested loop to a single-layer loop.
        fused_stmt = c_ast.Compound(inner_loop.stmt.block_items)
        fused_loop = c_ast.For(
            init=fused_init, cond=fused_cond, next=fused_next, stmt=fused_stmt
        )
        return fused_loop


def ast_loop_fusion(c_code):
    ast = parse_code_ast(c_code)
    # Custom visitor instance
    visitor = LoopNestFusionVisitor()
    # Visit the AST and merge nested for loops where possible.
    visitor.visit(ast)
    code = generate_code(ast)
    code = simplify_code(code)
    return code


if __name__ == "__main__":
    # C code containing nested for loops
    c_code = """
    void multiply_matrices(int* a, int* b, int* result) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                result[i * 10 + j] = a[i * 10 + j] * b[i * 10 + j];
            }
        }
    }
    """
    # Final code after merging loops
    final_code = ast_loop_fusion(c_code)
    print(final_code)

    c_code = """
    void add(float *lhs, float *rhs, float *add_1935)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
            for (int i = 0; i < 1024; i++)
            {
            (((float *) add_1935) + (((int) coreId) * 1024))[i] = (((float *) lhs) + (((int) coreId) * 1024))[i] + (((float *) rhs) + (((int) coreId) * 1024))[i];
            }

        }
    }
    """
    # Final code after merging loops
    final_code = ast_loop_fusion(c_code)
    print(final_code)

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
        for (int blockIdxx_threadIdxx_fused = 0; blockIdxx_threadIdxx_fused < 262144; ++blockIdxx_threadIdxx_fused)
        {
            if (blockIdxx_threadIdxx_fused < 4096)
            {
                T_add[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] = A[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] + B[(((int) blockIdxx) * 1024) + ((int) threadIdxx)];
            }
        }

    }
    """
    # Final code after merging loops
    final_code = ast_loop_fusion(c_code)
    print(final_code)
