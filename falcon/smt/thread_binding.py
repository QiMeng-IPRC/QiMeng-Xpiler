from pycparser import c_ast

from falcon.util import (NodeTransformer, generate_code, make_full_func,
                         parse_code_ast)

builtin_var = {
    "cuda": ["threadIdxx", "blockIdxx"],
    "mlu": ["coreId", "clusterId"],
    "hip": ["threadIdxx", "blockIdxx"],
}
builtin_dim = {
    "threadIdxx": 256,
    "blockIdxx": 1024,
    "coreId": 4,
    "clusterId": 12,
}


# Temporarily, we will binding the outermost with thread
class ThreadBindingTransformer(NodeTransformer):
    def __init__(self, parallel_loops, target="mlu"):
        self.binding_map = {}
        self.parallel_loops = parallel_loops
        self.target = target
        self.current_depth = 0

    def visit_For(self, node):
        self.current_depth += 1
        loop_var = (
            node.init.decls[0].name
            if isinstance(node.init, c_ast.DeclList)
            else None
        )
        extend = int(node.cond.right.value)

        # If the loop variable is a bound variable, return the loop body.
        if (
            self.target == "mlu"
            and self.parallel_loops >= 2
            and self.current_depth == 1
        ):
            thread_var = self._generate_thread_var(extend, 4)
            new_node = self._generate_new_node(thread_var, node)
            self.binding_map[loop_var] = thread_var
            return self.generic_visit(new_node)

        elif (
            self.target == "cuda"
            or self.target == "hip"
            and self.current_depth == 1
        ):
            thread_var = self._generate_thread_var(extend, 1024)
            new_node = self._generate_new_node(thread_var, node)
            self.binding_map[loop_var] = thread_var
            return self.generic_visit(new_node)

        return self.generic_visit(node)

    def _generate_thread_var(self, extend, limit):
        if extend <= limit:
            return c_ast.ID(name=builtin_var[self.target][0])
        else:
            return c_ast.BinaryOp(
                op="+",
                left=c_ast.BinaryOp(
                    op="*",
                    left=c_ast.ID(name=builtin_var[self.target][1]),
                    right=c_ast.Constant("int", value=str(limit)),
                ),
                right=c_ast.ID(name=builtin_var[self.target][0]),
            )

    def _generate_new_node(self, thread_var, node):
        return c_ast.If(
            cond=c_ast.BinaryOp(
                op="<", left=thread_var, right=node.cond.right
            ),
            iftrue=node.stmt,
            iffalse=None,
        )

    def visit_ID(self, node):
        return self.binding_map.get(node.name, node)


class LoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.current_depth = 0  # Current nesting depth
        self.max_depth = 0  # Maximum nesting depth

    def visit_For(self, node):
        # Each time a for loop is encountered, the nesting depth increases by
        # 1.
        self.current_depth += 1
        # Update maximum nesting depth.
        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth

        # Access child nodes
        self.generic_visit(node)

        # When exiting the for loop, decrease the nesting depth by 1.
        self.current_depth -= 1


def ast_thread_binding(code, target="mlu"):
    # Analytical code
    ast = parse_code_ast(code)

    # Count the number of loop layers.
    loop_visitor = LoopVisitor()
    loop_visitor.visit(ast)
    # Perform thread-bound conversion.
    transformer = ThreadBindingTransformer(loop_visitor.max_depth, target)
    ast = transformer.visit(ast)
    # Output the modified code.
    binding_code = generate_code(ast)

    return make_full_func(binding_code, target)


if __name__ == "__main__":
    # Sample code
    code = """
    void func() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 7; ++k) {
                    B[i * 4 * 7 + j * 7 + k] = A[i * 4 * 7 + j * 7 + k] + 1.0;
                }
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="mlu")
    print(output_code)

    code = """
    void func() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 7; ++k) {
                    B[i * 4 * 7 + j * 7 + k] = A[i * 4 * 7 + j * 7 + k] + 1.0;
                }
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="cuda")
    print(output_code)

    code = """
    void softmax(float *A, float *T_softmax_norm)
    {
        for (int k = 0; k < 5; ++k)
        {
            float maxVal = A[k * 128];
            for (int j = 1; j < 128; ++j)
            {
                if (A[(k * 128) + j] > maxVal)
                {
                    maxVal = A[(k * 128) + j];
                }
            }

            float denom = 0.0f;
            for (int j = 0; j < 128; ++j)
            {
                T_softmax_norm[(k * 128) + j] = expf(A[(k * 128) + j] - maxVal);
            }

            for (int j = 0; j < 128; ++j)
            {
                denom += T_softmax_norm[(k * 128) + j];
            }

            for (int j = 0; j < 128; ++j)
            {
                T_softmax_norm[(k * 128) + j] /= denom;
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="cuda")
    print(output_code)

    code = """
    void softmax(float *A, float *T_softmax_norm)
    {
        for (int k = 0; k < 5; ++k)
        {
            float maxVal = A[k * 128];
            for (int j = 1; j < 128; ++j)
            {
                if (A[(k * 128) + j] > maxVal)
                {
                    maxVal = A[(k * 128) + j];
                }
            }

            float denom = 0.0f;
            for (int j = 0; j < 128; ++j)
            {
                T_softmax_norm[(k * 128) + j] = expf(A[(k * 128) + j] - maxVal);
            }

            for (int j = 0; j < 128; ++j)
            {
                denom += T_softmax_norm[(k * 128) + j];
            }

            for (int j = 0; j < 128; ++j)
            {
                T_softmax_norm[(k * 128) + j] /= denom;
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="mlu")
    print(output_code)

    code = """
    void add(float *A, float *B, float *T_add)
    {
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 1024; j++) {
            T_add[k * 1024 + j] = A[k * 1024 + j] + B[k * 1024 + j];
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="mlu")
    print(output_code)
