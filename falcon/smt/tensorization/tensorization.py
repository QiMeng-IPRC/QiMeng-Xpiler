import re

import numpy as np
from pycparser import c_ast

from falcon.util import (NodeTransformer, generate_code, make_full_func,
                         parse_code_ast)


class PragmaToSIMDTransformer(NodeTransformer):
    def __init__(self):
        self.loop_exts = []
        self.vectorize_var = None

    def visit_Compound(self, node):
        """Find "for loop" in pragma."""
        new_block_items = []
        i = 0
        while i < len(node.block_items):
            stmt = node.block_items[i]

            # Check if it is a #pragma operation.
            if isinstance(stmt, c_ast.Pragma):
                pragma_text = stmt.string
                # Find the next for loop under pragma.
                if i + 1 < len(node.block_items) and isinstance(
                    node.block_items[i + 1], c_ast.For
                ):
                    transformed_stmt = self.transform_pragma_loop(
                        node.block_items[i + 1], pragma_text
                    )
                    new_block_items.append(
                        transformed_stmt
                    )  # Replace the for loop with target instructions.
                    # Skip the next for loop, as it has already been processed.
                    i += 2
                    continue
                else:
                    # Retain the #pragma annotations.
                    new_block_items.append(stmt)
            else:
                # Retain the unprocessed statements.
                new_block_items.append(stmt)

            i += 1

        node.block_items = new_block_items
        return self.generic_visit(node)

    def extract_argments(self, for_node):
        self.vectorize_var = for_node.init.decls[0].name
        self.generic_visit(for_node)
        self.vectorize_var = None

        def get_body(node):
            # Base case: if the node is not a `for` loop, return None
            if not isinstance(node, c_ast.For):
                return None

            # If the `for` loop has a body, check if it's another `for` loop
            body = node.stmt
            if isinstance(body, c_ast.Compound) and body.block_items:
                # Check if the first item in the block is a `for` loop
                first_stmt = body.block_items[0]
                if isinstance(first_stmt, c_ast.For):
                    # Recurse into the inner `for` loop
                    return get_body(first_stmt)
                else:
                    # Return the body if it's not a nested `for` loop
                    return first_stmt

            # If there's no body, return None
            return None

        stmt = get_body(for_node)
        assert isinstance(stmt, c_ast.Assignment)
        assert isinstance(stmt.lvalue, c_ast.ArrayRef)
        args = [
            c_ast.BinaryOp(
                op="+", left=stmt.lvalue.name, right=stmt.lvalue.subscript
            )
        ]
        right = stmt.rvalue
        if isinstance(right, c_ast.ArrayRef):
            args.append(
                c_ast.BinaryOp(op="+", left=right.name, right=right.subscript)
            )
        elif isinstance(right, c_ast.BinaryOp):
            args.append(
                c_ast.BinaryOp(
                    op="+", left=right.left.name, right=right.left.subscript
                )
            )
            args.append(
                c_ast.BinaryOp(
                    op="+", left=right.right.name, right=right.right.subscript
                )
            )
        return args

    def visit_ID(self, node):
        if node.name == self.vectorize_var:
            node.name = str(0)
        return node

    def transform_pragma_loop(self, for_loop, pragma_text):
        def get_args(pragma_text):
            # Use regular expressions to match input and output parameters.
            pattern = r"input\\[(.*?)\\].*?output\\[(.*?)\\]"
            match = re.search(pattern, pragma_text)

            if match:
                # Extract the input and output parameters.
                input_params = match.group(1).split(", ")
                output_params = match.group(2).split(", ")
                return output_params, input_params
            else:
                return None, None

        output_params, input_params = get_args(pragma_text)
        assert output_params is not None
        assert input_params is not None
        args = self.extract_argments(for_loop)
        # args = output_params + input_params
        # Reset the upper limit list and access the for_loop to populate
        # self.loop_exts.
        self.loop_exts = []
        # Recursively access the child nodes of for_loop to trigger visit_For.
        self.visit(for_loop)

        if "memory" in pragma_text:
            src_dir = (
                input_params[0].split("_")[1].upper()
                if "_" in input_params[0]
                else "GDRAM"
            )
            dst_dir = (
                output_params[0].split("_")[1].upper()
                if "_" in output_params[0]
                else "GDRAM"
            )
            direction = src_dir + "2" + dst_dir
            transformed_code = c_ast.FuncCall(
                name=c_ast.ID("__memcpy"),
                args=c_ast.ExprList(
                    [
                        *(args),
                        c_ast.Constant(
                            "int",
                            str(
                                4
                                * np.prod([int(ext) for ext in self.loop_exts])
                            ),
                        ),
                        c_ast.ID(direction),
                    ]
                ),
            )
        elif "matmul" in pragma_text:
            transformed_code = c_ast.FuncCall(
                name=c_ast.ID("__matmul"),
                args=c_ast.ExprList(
                    [
                        c_ast.ID("C_nram"),
                        c_ast.ID("A_nram"),
                        c_ast.ID("B_wram"),
                    ]
                ),
            )
        elif "add" in pragma_text:
            transformed_code = c_ast.FuncCall(
                name=c_ast.ID("__bang_add"),
                args=c_ast.ExprList(
                    [
                        *(args),
                        c_ast.Constant(
                            "int",
                            str(np.prod([int(ext) for ext in self.loop_exts])),
                        ),
                    ]
                ),
            )
        else:
            # If the pragma does not match, return to the original for loop.
            return for_loop

        # Wrap the function call in a code block.
        return transformed_code

    def visit_For(self, node):
        """提取 for 循环的上界值并存储在 self.loop_exts 列表中."""
        if isinstance(node.cond, c_ast.BinaryOp) and node.cond.op == "<":
            if isinstance(node.cond.right, c_ast.Constant):
                upper_bound = node.cond.right.value
                self.loop_exts.append(upper_bound)
        # Recursive access to nested for loops
        if isinstance(node.stmt, c_ast.Compound):
            for stmt in node.stmt.block_items:
                if isinstance(stmt, c_ast.For):
                    # Manually invoke visit to access the nested for loop.
                    self.visit(stmt)
        return self.generic_visit(node)


def ast_tensorization(code, target="mlu"):
    # Analytical code
    ast = parse_code_ast(code)

    # Perform the PragmaToSIMD conversion.
    transformer = PragmaToSIMDTransformer()
    ast = transformer.visit(ast)

    # Output the modified code.
    tensorized_code = generate_code(ast)
    return make_full_func(tensorized_code, target)


if __name__ == "__main__":
    # Sample code
    code = """
    void matmul(float * A, float * B, float * C)
    {
    __wram__ float B_wram[32768];
    __nram__ float A_nram[512];
    __nram__ float C_nram[64];

    # pragma operation(memory(input[B], output[B_wram]))
    for (int col=0; col < 64; col + +) {
        for (int i=0; i < 512; i + +) {
            B_wram[i * 64 + col] = B[i * 64 + col];
        }
    }

    # pragma operation(memory(input[A], output[A_wram]))
    for (int i=0; i < 512; i + +) {
        A_nram[i] = A[(clusterId * 4 + coreId) * 512 + i];
    }

    # pragma operation(matmul(input[A_nram, B_wram], output[C_nram]))
    for (int col=0; col < 64; col + +) {
        C_nram[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i=0; i < 512; i + +) {
            C_nram[col] += A_nram[i] * B_wram[i * 64 + col];
        }
    }

    # pragma operation(memory(input[C_nram], output[C]))
    for (int col=0; col < 64; col + +) {
        C[(clusterId * 4 + coreId) * 64 + col] = C_nram[col];
    }
    }
    """

    output_code = ast_tensorization(code)
    print(output_code)
