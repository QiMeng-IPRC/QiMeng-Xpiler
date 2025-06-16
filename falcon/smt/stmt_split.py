from pycparser import c_ast

from falcon.smt.const_inline import constant_inline
from falcon.util import (NodeTransformer, generate_code, make_full_func,
                         parse_code_ast)


class LoopSplitter(NodeTransformer):
    def visit_Compound(self, node):
        # Split each for loop containing multiple statements into separate
        # loops.
        new_block_items = []
        for stmt in node.block_items:
            if isinstance(stmt, c_ast.For):
                # Check if the loop body contains multiple statements.
                if (
                    isinstance(stmt.stmt, c_ast.Compound)
                    and len(stmt.stmt.block_items) > 1
                ):
                    # Determine whether the loop body contains a declaration
                    # statement.
                    contains_decl = any(
                        isinstance(item, c_ast.Decl)
                        for item in stmt.stmt.block_items
                    )

                    if contains_decl:
                        # If it contains a declaration statement, keep the
                        # entire loop body intact without splitting it.
                        new_block_items.append(stmt)
                    else:
                        # If declaration statements are not included, split
                        # each statement into separate `for` loops.
                        for single_stmt in stmt.stmt.block_items:
                            new_for = c_ast.For(
                                init=stmt.init,
                                cond=stmt.cond,
                                next=stmt.next,
                                stmt=c_ast.Compound(
                                    [single_stmt]
                                ),  # Single-statement loop body
                            )
                            new_block_items.append(new_for)
                else:
                    # If the loop contains only one statement, add it directly.
                    new_block_items.append(stmt)
            else:
                # Non-looping statements remain unchanged.
                new_block_items.append(stmt)

        # Update the statements within the block using the split loop.
        node.block_items = new_block_items
        return self.generic_visit(node)


def ast_stmt_split(code, target="None"):
    # Parse code and apply loop splitting
    code = constant_inline(code)
    ast = parse_code_ast(code)
    # Apply loop splitting transformation
    splitter = LoopSplitter()
    split_ast = splitter.visit(ast)

    # Generate and print transformed code
    code = generate_code(split_ast)
    code = make_full_func(code, target)
    return code


if __name__ == "__main__":
    # Sample code to transform
    code = """
    void sum(float* expf, float* T_softmax_maxelem) {
        float denom = 0.0f;
        float maxVal = -3.0f;
        for (int i = 0; i < 5; ++i) {
            T_softmax_maxelem[threadIdxx * 5 + i] = expf(A[threadIdxx * 5 + i] - maxVal);
            denom += T_softmax_maxelem[threadIdxx * 5 + i];
        }
    }
    """
    code = ast_stmt_split(code)
    print(code)
    code = """
    void softmax(float *A, float *T_softmax_norm)
    {
        for (int threadIdxx = 0; threadIdxx < 5; ++threadIdxx)
        {
            int rowStart = threadIdxx * 128;
            float maxVal = A[rowStart];
            for (int i = 1; i < 128; ++i)
            {
            if (A[rowStart + i] > maxVal)
            {
                maxVal = A[rowStart + i];
            }
            }

            float denom = 0.0f;
            for (int i = 0; i < 128; ++i)
            {
            T_softmax_norm[rowStart + i] = expf();
            denom += T_softmax_norm[rowStart + i];
            }

            for (int i = 0; i < 128; ++i)
            {
            T_softmax_norm[rowStart + i] /= denom;
            }

        }

    }
    """
    code = ast_stmt_split(code)
    print(code)

    code = """
    extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *add_1605) {
        __nram__ float lhs_local_nram[512];
        if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
            __memcpy(
                ((float *)lhs_local_nram + (0)),
                ((float *)lhs + (((((int)clusterId) * 1024) + (((int)coreId) * 256)))),
                1024, GDRAM2NRAM);
        }
        if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
            __memcpy(
                ((float *)lhs_local_nram + (256)),
                ((float *)rhs + (((((int)clusterId) * 1024) + (((int)coreId) * 256)))),
                1024, GDRAM2NRAM);
        }
        if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
            __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)),
                    ((float *)lhs_local_nram + (256)), 256);
        }
        if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
            __memcpy(((float *)add_1605 +
                    (((((int)clusterId) * 1024) + (((int)coreId) * 256)))),
                    ((float *)lhs_local_nram + (0)), 1024, NRAM2GDRAM);
        }
    }
    """
    code = ast_stmt_split(code)
    print(code)
