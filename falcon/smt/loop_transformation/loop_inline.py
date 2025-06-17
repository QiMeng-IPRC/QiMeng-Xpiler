from pycparser import c_ast, c_generator

from falcon.util import NodeTransformer, parse_code_ast


class LoopMerger(NodeTransformer):
    def visit_Compound(self, node):
        # Visit each statement in a compound block and merge consecutive loops
        merged_body = []
        i = 0
        while i < len(node.block_items):
            stmt = node.block_items[i]

            # Check if the statement is a `For` loop and can be merged
            if isinstance(stmt, c_ast.For) and i + 1 < len(node.block_items):
                next_stmt = node.block_items[i + 1]
                if isinstance(next_stmt, c_ast.For) and self.can_merge(
                    stmt, next_stmt
                ):
                    # Create merged loop body
                    new_body = c_ast.Compound(
                        stmt.stmt.block_items + next_stmt.stmt.block_items
                    )
                    merged_loop = c_ast.For(
                        stmt.init, stmt.cond, stmt.next, new_body
                    )
                    merged_body.append(merged_loop)
                    i += 2  # Skip next loop, as it's merged
                    continue

            # If not mergeable, add as is
            merged_body.append(stmt)
            i += 1

        # Update block items with merged loops
        node.block_items = merged_body
        return node

    def can_merge(self, loop1, loop2):
        # Checks if two loops have the same structure and can be merged
        return (
            isinstance(loop1, loop2)
            and loop1.init.show() == loop2.init.show()
            and loop1.cond.show() == loop2.cond.show()
            and loop1.next.show() == loop2.next.show()
        )


def ast_loop_inline(code):
    # Parse the code and apply loop merging
    ast = parse_code_ast(code)

    # Apply loop merging transformation
    merger = LoopMerger()
    merged_ast = merger.visit(ast)

    # Generate and print optimized code
    generator = c_generator.CGenerator()
    return generator.visit(merged_ast)


if __name__ == "__main__":

    # Sample C code
    code = """
    void main(float* arr1, float* arr2, int n) {
        for (int i = 0; i < n; i++) {
            arr1[i] = arr1[i] * 2;
        }
        for (int j = 0; j < n; j++) {
            arr2[j] = arr2[j] + 3;
        }
    }
    """
    code = ast_loop_inline(code)
    print(code)
