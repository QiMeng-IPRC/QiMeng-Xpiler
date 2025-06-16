import random

from pycparser import c_ast

from falcon.util import generate_code, parse_code_ast


class LoopReorderVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.for_loops = []

    def visit_For(self, node):
        self.for_loops.append(node)
        self.generic_visit(node)

    def reorder_loops(self):
        # Traverse all the for loops to find nested loops that can be swapped.
        for node in self.for_loops:
            if isinstance(node.stmt, c_ast.Compound) and node.stmt.block_items:
                first_item = node.stmt.block_items[0]
                if isinstance(first_item, c_ast.For):
                    # Randomly decide whether to exchange.
                    if random.choice([True, False]):
                        inner_loop = first_item
                        stmt_node = inner_loop.stmt
                        # Swap the outer and inner loops
                        new_inner_loop = c_ast.For(
                            init=node.init,
                            cond=node.cond,
                            next=node.next,
                            stmt=stmt_node,
                        )
                        node.init = inner_loop.init
                        node.cond = inner_loop.cond
                        node.next = inner_loop.next
                        node.stmt = c_ast.Compound(
                            block_items=[new_inner_loop]
                        )


def ast_loop_reorder(c_code):
    ast = parse_code_ast(c_code)
    # Create a visitor instance and perform cyclic exchange.
    visitor = LoopReorderVisitor()
    visitor.visit(ast)
    visitor.reorder_loops()
    return generate_code(ast)


if __name__ == "__main__":
    c_code = """
    int factorial(int result) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                result += i + j;
            }
        }
        return result;
    }
    """
    code = ast_loop_reorder(c_code)
    print(code)
