from pycparser import c_ast

from falcon.mcts.actions import actions as ActionSpace
from falcon.util import parse_code_ast


class CallNodeTransformer(c_ast.NodeVisitor):
    def __init__(self):
        self.func_call = False

    def visit_FuncCall(self, node):
        self.func_call = True


def visit_func_call(code, target=None):
    ast = parse_code_ast(code, target=target)
    # Count the number of loop layers.
    loop_visitor = CallNodeTransformer()
    loop_visitor.visit(ast)
    return loop_visitor.func_call


class CompoundNodeTransformer(c_ast.NodeVisitor):
    def __init__(self):
        # Used to mark whether a compound statement is encountered.
        self.has_compound_stmt = False

    def visit_Compound(self, node):
        # Check if the compound statement contains multiple statements.
        if len(node.block_items) > 1:
            self.has_compound_stmt = True
        self.generic_visit(node)


def visit_compound_stmt(code, target=None):
    ast = parse_code_ast(code, target=target)
    compound_visitor = CompoundNodeTransformer()
    compound_visitor.visit(ast)
    return compound_visitor.has_compound_stmt


def get_invalid_actions(code, source_platform, target_platform):
    invalid_mask = [0] * len(ActionSpace)

    if source_platform == "cpu":
        invalid_mask[0] = 1

    # add compound stmt check
    if not visit_func_call(code, source_platform):
        invalid_mask[2] = 1

    if not visit_compound_stmt(code, source_platform):
        invalid_mask[1] = 1

    if target_platform == "cpu":
        invalid_mask[7] = 1
        invalid_mask[8] = 1
        invalid_mask[10] = 1

    if (
        "coreId" not in code
        and "threadIdx." not in code
        and "blockIdx.x" not in code
    ):
        invalid_mask[0] = 1
    return invalid_mask


if __name__ == "__main__":
    code = """
    int square(int x) {
        return x * x;
    }
    """
    result = visit_func_call(code)
    print(result)

    code = """
    int main() {
        int a = 3;
        int b = square(a);  // <--- 函数调用
        return b;
    }
    """
    result = visit_func_call(code)
    print(result)

    code = """
    int main() {
        int a = 3;
        square(a);  // <--- 函数调用
        return a;
    }
    """
    result = visit_func_call(code)
    print(result)
