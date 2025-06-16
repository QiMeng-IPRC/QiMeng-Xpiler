import copy

from pycparser import c_ast

from falcon.util import NodeTransformer, generate_code, parse_code_ast


class ConstInlineTransformer(NodeTransformer):
    def __init__(self):
        super().__init__()
        self.constants = {}  # Constant Record
        # Variables that have been reassigned are documented.
        self.reassigned = set()

    def visit_Decl(self, node):
        if node.init and isinstance(node.init, c_ast.Constant):
            if isinstance(node.type, c_ast.TypeDecl):
                tnames = node.type.type.names
                if "int" in tnames or "float" in tnames:
                    # The initial assignment is to a constant, temporarily
                    # stored.
                    self.constants[node.name] = copy.deepcopy(node.init)
                    # ⚠️ Caution: Do not remove the declaration, otherwise the variable scope will be lost.
                    return node
        return self.generic_visit(node)

    def visit_Assignment(self, node):
        # Determine whether the assignment is a simple constant and whether it
        # is the initial assignment.
        if isinstance(node.lvalue, c_ast.ID) and isinstance(
            node.rvalue, c_ast.Constant
        ):
            varname = node.lvalue.name
            if (
                varname not in self.constants
                and varname not in self.reassigned
            ):
                self.constants[varname] = copy.deepcopy(node.rvalue)
                return node

        # Otherwise, it indicates that the variable has been written multiple
        # times and should not be replaced.
        if isinstance(node.lvalue, c_ast.ID):
            self.reassigned.add(node.lvalue.name)

        node.lvalue = self.visit(node.lvalue)
        node.rvalue = self.visit(node.rvalue)
        return node

    def visit_ID(self, node):
        if node.name in self.constants and node.name not in self.reassigned:
            return copy.deepcopy(self.constants[node.name])
        return node

    def visit_For(self, node):
        node.init = self.visit(node.init) if node.init else None
        node.cond = self.visit(node.cond) if node.cond else None
        node.stmt = self.visit(node.stmt)

        if node.next is None:
            loop_var = None
            if isinstance(node.cond, c_ast.BinaryOp) and isinstance(
                node.cond.left, c_ast.ID
            ):
                loop_var = node.cond.left.name
            if loop_var:
                node.next = c_ast.Assignment(
                    op="+=",
                    lvalue=c_ast.ID(loop_var),
                    rvalue=c_ast.Constant("int", "1"),
                )
        return node

    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node):
        node.expr = self.visit(node.expr)
        return node

    def visit_ArrayRef(self, node):
        node.name = self.visit(node.name)
        node.subscript = self.visit(node.subscript)
        return node

    def visit_DeclList(self, node):
        node.decls = [d for d in node.decls if d is not None]
        return node if node.decls else None

    def visit_FuncDef(self, node):
        self.generic_visit(node)
        if node.body and node.body.block_items:
            node.body.block_items = [
                stmt for stmt in node.body.block_items if stmt is not None
            ]
        return node


def constant_inline(code):
    ast = parse_code_ast(code)
    # "Proceed with the conversion."
    transformer = ConstInlineTransformer()
    ast = transformer.visit(ast)
    return generate_code(ast)


if __name__ == "__main__":
    # # Example code
    code = """
    void add_kernel(float *input1, float *input2, float *output)
    {
    int dim1 = 4;
    int dim2 = 4;
    int dim3 = 4;
    int dim4 = 64;
    for (int k = 0; k < dim3; k++)
    {
        for (int l = 0; l < dim4; l++)
        {
        int index = (((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l;
        #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
        output[index] = input1[index] + input2[index];
        }
    }
    }
    """
    code = constant_inline(code)
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
    code = constant_inline(code)
    print(code)
