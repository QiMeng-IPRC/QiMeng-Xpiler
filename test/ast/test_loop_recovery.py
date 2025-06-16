import re

from pycparser import c_ast, c_generator, c_parser

from falcon.util import NodeTransformer, remove_target_prefix,parse_code_ast

# TODO(dongshouyang): Add more varaibles
ParaVar = {"threadIdx.x": 1024, "blockIdx.x": 256, "coreId": 4, "clusterId": 4}

cuda_paravar = ["threadIdx.x", "threadIdx.y", "blockIdx.x", "blockIdx.y"]
mlu_paravar = ["coreId", "clusterId"]


def update_dim(cuda_code):
    """The re module in Python is used to write a regular expression
    that matches the number inside the parentheses."""
    match = re.search(r"__launch_bounds__\((\d+)\)", cuda_code)
    if match:
       # Print the matching values
        launch_bounds_value = int(match.group(1))
        ParaVar["threadIdx.x"] = launch_bounds_value
    return ParaVar


class LoopRecoveryVisitor(NodeTransformer):
    def __init__(self, variable_map):
        self.variable_map = variable_map

    def visit_FuncDef(self, node):
        self.visit(node.body)
        body_node = node.body
        for var, ext in self.variable_map.items():
            init_node = c_ast.Decl(
                name=var.replace(".", ""),
                quals=[],
                align=[],
                storage=[],
                funcspec=[],
                type=c_ast.TypeDecl(
                    declname=var.replace(".", ""),
                    quals=[],
                    align=None,
                    type=c_ast.IdentifierType(["int"]),
                ),
                init=c_ast.Constant("int", "0"),
                bitsize=None,
            )
            cond_node = c_ast.BinaryOp(
                "<",
                c_ast.ID(var.replace(".", "")),
                c_ast.Constant("int", ext),
            )
            next_node = c_ast.UnaryOp("++", c_ast.ID(var.replace(".", "")))

            inner_loop = c_ast.For(
                init=init_node, cond=cond_node, next=next_node, stmt=body_node
            )
            body_node = c_ast.Compound(block_items=[inner_loop])

        node = c_ast.FuncDef(
            decl=node.decl, param_decls=node.param_decls, body=body_node
        )
        return node

    def visit_StructRef(self, node):
        assert node.name.name in ["threadIdx", "blockIdx"]
        name = node.name.name
        filed = node.field.name
        return c_ast.ID(name=name + filed)


def ast_loop_recovery(code, target="cuda"):
    ParaVar = update_dim(code)
    builtin_map = {}
    if target == "cuda" or target == "hip":
        for builtin_var in cuda_paravar:
            if builtin_var in code:
                builtin_map[builtin_var] = ParaVar[builtin_var]
    elif target == "mlu":
        for builtin_var in mlu_paravar:
            if builtin_var in code:
                builtin_map[builtin_var] = ParaVar[builtin_var]

    # insert the parallel loop
    ast = parse_code_ast(code)
    generator = c_generator.CGenerator()
    visitor = LoopRecoveryVisitor(builtin_map)
    visitor.visit(ast)
    return generator.visit(ast)


if __name__ == "__main__":
    cuda_code = """
    void add(float*  A, float*  B, float*  T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    converted_code = ast_loop_recovery(cuda_code, "cuda")
    print(converted_code)

    bang_code = """
    void matmul_kernel(float *A, float *B, float *C) {
        for (int col = 0; col < 128; col++) {
            C[(clusterId * 4 + coreId) * 128 + col] = 0.0f;
            for (int i = 0; i < 128; i++) {
                C[(clusterId * 4 + coreId) * 128 + col] += A[(clusterId * 4 + coreId) * 128 + i] * B[i * 128 + col];
            }
        }
    }
    """
    converted_code = ast_loop_recovery(bang_code, "mlu")
    print(converted_code)

    cuda_code = """
    extern "C" __global__ void __launch_bounds__(960) add(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        T_add[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
    }
    """
    converted_code = ast_loop_recovery(cuda_code, "cuda")
    print(converted_code)
