import re

from pycparser import c_ast, c_generator

from falcon.util import parse_code_ast


class SplitForLoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.factor = None  # Used to store the split factor extracted from the pragma
        self.axis_name = None
        self.org_extent = None

    def visit_Compound(self, node):
        """Use pragma loop_split and obtain the split factor to apply to subsequent for loops."""
        blocks = node.block_items
        if not blocks:
            return

        new_block_items = []
        skip_next = False

        # Iterate through `block_items`.
        for index, subnode in enumerate(blocks):
            if skip_next:
                skip_next = False
                continue

            # Check if it is `#pragma loop_split(<factor>)`.
            if (
                isinstance(subnode, c_ast.Pragma)
                and "loop_split" in subnode.string
            ):
                # Extract factor values
                pragma_content = subnode.string.strip()
                self.factor = int(
                    re.search(r"\\((?:factor=)?(\\d+)\\)", pragma_content).group(
                        1
                    )
                )

                # Check if the next node is a `for` loop.
                if index + 1 < len(blocks) and isinstance(
                    blocks[index + 1], c_ast.For
                ):
                    self.axis_name = blocks[index + 1].init.decls[0].name
                    # Application of cyclic decomposition
                    split_for_loop = self.split_for_loop(blocks[index + 1])
                    new_block_items.append(split_for_loop)

                    # Skip the `for` loop of the next node.
                    skip_next = True
                else:
                    # This is not a case for a `for` loop; add the original
                    # node.
                    new_block_items.append(subnode)
            else:
                # If it is neither `#pragma loop_split` nor `for`, directly add
                # the node.
                new_block_items.append(subnode)

        # Update `block_items`.
        node.block_items = new_block_items
        self.generic_visit(node)

    def split_for_loop(self, node):
        """Split for loop"""
        # Extract the maximum value of the original cycle (cycle range).
        self.org_extent = int(node.cond.right.value)
        outer_extent = self.factor

        # Create an internal loop.
        inner_init = c_ast.Decl(
            name=self.axis_name + "_in",
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname=self.axis_name + "_in",
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=c_ast.Constant("int", "0"),
            bitsize=None,
        )
        inner_cond = c_ast.BinaryOp(
            node.cond.op,
            c_ast.ID(self.axis_name + "_in"),
            c_ast.Constant("int", str(self.org_extent // self.factor)),
        )
        inner_next = c_ast.UnaryOp(
            node.next.op, c_ast.ID(self.axis_name + "_in")
        )

        # The `for` structure of the inner loop
        inner_for = c_ast.For(
            init=inner_init,
            cond=inner_cond,
            next=inner_next,
            stmt=node.stmt,
        )

        # Wrap the inner loop in a `Compound` block.
        inner_compound = c_ast.Compound(block_items=[inner_for])

        # Outer loop
        outer_init = c_ast.Decl(
            name=self.axis_name + "_out",
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname=self.axis_name + "_out",
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=c_ast.Constant("int", "0"),
            bitsize=None,
        )
        outer_cond = c_ast.BinaryOp(
            node.cond.op,
            c_ast.ID(self.axis_name + "_out"),
            c_ast.Constant("int", str(outer_extent)),
        )
        outer_next = c_ast.UnaryOp(
            node.next.op, c_ast.ID(self.axis_name + "_out")
        )

        # The `for` structure of the outer loop
        outer_for = c_ast.For(
            init=outer_init,
            cond=outer_cond,
            next=outer_next,
            stmt=inner_compound,
        )

        # Revise the reference to `axis_name` in the inner loop.
        self.generic_visit(inner_for)
        return outer_for

    def visit_ID(self, node):
        if node.name == self.axis_name:
            node.name = (
                self.axis_name
                + "_out"
                + " * "
                + str(self.org_extent // self.factor)
                + " + "
                + self.axis_name
                + "_in"
            )
        self.generic_visit(node)


def ast_loop_split(code):
    ast = parse_code_ast(code)
    generator = c_generator.CGenerator()
    # Custom visitor instance
    visitor = SplitForLoopVisitor()
    # Visit the AST to split 'for' loops with loop count 10 into 2 loops with
    # counts 2 and 5
    visitor.visit(ast)
    return generator.visit(ast)


if __name__ == "__main__":
    code = """
    int factorial(int result) {
        # pragma loop_split(2)
        for (int i=0; i < 10; i + +) {
            result += i;
        }
        return result;
    }
    """
    final_node = ast_loop_split(code)
    print(final_node)

    code = """
    void add_kernel(float * A, float * B, float * T_add) {
        for (int i=0; i < 256; i + +) {
            # pragma loop_split(4)
            for (int j=0; j < 1024; j + +) {
                T_add[((i * 1024) + j)] = (A[((i * 1024) + j)] +
                       B[((i * 1024) + j)]);
            }
        }
    }
    """
    final_node = ast_loop_split(code)
    print(final_node)

    code = """
    void softmax(float * A, float * T_softmax_norm)
    {
    for (int threadIdxx=0; threadIdxx < 5; ++threadIdxx)
    {
        float maxVal = A[threadIdxx * 128];

        # pragma loop_split(factor=4)
        for (int i=1; i < 128; ++i)
        {
        if (A[(threadIdxx * 128) + i] > maxVal)
        {
            maxVal = A[(threadIdxx * 128) + i];
        }
        }

        float denom = 0.0f;

        # pragma loop_split(factor=4)
        for (int i=0; i < 128; ++i)
        {
        T_softmax_norm[(threadIdxx * 128) +
                        i] = expf(A[(threadIdxx * 128) + i] - maxVal);
        }

        # pragma loop_split(factor=4)
        for (int i=0; i < 128; ++i)
        {
        denom += T_softmax_norm[(threadIdxx * 128) + i];
        }

        # pragma loop_split(factor=4)
        for (int i=0; i < 128; ++i)
        {
        T_softmax_norm[(threadIdxx * 128) + i] /= denom;
        }
    }
    }
    """
    final_node = ast_loop_split(code)
    print(final_node)
