import re

from pycparser import c_ast, c_generator

from falcon.util import NodeTransformer, generate_code, parse_code_ast


class MergeLoopsAndIfsVisitor(NodeTransformer):
    def __init__(self):
        self.loop_vars = []
        self.loop_bounds = {}

    def visit_For(self, node):
        if isinstance(node.cond.right, c_ast.Constant) and isinstance(
            node.init, c_ast.DeclList
        ):
            if node.cond.op == "<":
                self.loop_bounds[node.init.decls[0].name] = (
                    int(node.cond.right.value) - 1
                )
            elif node.cond.op == "<=":
                self.loop_bounds[node.init.decls[0].name] = int(
                    node.cond.right.value
                )
        return self.generic_visit(node)

    def visit_Compound(self, node):
        """查找并合并连续的 for 循环和 if 语句."""
        new_block_items = []
        i = 0
        while i < len(node.block_items):
            current_node = node.block_items[i]
            # Check if it is a for loop and if the next node is also a for
            # loop.
            if (
                isinstance(current_node, c_ast.For)
                and i + 1 < len(node.block_items)
                and isinstance(node.block_items[i + 1], c_ast.For)
            ):
                combined_body = []

                # Extract the condition and step of the first loop.
                init_stmt = current_node.init
                cond_stmt = current_node.cond
                next_stmt = current_node.next

                # Traverse consecutive similar for loops.
                while (
                    i < len(node.block_items)
                    and isinstance(node.block_items[i], c_ast.For)
                    and self.nodes_equal(current_node, node.block_items[i])
                ):
                    loop_var = self.get_loop_variable(node.block_items[i])
                    if loop_var:
                        self.loop_vars.append(loop_var)

                    # Merge the contents of the loop bodies.
                    combined_body.extend(node.block_items[i].stmt.block_items)
                    i += 1  # Continue to inspect the next node.

                # Update the loop variable in the loop body.
                combined_body = self.rename_loop_variables(combined_body)

                # Create a merged for loop.
                combined_for = c_ast.For(
                    init=init_stmt,
                    cond=cond_stmt,
                    next=next_stmt,
                    stmt=c_ast.Compound(block_items=combined_body),
                )

                # Add the merged loops to the new block_items.
                new_block_items.append(combined_for)
            # Check if it is an if statement and whether the next node is also
            # an if statement.
            elif (
                isinstance(current_node, c_ast.If)
                and i + 1 < len(node.block_items)
                and isinstance(node.block_items[i + 1], c_ast.If)
                and self.nodes_equal(current_node, node.block_items[i + 1])
            ):
                combined_body = []

                # Traverse consecutive similar if statements.
                while (
                    i < len(node.block_items)
                    and isinstance(node.block_items[i], c_ast.If)
                    and self.nodes_equal(current_node, node.block_items[i])
                ):
                    # Merge the contents of the if statement's body.
                    if node.block_items[i].iftrue:
                        combined_body.extend(
                            node.block_items[i].iftrue.block_items
                        )
                    i += 1  # Continue to check the next node.

                # Create the merged if statement.
                combined_if = c_ast.If(
                    cond=current_node.cond,
                    iftrue=c_ast.Compound(block_items=combined_body),
                    iffalse=None,
                )

                # Add the merged if statements to the new block_items.
                new_block_items.append(combined_if)
            else:
                # Directly add nodes other than for loops or if statements.
                new_block_items.append(current_node)
                i += 1

        # Update the node's block_items.
        node.block_items = new_block_items
        return self.generic_visit(node)

    def is_similar_loop(self, loop1, loop2):
        """检查两个 for 循环是否具有相同的循环条件、初始条件和步进操作."""
        return (
            isinstance(loop1, c_ast.For)
            and isinstance(loop2, c_ast.For)
            and loop1.init.__class__ == loop2.init.__class__
            and loop1.cond.__class__ == loop2.cond.__class__
            and loop1.next.__class__ == loop2.next.__class__
            and loop1.cond.right.value
            == loop2.cond.right.value  # Ensure consistent scope.
            and loop1.cond.op == loop2.cond.op  # Ensure consistent operators.
        )

    def get_loop_variable(self, for_loop):
        """获取 for 循环中的循环变量."""
        if isinstance(for_loop, c_ast.For):
            return for_loop.init.decls[0].name
        return None

    def rename_loop_variables(self, block_items):
        """重命名循环体中的循环变量."""
        # Rename using the name of the first loop variable.
        for item in block_items:
            self.generic_visit(item)
        return block_items

    def visit_ID(self, node):
        if node.name in self.loop_vars:
            node.name = self.loop_vars[0]
        return node

    def nodes_equal(self, node1, node2):
        """递归地比较两个 AST 节点是否相同."""
        generator = c_generator.CGenerator()
        output_code = generator.visit(node1)
        generator = c_generator.CGenerator()
        output_code = generator.visit(node2)
        return output_code == output_code

    def visit_If(self, node):
        # Check if the 'if' condition is always true.
        if self.is_condition_always_true(node.cond):
            # If the condition is always true, replace the if statement with
            # its body content.
            return node.iftrue.block_items
        # Otherwise, keep the if statement.
        return self.generic_visit(node)

    def is_condition_always_true(self, condition):
        # Only handles simple `<` and `<=` cases.
        if isinstance(condition, c_ast.BinaryOp) and condition.op in (
            "<",
            "<=",
        ):
            left, right = condition.left, condition.right
            # Determine if the right side is a constant.
            if isinstance(right, c_ast.Constant):
                # Check if the left side contains loop variables and range, and
                # determine if the condition is always true.
                return self.is_left_expression_in_bounds(
                    left, int(right.value)
                )
        return False

    def is_left_expression_in_bounds(self, expr, bound):
        # Check if the maximum possible value of the expression through
        # recursion is less than the given limit.
        expr_bound = self.get_expression_bound(expr)
        return expr_bound is not None and expr_bound < bound

    def get_expression_bound(self, expr):
        # Calculate the maximum possible value of the expression to determine
        # if it is within the upper limit.
        if isinstance(expr, c_ast.ID):
            # "Get the upper bound of the variable's range."
            return self.get_variable_bound(expr.name)
        elif isinstance(expr, c_ast.Constant):
            # If it is a constant, return its value.
            return int(expr.value)
        elif isinstance(expr, c_ast.BinaryOp):
            left_bound = self.get_expression_bound(expr.left)
            right_bound = self.get_expression_bound(expr.right)
            if expr.op == "+":
                # Handle the addition operation.
                if left_bound is not None and right_bound is not None:
                    return left_bound + right_bound
            elif expr.op == "*":
                # Handle the multiplication operation.
                if left_bound is not None and right_bound is not None:
                    return left_bound * right_bound
            elif expr.op == "-":
                # Handle the subtraction operation.
                if left_bound is not None and right_bound is not None:
                    return left_bound - right_bound
            elif expr.op == "/":
                # Handle division operations, avoid division by zero.
                if (
                    left_bound is not None
                    and right_bound is not None
                    and right_bound != 0
                ):
                    return left_bound // right_bound  # Integer division
        return None

    def get_variable_bound(self, var_name):
        # Return the upper bound of the loop variable (assuming the range of
        # these variables is known).
        return self.loop_bounds.get(var_name, None)


def ast_stmt_simplification(code):
    code = re.sub(r"//.*?\n|/\*.*?\*/", "", code, flags=re.S)
    # Analyze the code and apply the merge.
    ast = parse_code_ast(code)
    # Traverse and merge using MergeForLoopsVisitor.
    merge_visitor = MergeLoopsAndIfsVisitor()
    ast = merge_visitor.visit(ast)

    # Output the modified code.
    return generate_code(ast)


if __name__ == "__main__":
    # Example usage
    code = """
    void add(float *lhs, float *rhs, float *add_1515)
    {
    float lhs_local_nram[128];
    for (int i = 0; i < 64; i++)
    {
        (((float *) lhs_local_nram) + 0)[i] = (((float *) lhs) + ((((int) clusterId) * 256) + (((int) coreId) * 64)))[i];
    }

    for (int i = 0; i < 64; i++)
    {
        (((float *) lhs_local_nram) + 64)[i] = (((float *) rhs) + ((((int) clusterId) * 256) + (((int) coreId) * 64)))[i];
    }

    for (int i = 0; i < 64; i++)
    {
        (((float *) lhs_local_nram) + 0)[i] = (((float *) lhs_local_nram) + 0)[i] + (((float *) lhs_local_nram) + 64)[i];
    }


    for (int i = 0; i < 64; i++)
    {
        (((float *) add_1515) + ((((int) clusterId) * 256) + (((int) coreId) * 64)))[i] = (((float *) lhs_local_nram) + 0)[i];
    }

    }
    """
    code = ast_stmt_simplification(code)
    print(code)

    code = """
    void tanh(float *input0, float *active_tanh_210)
    {
    for (int clusterId = 0; clusterId < 4; ++clusterId)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
        float input0_local_nram[640];
        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = (((float *) input0) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i];
        }

        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = tanh((((float *) input0_local_nram) + 0)[i]);
        }

        for (int i = 0; i < 640; i++)
        {
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i] = (((float *) input0_local_nram) + 0)[i];
        }

        }

    }

    }
    """
    code = ast_stmt_simplification(code)
    print(code)

    code = """
    void tanh(float *input0, float *active_tanh_210)
    {
    for (int clusterId = 0; clusterId < 4; ++clusterId)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
        float input0_local_nram[640];
        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = (((float *) input0) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i];
        }

        for (int j = 0; j < 640; j++)
        {
            (((float *) input0_local_nram) + 0)[j] = tanh((((float *) input0_local_nram) + 0)[j]);
        }

        for (int k = 0; k < 640; k++)
        {
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[k] = (((float *) input0_local_nram) + 0)[k];
        }

        }

    }

    }
    """
    code = ast_stmt_simplification(code)
    print(code)

    c_code = """
    void sign(float *input0, float *active_sign_147)
    {
        for (int clusterId = 0; clusterId < 4; ++clusterId)
        {
            for (int coreId = 0; coreId < 4; ++coreId)
            {
                float input0_local_nram[25];
                for (int i0_outer_outer_outer = 0; i0_outer_outer_outer < 3; ++i0_outer_outer_outer)
                {
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        int src_offset = ((i0_outer_outer_outer * 400) + (((int) clusterId) * 100)) + (((int) coreId) * 25);
                        int dst_offset = 0;
                        for (int i = 0; i < 25; ++i)
                        {
                            input0_local_nram[dst_offset + i] = input0[src_offset + i];
                        }
                    }
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        // Detensorizing the __bang_active_sign
                        for (int i = 0; i < 25; ++i)
                        {
                            if (input0_local_nram[i] >= 0)
                                input0_local_nram[i] = 1.0f;
                            else
                                input0_local_nram[i] = -1.0f;
                        }
                    }
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        int src_offset = 0;
                        int dst_offset = ((i0_outer_outer_outer * 400) + (((int) clusterId) * 100)) + (((int) coreId) * 25);
                        for (int i = 0; i < 25; ++i)
                        {
                            active_sign_147[dst_offset + i] = input0_local_nram[src_offset + i];
                        }
                    }
                }
            }
        }
    }
    """
    code = ast_stmt_simplification(c_code)
    print(code)

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
        for (int k = 0; k < 16; k++)
        {
            for (int j = 0; j < 256; j++)
            {
                if (((k * 256) + j) < 4096)
                {
                    T_add[(k * 256) + j] = A[(k * 256) + j] + B[(k * 256) + j];
                }
            }

        }
    }
    """
    code = ast_stmt_simplification(c_code)
    print(code)
