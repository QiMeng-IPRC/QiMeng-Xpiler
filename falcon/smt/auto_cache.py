import logging

from pycparser import c_ast

from falcon.util import (
    NodeTransformer,
    add_memory_prefix,
    generate_code,
    parse_code_ast,
)


class LoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        # self.cache_node = {}
        self.cache_size = []

    def visit_Compound(self, node):
        start_cache = False
        for item in node.block_items:
            if isinstance(item, c_ast.Pragma) and "__bang" in item.string:
                # The detection of a #pragma line indicates that caching
                # operations are required.
                start_cache = True
                # self.cache_node[item.string] = None
            elif isinstance(item, c_ast.For) and start_cache:
                # self.cache_node[item.string] = item
                self.cache_size.append(item.cond.right.value)
                start_cache = False  # Reset flag
        self.generic_visit(node)


class CacheTransformationVisitor(NodeTransformer):
    def __init__(self, space_map, cache_size):
        super().__init__()
        self.space_map = space_map
        self.cache_size = cache_size

    def visit_FuncDef(self, node):
        """在函数定义节点内创建缓存缓冲区，并添加缓存加载和写回逻辑."""
        self.create_cache_buffers(
            node
        )  # Create a cache buffer at the beginning of the function.
        return node

    def create_cache_buffers(self, node):
        """根据 space_map 创建 NRAM 缓冲区."""
        size_param = c_ast.Constant(type="int", value=self.cache_size[0])
        declarations = []

        # Dynamically create NRAM buffers based on the input and output in
        # space_map.
        for mapping in self.space_map:
            for var_name, location in mapping.get("input", {}).items():
                nram_decl = c_ast.Decl(
                    name=f"{var_name}_{location}",
                    quals=[],
                    storage=[],
                    funcspec=[],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl(
                            declname=f"{var_name}_{location}",
                            quals=[],
                            align=None,
                            type=c_ast.IdentifierType(["float"]),
                        ),
                        dim=size_param,
                        dim_quals=[],
                    ),
                    align=None,
                    init=None,
                    bitsize=None,
                )
                declarations.append(nram_decl)

            for var_name, location in mapping.get("output", {}).items():
                nram_decl = c_ast.Decl(
                    name=f"{var_name}_{location}",
                    quals=[],
                    storage=[],
                    funcspec=[],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl(
                            declname=f"{var_name}_{location}",
                            quals=[],
                            align=None,
                            type=c_ast.IdentifierType(["float"]),
                        ),
                        dim=size_param,
                        dim_quals=[],
                    ),
                    align=None,
                    init=None,
                    bitsize=None,
                )
                declarations.append(nram_decl)

        # Insert at the beginning of the function body.
        node.body.block_items = declarations + node.body.block_items
        self.generic_visit(node)

    def visit_Compound(self, node):
        """在找到 for 循环后插入缓存读写操作."""
        new_block_items = []
        start_cache = False
        for item in node.block_items:
            if isinstance(item, c_ast.Pragma) and "__bang" in item.string:
                # The detection of a #pragma line indicates that caching
                # operations are required.
                start_cache = True
            elif isinstance(item, c_ast.For) and start_cache:
                reads, writes, inner_for = self.extract_index_expression(item)
                # Insert cache read (read operation)
                read_items = self.create_read_operations(inner_for, reads)
                # Insert the original for loop.
                new_block_items.extend(read_items)
                # Modify the variable in the loop body.
                new_item = self.modify_for_loop_body(item, reads, writes)
                new_block_items.append(new_item)
                # Insert cache write-back (write operation)
                write_items = self.create_write_operations(inner_for, writes)
                new_block_items.extend(write_items)
                start_cache = False  # Reset flag
            else:
                new_block_items.append(item)
        node.block_items = new_block_items
        return self.generic_visit(node)

    def modify_for_loop_body(self, for_node, reads, writes):
        """将 for 循环体内的变量替换为 NRAM 缓冲区变量."""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        inputs = []
        for var_name, location in self.space_map[0]["input"].items():
            inputs.append(
                c_ast.ArrayRef(
                    name=c_ast.ID(name=f"{var_name}_{location}"),
                    subscript=index,
                )
            )
        ouptuts = []
        for var_name, location in self.space_map[0]["output"].items():
            ouptuts.append(
                c_ast.ArrayRef(
                    name=c_ast.ID(name=f"{var_name}_{location}"),
                    subscript=index,
                )
            )

        # Find the first Assignment and the inner for-loop that contains it
        def find_assignment_in_for(node):
            # returns (assignment_stmt, target_for_node) or (None, None)
            if hasattr(node.stmt, "block_items") and node.stmt.block_items:
                for it in node.stmt.block_items:
                    if isinstance(it, c_ast.Assignment):
                        return it, node
                    if isinstance(it, c_ast.For):
                        res = find_assignment_in_for(it)
                        if res[0] is not None:
                            return res
            return None, None

        stmt, target_for = find_assignment_in_for(for_node)
        if stmt is None:
            logging.error(
                "No Assignment found in for loop body when modifying loop body for cache replacement."
            )
            raise AssertionError(
                "Expected an Assignment in for loop body but none found."
            )
        right = stmt.rvalue
        left_value = ouptuts[0]
        right_value = None
        if isinstance(right, c_ast.BinaryOp):
            right_value = c_ast.BinaryOp(
                op=right.op, left=inputs[0], right=inputs[1]
            )

        # Build a new inner for-loop that performs the assignment using cached
        # buffers
        final_inner = c_ast.For(
            init=target_for.init,
            cond=target_for.cond,
            next=target_for.next,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=", lvalue=left_value, rvalue=right_value
                    )
                ]
            ),
        )

        # Replace the target_for inside the original for_node with final_inner
        def replace_target_for(node):
            if node is target_for:
                return final_inner
            if hasattr(node.stmt, "block_items") and node.stmt.block_items:
                new_items = []
                for it in node.stmt.block_items:
                    if isinstance(it, c_ast.For):
                        new_it = replace_target_for(it)
                        new_items.append(new_it)
                    else:
                        new_items.append(it)
                node.stmt.block_items = new_items
            return node

        replaced = replace_target_for(for_node)
        return replaced

    def extract_index_expression(self, for_node):
        src_index = {}
        # Find the first Assignment and its containing for-loop (inner loop)

        def find_assignment_in_for(node):
            if hasattr(node.stmt, "block_items") and node.stmt.block_items:
                for it in node.stmt.block_items:
                    if isinstance(it, c_ast.Assignment):
                        return it, node
                    if isinstance(it, c_ast.For):
                        res = find_assignment_in_for(it)
                        if res[0] is not None:
                            return res
            return None, None

        stmt, target_for = find_assignment_in_for(for_node)
        if stmt is None:
            logging.error(
                "No Assignment found in for loop body when extracting index expression."
            )
            raise AssertionError(
                "Expected an Assignment in for loop body but none found."
            )
        right = stmt.rvalue
        if isinstance(right, c_ast.BinaryOp):
            # left and right may be ID or more complex; try to extract names
            def get_name(expr):
                if isinstance(expr, c_ast.ID):
                    return expr.name
                if hasattr(expr, "name") and isinstance(expr.name, c_ast.ID):
                    return expr.name.name
                return None

            left_name = get_name(right.left)
            right_name = get_name(right.right)
            try:
                if left_name:
                    src_index[left_name] = right.left
                if right_name:
                    src_index[right_name] = right.right
            except Exception:
                logging.exception(
                    "Failed to extract index expressions from BinaryOp"
                )
        # return src_index, lvalue, and the inner for-loop containing the
        # assignment
        return src_index, stmt.lvalue, target_for

    def create_read_operations(self, for_loop, src_index):
        """Insert cache read operations with complex indexing."""
        reads = []
        for var_name, location in self.space_map[0]["input"].items():
            reads.append(
                self.create_load_loop(
                    var_name, f"{var_name}_{location}", for_loop, src_index
                )
            )
        return reads

    def create_write_operations(self, for_loop, dest_index):
        """Insert cache write-back operations with complex indexing."""
        writes = []
        for var_name, location in self.space_map[0]["output"].items():
            writes.append(
                self.create_write_back_loop(
                    f"{var_name}_{location}", for_loop, dest_index
                )
            )
        return writes

    def create_load_loop(self, src, dest, for_node, src_index):
        """Creates a load loop with specified complex index expression."""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        return c_ast.For(
            init=for_node.init,
            cond=for_node.cond,
            next=for_node.next,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=",
                        lvalue=c_ast.ArrayRef(
                            name=c_ast.ID(name=dest), subscript=index
                        ),
                        rvalue=src_index[src],
                    )
                ]
            ),
        )

    def create_write_back_loop(self, src, for_node, index_expr):
        """Creates a write-back loop with specified complex index
        expression."""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        return c_ast.For(
            init=for_node.init,
            cond=for_node.cond,
            next=for_node.next,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=",
                        lvalue=index_expr,
                        rvalue=c_ast.ArrayRef(
                            name=c_ast.ID(name=src), subscript=index
                        ),
                    )
                ]
            ),
        )


def ast_auto_cache(code, space_map, target="mlu"):
    print("[INFO] Start auto cache process...", code)
    print("[INFO] space_map: ", space_map)
    # Analytical code
    ast = parse_code_ast(code)
    # Perform cache loading and write-back insertion.
    cache_visitor = LoopVisitor()
    cache_visitor.visit(ast)
    transformer = CacheTransformationVisitor(
        space_map, cache_visitor.cache_size
    )
    ast = transformer.visit(ast)

    # Output the final code.
    cache_code = generate_code(ast)
    if target == "mlu":
        return add_memory_prefix(cache_code)
    else:
        return "__global__ " + cache_code


if __name__ == "__main__":
    # Example code and space_map
    code = """
    void __bang_add(float *A, float *C, float *B) {
        #pragma __bang_add(input[Nram, Nram], output[Nram])
        for (int i_add = 0; i_add < 128; i_add++) {
            C[i_add] = A[i_add] + B[i_add];
        }
    }
    """

    space_map = [
        {"input": {"A": "Nram", "B": "Nram"}, "output": {"C": "Nram"}}
    ]
    output_code = ast_auto_cache(code, space_map)

    print(output_code)
    code = """extern "C" __mlu_global__ void add_kernel(float *output, float *input1, float *input2)
        {
        if (coreId < 4)
        {
            #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram])))
            for (int j = 0; j < 4; j++)
            {
            for (int k = 0; k < 128; k++)
            {
                for (int l = 0; l < 128; l++)
                {
                output[(((((coreId * 4) * 128) * 128) + ((j * 128) * 128)) + (k * 128)) + l] = input1[(((((coreId * 4) * 128) * 128) + ((j * 128) * 128)) + (k * 128)) + l] + input2[(((((coreId * 4) * 128) * 128) + ((j * 128) * 128)) + (k * 128)) + l];
                }

            }

            }

        }
        }
       """
    space_map = [
        {
            "input": {"input1": "Nram", "input2": "Nram"},
            "output": {"output": "Nram"},
        }
    ]
    output_code = ast_auto_cache(code, space_map)
    print(output_code)
