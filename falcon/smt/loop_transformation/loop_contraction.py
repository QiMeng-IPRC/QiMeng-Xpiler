from pycparser import c_ast, c_generator

from falcon.simplification import simplify_code
from falcon.util import NodeTransformer, generate_code, parse_code_ast


class LoopNestFusionVisitor(NodeTransformer):
    """把两段相同循环域的 for-loop 融合成一个： for(...) {   // outer1 for(...) { /* body1 */
    } } for(...) {   // outer2，与 outer1 的 init/cond/next 相同 for(...) { /* body2
    */ } } 变成： for(...) { for(...) { /* body1 */ } for(...) { /* body2 */ }
    }"""

    def visit_Compound(self, node):
        if not node.block_items:
            return node
        new_items = []
        i = 0
        while i < len(node.block_items):
            stmt = node.block_items[i]
            # Check if the current is a for loop and the next one is also a for
            # loop
            if (
                isinstance(stmt, c_ast.For)
                and i + 1 < len(node.block_items)
                and isinstance(node.block_items[i + 1], c_ast.For)
            ):
                outer1 = stmt
                outer2 = node.block_items[i + 1]
                # Compare whether the init/cond/next of the two outer-loops are
                # consistent.
                if self._same_loop_header(outer1, outer2):
                    # Merge: Place outer2.body behind outer1.body.
                    fused = self._fuse_loops(outer1, outer2)
                    new_items.append(self.visit(fused))
                    i += 2
                    continue
            # Otherwise, retain as normal.
            new_items.append(self.visit(stmt))
            i += 1

        node.block_items = new_items
        return node

    def _same_loop_header(self, f1: c_ast.For, f2: c_ast.For) -> bool:
        # Simply compare the textual representations of init/cond/next.
        return (
            self._node_to_str(f1.init) == self._node_to_str(f2.init)
            and self._node_to_str(f1.cond) == self._node_to_str(f2.cond)
            and self._node_to_str(f1.next) == self._node_to_str(f2.next)
        )

    def _fuse_loops(self, outer1: c_ast.For, outer2: c_ast.For) -> c_ast.For:
        # Integrate outer2.body.block_items into outer1.body.
        body1 = outer1.stmt
        body2 = outer2.stmt
        # Ensure that both bodies are Compounds; otherwise, wrap them in a
        # layer.
        if not isinstance(body1, c_ast.Compound):
            body1 = c_ast.Compound([body1])
        if not isinstance(body2, c_ast.Compound):
            body2 = c_ast.Compound([body2])
        # Generate a new body.
        fused_body = c_ast.Compound(body1.block_items + body2.block_items)
        outer1.stmt = fused_body
        return outer1

    def _node_to_str(self, node):
        """把 AST 节点转字符串，方便比较."""
        if node is None:
            return ""
        return c_generator.CGenerator().visit(node)


def ast_loop_contraction(c_code):
    """Start to run loop contraction."""
    # 1. Analysis
    ast = parse_code_ast(c_code)

    # 2. Conversion (Fusion Cycle)
    visitor = LoopNestFusionVisitor()
    visitor.visit(ast)

    # 3. Generate C code
    c_generator.CGenerator()
    code = generate_code(ast)
    code = simplify_code(code)
    return code


if __name__ == "__main__":
    code = r"""
  void kernel(float A[N][M], float B[N][M], float C[N][M], float D[N][M]) {
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < M; j++) {
        A[i][j] = B[i][j] + C[i][j];
      }
    }

    for(int i = 0; i < N; i++) {
      for(int j = 0; j < M; j++) {
        D[i][j] = A[i][j] * 2;
      }
    }
  }
  """
    code = ast_loop_contraction(code)
    print(code)
