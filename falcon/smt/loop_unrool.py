import sys
import re
from pycparser import c_parser, c_ast, c_generator

# --- 1. AST Modifier ---
class LoopUnrollOptimizer(c_ast.NodeVisitor):
    """
    AST Visitor to identify 'For' loops and insert a Pragma node
    immediately preceding them in the Compound block (the function body).
    """
    def __init__(self, unroll_factor=None):
        self.unroll_factor = unroll_factor

    def visit_Compound(self, node):
        """
        Processes a Compound block ({...}) to insert pragmas before loops.
        """
        if node.block_items:
            new_items = []
            for item in node.block_items:
                if isinstance(item, c_ast.For):
                    # Construct Pragma node
                    pragma_str = 'unroll'
                    if self.unroll_factor:
                        pragma_str += f' {self.unroll_factor}'
                    
                    pragma_node = c_ast.Pragma(string=pragma_str)
                    new_items.append(pragma_node)
                
                new_items.append(item)
                # Recurse for nested loops/blocks
                self.visit(item)
            node.block_items = new_items

# --- 2. Preprocessing Tools ---

def strip_comments(text):
    """
    Uses regex to safely remove C-style comments (// and /* ... */), 
    while preventing removal of similar characters inside string literals.
    """
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # group(1): string content (keep)
    # group(2): comment content (remove)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    
    def _replacer(match):
        if match.group(2) is not None:
            return "" # Remove comment
        else:
            return match.group(1) # Keep string content
            
    return regex.sub(_replacer, text)

def preprocess_cuda_code(code_str):
    """
    Cleans CUDA/C code for pycparser: removes comments, directives, and CUDA keywords.
    """
    # 1. Remove comments (Must be done first to prevent issues with directives inside comments)
    code_str = strip_comments(code_str)

    # 2. Remove preprocessor directives (#include, #define, etc.)
    code_str = re.sub(r'^\s*#.*', '', code_str, flags=re.MULTILINE)

    # 3. Strip CUDA keywords (Replace with empty string for parsing)
    keywords = [
        r'\b__global__\b', 
        r'\b__device__\b', 
        r'\b__host__\b', 
        r'\b__shared__\b', 
        r'\b__constant__\b',
    ]
    for kw in keywords:
        code_str = re.sub(kw, '', code_str)

    # 4. Remove Kernel Launch syntax <<<...>>>
    code_str = re.sub(r'<<<.*?>>>', '', code_str)

    return code_str

# --- 3. Main Logic ---
def apply_unroll_pass(code_str, unroll_factor=None):
    """ Applies the unroll optimization pass to the provided code string. """
    parser = c_parser.CParser()
    
    # Clean the code
    clean_code = preprocess_cuda_code(code_str)
    
    try:
        # Parse into AST
        ast = parser.parse(clean_code)
        
        # Apply optimization
        optimizer = LoopUnrollOptimizer(unroll_factor)
        optimizer.visit(ast)
        
        # Generate new code
        generator = c_generator.CGenerator()
        generated_code = generator.visit(ast)
        
        return generated_code
        
    except Exception as e:
        # Print error details for debugging
        print(f"Error parsing code: {e}")
        print("--- Preprocessed Code causing error (First 10 lines) ---")
        lines = clean_code.splitlines()
        for i, line in enumerate(lines[:10]):
            print(f"{i+1}: {line}")
        return None

# --- Test Case ---
if __name__ == "__main__":
    cuda_code = """
    #include <stdio.h>

    __global__ void vectorAdd(int *a, int *b, int *c, int n) {
        int i = 0;
        // This is a regular loop, which should be unrolled
        for (i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
        
        /* This is a 
           multi-line
           comment */
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                a[j] += k;
            }
        }
    }
    """
    print("--- Optimized Code (Auto Unroll) ---")
    optimized = apply_unroll_pass(cuda_code)
    if optimized:
        print(optimized)
    
    print("\n--- Optimized Code (Unroll 4) ---")
    optimized_4 = apply_unroll_pass(cuda_code, unroll_factor=4)
    if optimized_4:
        print(optimized_4)
