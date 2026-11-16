"""Existing transcript audit with explicit two-thread CPU support on CUDA."""
import ast
from pathlib import Path
SOURCE=Path('research/wakeword_p4/training/audit_comparison_asr.py')
class Threads(ast.NodeTransformer):
    def visit_keyword(self,node):
        if node.arg=='cpu_threads':node.value=ast.Constant(2)
        return self.generic_visit(node)
if __name__=='__main__':
    tree=Threads().visit(ast.parse(SOURCE.read_text()))
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),dict(__name__='__main__',__file__=__file__))
