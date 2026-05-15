import ast
import re
import tokenize
from io import StringIO

def remove_comments_and_docstrings(source: str) -> str:
    if not isinstance(source, str) or not source.strip(): return ""
    try:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type, token_string, (start_line, start_col), (end_line, end_col), _ = tok
            if start_line > last_lineno: last_col = 0
            if start_col > last_col: out += (" " * (start_col - last_col))
            if token_type == tokenize.COMMENT: pass
            elif token_type == tokenize.STRING:
                if prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL): pass
                else: out += token_string
            else: out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        return out
    except Exception: return source

def normalize_whitespace(source: str) -> str:
    if not isinstance(source, str): return ""
    source = source.expandtabs(4)
    source = re.sub(r'\r\n', '\n', source)
    source = re.sub(r'\n\s*\n', '\n', source)
    return source.strip()

def rename_identifiers(source: str) -> str:
    """Premenuje premenné. Ak kód nie je validný Python, vráti pôvodný."""
    try:
        tree = ast.parse(source)
        class Renamer(ast.NodeTransformer):
            def __init__(self):
                self.mapping = {}
                self.counter = 0
            def _get_name(self, old_name):
                if old_name not in self.mapping:
                    self.mapping[old_name] = f"var_{self.counter}"
                    self.counter += 1
                return self.mapping[old_name]
            def visit_Name(self, node):
                if isinstance(node.ctx, (ast.Store, ast.Load)):
                    node.id = self._get_name(node.id)
                return node
        new_tree = Renamer().visit(tree)
        return ast.unparse(new_tree)
    except Exception:
        return source