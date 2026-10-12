"""Five checked literal edits to the frozen train function; no loop/loss rewrite."""
import ast
import difflib
from b4lib import require

def derive(source, num_nodes, num_depths):
    module = ast.parse(source)
    original = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    old = ast.unparse(original)
    tree = ast.Module(body=[original], type_ignores=[])
    c, d = num_nodes, num_depths
    changes = {'num_labels': 0, 'graph_max': 0, 'embedding_size': 0, 'label_slices': 0}
    class Replace(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg == 'num_labels' and isinstance(node.value, ast.Constant) and node.value.value == 32:
                node.value = ast.Constant(c)
                changes['num_labels'] += 1
            return self.generic_visit(node)
        def visit_Compare(self, node):
            if ast.unparse(node.left) == 'model.get_input_embeddings().graph.path_list.max().item()':
                require(len(node.comparators) == 1 and ast.unparse(node.comparators[0]) == '33', 'Graph assertion drift')
                node.comparators[0] = ast.Constant(c + d - 1)
                changes['graph_max'] += 1
            return self.generic_visit(node)
        def visit_BinOp(self, node):
            if ast.unparse(node) == 'tokenizer.vocab_size + 35':
                node.right = ast.Constant(c + d + 1)
                changes['embedding_size'] += 1
            return self.generic_visit(node)
        def visit_Subscript(self, node):
            if isinstance(node.slice, ast.Slice) and ast.unparse(node.slice) == '-35:-3':
                node.slice = ast.Slice(lower=ast.UnaryOp(ast.USub(), ast.Constant(c + d + 1)),
                    upper=ast.UnaryOp(ast.USub(), ast.Constant(d + 1)))
                changes['label_slices'] += 1
            return self.generic_visit(node)
    tree = Replace().visit(tree)
    ast.fix_missing_locations(tree)
    require(changes == {'num_labels': 1, 'graph_max': 1, 'embedding_size': 1, 'label_slices': 2}, 'Unexpected frozen train structure')
    derived = ast.unparse(tree)
    diff = '\n'.join(difflib.unified_diff(old.splitlines(), derived.splitlines(),
        fromfile='frozen_exp2_train', tofile='exp4b_dimension_derived_train', lineterm='')) + '\n'
    return tree, {'literal_edits': changes, 'num_supervision_nodes': c, 'num_depths': d,
        'optimizer_scheduler_training_loop_loss_formula_unchanged': True}, diff

def compile_train(source, namespace, hierarchy):
    tree, report, diff = derive(source, hierarchy['num_nodes'], hierarchy['num_depths'])
    namespace = dict(namespace)
    exec(compile(tree, '<Exp4B frozen train: five dimension literal edits>', 'exec'), namespace)
    return namespace['train'], report, diff
