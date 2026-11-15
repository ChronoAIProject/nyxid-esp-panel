"""Six deployed competing classes; scalar output is Nyx softmax probability."""
CLASSES=('background','Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')
def create(base,tf):
    # Load original initializer into base before constructing this new head.
    flat=base.layers[-1].input
    assert int(flat.shape[-1])==3480
    scores=tf.keras.layers.Dense(6,activation='softmax',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2726),bias_initializer='zeros',name='lexical_competition')(flat)
    # Weight-free standard layers preserve the scalar deployment ABI and keep
    # all six competitors in its probability denominator.
    selected=tf.keras.layers.Reshape((6,1),name='class_axis')(scores)
    selected=tf.keras.layers.Cropping1D((1,4),name='nyx_class')(selected)
    selected=tf.keras.layers.Flatten(name='nyx_probability')(selected)
    model=tf.keras.Model(base.inputs,selected,name='causal_lexical_competition')
    assert len(model.variables)==31 and tuple(model.variables[29].shape)==(3480,6)
    assert model.count_params()==43386
    return model

def logits(model,features,tf):
    # Share original encoder implementation without modifying historical code.
    import ast
    from pathlib import Path
    source=Path(__file__).with_name('causal_fullsequence_graph_v1.py')
    function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='logits')
    class Head(ast.NodeTransformer):
        def visit_Assert(self,node):
            if 'vs[29]' in ast.unparse(node.test):
                return ast.copy_location(ast.Assert(test=ast.parse('tuple(vs[29].shape)==(3480,6) and tuple(vs[30].shape)==(6,)',mode='eval').body,msg=None),node)
            return node
        def visit_List(self,node):
            if ast.unparse(node)=='[58, 60, 1]':return ast.copy_location(ast.List(elts=[ast.Constant(58),ast.Constant(60),ast.Constant(6)],ctx=ast.Load()),node)
            return self.generic_visit(node)
    env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[Head().visit(function)],type_ignores=[])),str(source),'exec'),env)
    return env['logits'](model,features,tf)

def presentation_loss(z,positive_weights,negative_weights,labels,tf):
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=z)
    return tf.reduce_mean(tf.reduce_sum(loss*(positive_weights+20*negative_weights),axis=1))
