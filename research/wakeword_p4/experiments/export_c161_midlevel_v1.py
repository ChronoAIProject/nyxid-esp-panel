"""Embed a bounded teacher vote and a 12-unit acoustic readout in one C161 graph."""
from copy import deepcopy

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema

from export_c161_dual_readout_v1 import graph


def export(parent, destination, weights, bias, output_weights, output_bias):
    model = graph(parent)
    g = model.subgraphs[0]
    fc, logistic = g.operators[32:34]
    assert list(fc.inputs) == [62,14,13] and list(logistic.inputs) == [63]
    join = g.operators[30]
    assert model.operatorCodes[join.opcodeIndex].builtinCode == schema.BuiltinOperator.CONCATENATION
    act = schema.ActivationFunctionType
    log_scale = g.tensors[63].quantization.scale[0]
    feature_scale = g.tensors[58].quantization.scale[0]
    hidden_scale = np.float32(6/255)

    def tensor(template, name, shape, scale, zero=0, data=None):
        t = deepcopy(g.tensors[template])
        t.name = name.encode()
        t.shape = list(shape)
        t.quantization.scale = np.array([scale],np.float32)
        t.quantization.zeroPoint = np.array([zero],np.int64)
        if data is not None:
            buf = schema.BufferT()
            buf.data = np.frombuffer(data.tobytes(),np.uint8)
            t.buffer = len(model.buffers)
            model.buffers.append(buf)
        g.tensors.append(t)
        return len(g.tensors)-1

    def dense(source, name, w, b, sw, sx, target, activation):
        qw = np.rint(w/sw).clip(-127,127).astype(np.int8)
        sb = np.float32(sw*sx)
        qb = np.rint(b/sb).astype("<i4")
        op = deepcopy(fc)
        op.inputs = [source, tensor(14,name+"_weights",qw.shape,sw,data=qw),
                     tensor(13,name+"_bias",qb.shape,sb,data=qb)]
        op.outputs = [target]
        op.builtinOptions.fusedActivationFunction = activation
        return op

    hidden = tensor(63,"midlevel_relu6",(1,len(bias)),hidden_scale,-128)
    vote = tensor(63,"bounded_teacher",(1,1),hidden_scale,-128)
    combined = tensor(63,"midlevel_and_teacher",(1,len(bias)+1),hidden_scale,-128)
    output = tensor(63,"midlevel_logit",(1,1),log_scale,int(g.tensors[63].quantization.zeroPoint[0]))
    a = dense(58,"midlevel",weights,bias,np.float32(abs(weights).max()/127),feature_scale,hidden,act.RELU6)
    v = dense(63,"teacher",np.ones((1,1)),np.array([3.]),np.float32(1/64),log_scale,vote,act.RELU6)
    concat = deepcopy(join)
    concat.inputs = [hidden,vote]
    concat.outputs = [combined]
    concat.builtinOptions.axis = 1
    last = dense(combined,"midlevel_sum",output_weights,output_bias,np.float32(abs(output_weights).max()/127),
                 hidden_scale,output,act.NONE)
    logistic.inputs = [output]
    g.operators[33:33] = [a,v,concat,last]
    builder = flatbuffers.Builder(90000)
    builder.Finish(model.Pack(builder),file_identifier=b"TFL3")
    with destination.open("xb") as f:
        f.write(bytes(builder.Output()))
    return hidden, vote, output
