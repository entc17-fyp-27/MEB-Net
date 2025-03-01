from __future__ import absolute_import
from collections import OrderedDict


from ..utils import to_torch

def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cuda()
    if modules is None:
        outputs = model(inputs)
        
        #print("Size of the output", len(outputs))
        outputs[0] = outputs[0].data.cpu()
        outputs[1] = outputs[1].data.cpu()
        outputs[2] = outputs[2].data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
