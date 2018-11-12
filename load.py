import os
import torch
from collections import OrderedDict

def load(model, model_dir='', train_device='cpu'):

    model_path = os.path.join(model_dir, 'model.pth')
    state_dict = torch.load(model_path)

    if train_device == 'cuda:0':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    return model