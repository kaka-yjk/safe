"""
AIO -- All Trains in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from trains.multiTask.SAFE_trainer import SAFE_trainer

__all__ = ['ATIO']


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'safe': SAFE_trainer,
        }

    def getTrain(self, args):

        model_name_lower = args.modelName.lower()
        if model_name_lower not in self.TRAIN_MAP:
            raise ValueError(f"Unknown model name '{args.modelName}' in TRAIN_MAP.")


        if hasattr(args, 'model'):
            return self.TRAIN_MAP[model_name_lower](args.model, args)
        else:
            raise AttributeError("The 'args' object must have a 'model' attribute before calling getTrain.")
