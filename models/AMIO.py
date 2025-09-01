import torch
import torch.nn as nn

from models.subNets.AlignNets import AlignSubNet
from models.multiTask.SAFE import SAFE 

__all__ = ['AMIO']

MODEL_MAP = {
    'safe': SAFE,


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.args = args
        self.need_model_aligned = args.get('need_model_aligned', False)
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args.seq_lens = self.alignNet.get_seq_len()

        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, rationale_text, rationale_vision, rationale_audio, deterministic=False):
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)

        return self.Model(text_x, audio_x, video_x, rationale_text, rationale_vision, rationale_audio,
                          deterministic=deterministic)
