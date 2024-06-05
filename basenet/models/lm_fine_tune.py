import torch
from torch.nn import Module
from transformers import Wav2Vec2Model
import torch.nn as nn

class Model(Module):
    def __init__(self):
        super().__init__()
        self.alphabet = [ "N", "A", "C", "G", "T" ]
        self.sub_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        for single_cnn in self.sub_model.feature_extractor.conv_layers:
            single_cnn.conv.stride=(1,)
        single_cnn.conv.stride=(5,)
        self.drop_out = nn.Dropout(0.1)
        self.fc= nn.Linear(768, 5)

    def forward(self, x, mask = None):
        x = self.sub_model(input_values = x, attention_mask = mask)
        x = self.fc(self.drop_out(x[0]))
        return nn.functional.log_softmax(x, dim=-1, dtype=torch.float32).transpose(0, 1)
