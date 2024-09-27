import torch, math

class Embeddings(torch.nn.Module):
    def __init__(self, d_model, vocab, pad):
        super(Embeddings, self).__init__()
        self.lut = torch.nn.Embedding(vocab+2, d_model, pad)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Clamp(torch.nn.Module):

    def forward(self, x):
        return torch.clamp(x, min=-0.5, max=3.5)

class Feature_extract(torch.nn.Module):
    def __init__(self, d_model, stride):
        super(Feature_extract, self).__init__()
        self.conv_ex = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, 5, stride=1, padding=5//2, bias=True),
            torch.nn.BatchNorm1d(4),
            torch.nn.SiLU(),
            Clamp(),
            torch.nn.Conv1d(4, 16, 5, stride=1, padding=5//2, bias=True),
            torch.nn.BatchNorm1d(16),
            torch.nn.SiLU(),
            Clamp(),
            torch.nn.Conv1d(16, d_model, 19, stride=stride, padding=19//2, bias=True),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.SiLU(),
            Clamp(),
        )

    def forward(self, x):
        return self.conv_ex(x).permute(0, 2, 1)
