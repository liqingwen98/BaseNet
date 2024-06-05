
from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch, copy, math
import torch.nn as nn
from torch.nn.functional import log_softmax
from ..utils.model_utils import Feature_extract, Embeddings, PositionalEncoding

"""
Generate mask for padding position and causal mask
"""

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
        tgt_mask.data
    )
    return tgt_mask

def generate_mask(lengths, max_len=2000):
    n = len(lengths)
    mask = np.zeros((n, max_len), dtype=int)
    for i in range(n):
        mask[i, :lengths[i]] = 1
    return torch.tensor(mask,dtype=torch.bool).unsqueeze(-2)

"""
Beam search decode strategy for auto-regressive
"""

class BeamSearchDecoder():
    def __init__(
        self,
        beam_size: int = 4,
        eot: int = 0,
        weight = 0.4
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.patience = 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None
        self.weight = weight

    def reset(self):
        self.finished_sequences = None

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor, leng, is_la
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                if len(prefix) >= int(self.weight*0.6375*leng[i]):
                    for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                        new_logprob = (sum_logprobs[idx] + logprob).item()
                        sequence = tuple(prefix + [token.item()])
                        scores[sequence] = new_logprob
                        sources[sequence] = idx
                else:
                    for logprob, token in zip(*logprobs[idx][1:5].topk(self.beam_size)):
                        new_logprob = (sum_logprobs[idx] + logprob).item()
                        sequence = tuple(prefix + [token.item()+1])
                        scores[sequence] = new_logprob
                        sources[sequence] = idx    

            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)

        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  
                previously_finished[seq] = newly_finished[seq]

        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs
    
class MaximumLikelihoodRanker():

    def __init__(self, length_penalty = 0.6):
        self.length_penalty = length_penalty

    def rank(self, tokens, sum_logprobs):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]

class DecodingTask:
    def __init__(self, model, weight):
        self.model = model.eval()
        self.n_group: int = 4
        self.sample_len: int = 1274
        self.initial_tokens: Tuple[int] = 5
        self.decoder = BeamSearchDecoder(weight = weight)
        self.sequence_ranker = MaximumLikelihoodRanker(0.6)
          
    def _main_loop(self, audio_features: Tensor, tokens: Tensor, src_mask, leng, is_la):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)

        for i in range(self.sample_len):
            tgt_mask = make_std_mask(tokens, 6)
            logits = self.model.decode(audio_features, src_mask, tokens, tgt_mask)
            logits = logits[:, -1]
            tokens, completed = self.decoder.update(tokens, logits, sum_logprobs, leng, is_la)
            if completed or tokens.shape[-1] > self.sample_len:
                break
            
        return tokens, sum_logprobs
            
    @torch.no_grad()
    def run(self, audio_features, src_mask, leng, is_la):
        self.decoder.reset()
        n_audio = audio_features.shape[0]
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        src_mask = src_mask.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)  
        tokens, sum_logprobs = self._main_loop(audio_features, tokens, src_mask, leng, is_la)  
        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)   
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)    
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        
        return tokens

"""
The Encoder-Decoder achetecture of BaseFormer 
"""
    
class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    # def forward(self, x):
    #     return log_softmax(self.proj(x), dim=-1).permute(1, 0, 2)
    def forward(self, x):
        return self.proj(x)
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Model(nn.Module):

    def __init__(self, tgt_vocab=5, d_model=384, 
                 n_layers=6, d_ff=2048, dropout=0.1, heads=8):
        super(Model, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position1 = PositionalEncoding(d_model, dropout, 2000)
        position2 = PositionalEncoding(d_model, dropout, 1274)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers)
        self.src_embed = nn.Sequential(Feature_extract(d_model), c(position1))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position2))
        self.generator = Generator(d_model, tgt_vocab)
        self.alphabet = [ "N", "A", "C", "G", "T" ]
        self.stride = 5

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


