import torch
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from ..utils.model_utils import Feature_extract, Embeddings, PositionalEncoding
from ..utils.mask_utils import generate_lower_triangular_mask, reverse_pad_list, make_pad_mask, add_sos_eos
from ..utils.rescore_utils import ctc_beam_search

class Model(torch.nn.Module):
    def __init__(self,
                vocab_size = 5,
                dim = 512,
                decode_method = 'ctc',
                pad: int = 6,
                sos: int = 5,
                eos: int = 0,
                stride = 5,
                head = 8,
                num_layers=3,
                weight = 0.5,
                ffd = 2048):
        super().__init__()

        self.feature = Feature_extract(dim, stride)
        self.position1 = PositionalEncoding(dim, 0.1)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=head, batch_first=True, dim_feedforward=ffd)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers*2)
        self.drop_out = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(dim, vocab_size)
        
        self.emb = Embeddings(dim, vocab_size, pad)
        self.position2 = PositionalEncoding(dim, 0.1)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=dim, nhead=head, batch_first=True, dim_feedforward=ffd)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)      
        self.fc2 = torch.nn.Linear(dim, vocab_size)

        self.embr = Embeddings(dim, vocab_size, pad)
        self.position2r = PositionalEncoding(dim, 0.1)
        r_decoder_layer = torch.nn.TransformerDecoderLayer(d_model=dim, nhead=head, batch_first=True, dim_feedforward=ffd)
        self.r_decoder = torch.nn.TransformerDecoder(r_decoder_layer, num_layers=num_layers)      
        self.fc2r = torch.nn.Linear(dim, vocab_size)
        
        self.alphabet = ["N", "A", "C", "G", "T"]
        self.sos = sos
        self.eos = eos 
        self.vocab_size = vocab_size
        self.pad = pad
        self.stride = stride
        self.weight = weight
        self.decode_method = decode_method
        
        self.ctc = torch.nn.CTCLoss()
        self.att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=pad,
            smoothing=0.1,
            normalize_length=True,
        )

    def forward(self, 
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor = None,
            text_lengths: torch.Tensor = None,):
        
        if text is not None:
            text = text[:, : text_lengths.max()]

        speech = speech[:, :, :speech_lengths.max()]
        encoder_out_lens = torch.ceil(speech_lengths/self.stride).long()
        speech_masks = make_pad_mask(encoder_out_lens).bool().to(speech.device)
        # if speech_masks.shape[-1] != 2000:
        #     speech_masks = torch.cat((speech_masks, torch.ones(speech_masks.shape[0], 2000-speech_masks.shape[-1], dtype=torch.bool, device=speech.device)), dim=1)
        
        # feature extract + position encoding
        encoder_in = self.feature(speech)
        encoder_in = self.position1(encoder_in)
        
        # Encoder
        encoder_inter = self.encoder(src=encoder_in, src_key_padding_mask=speech_masks)
        # for mod in self.encoder:
        #     encoder_in = mod(encoder_in, speech_masks)
        encoder_out = self.fc1(self.drop_out(encoder_inter))
        ctc_in = torch.nn.functional.log_softmax(encoder_out, -1).permute(1, 0, 2)
        
        if self.training:            
            loss_ctc = self.ctc(ctc_in, text, encoder_out_lens, text_lengths)
            # Decoder
            # prepare input data for decoder
            text_x = torch.where(text == 0, torch.tensor(6, device=speech.device), text).int().to(speech.device)

            # prepare forward and reverse input
            r_text_x = reverse_pad_list(text_x, text_lengths, float(self.pad))
            ys_in, ys_out = add_sos_eos(text_x, self.sos, self.eos, self.pad)
            r_ys_in, r_ys_out = add_sos_eos(r_text_x, self.sos, self.eos, self.pad)
            ys_in = self.emb(ys_in)
            ys_in = self.position2(ys_in)
            r_ys_in = self.embr(r_ys_in)
            r_ys_in = self.position2r(r_ys_in)
            text_lengths = text_lengths + 1
            tgt_pad_mask = make_pad_mask(text_lengths).bool().to(text.device)
            tgt_mask = generate_lower_triangular_mask(ys_in.shape[1]).bool().to(text.device)

            decoder_out = self.decoder(tgt=ys_in, memory=encoder_inter, tgt_mask=~tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=speech_masks)
            decoder_out = self.fc2(decoder_out)

            r_decoder_out = self.r_decoder(tgt=r_ys_in, memory=encoder_inter, tgt_mask=~tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=speech_masks)
            r_decoder_out = self.fc2r(r_decoder_out)
            # calc attention loss
            loss_att = (1 - self.weight)*self.att(decoder_out, ys_out) + self.weight*self.att(r_decoder_out, r_ys_out)
            loss = self.weight*loss_ctc + (1 - self.weight)*loss_att
            return loss
        else:
            if self.decode_method == 'ctc':
                return ctc_in, encoder_out_lens
            elif self.decode_method == 'rescore':
                i = 0
                seq = []
                score = []
                for p in encoder_out:
                    # get beam size candidate input
                    hyps = ctc_beam_search(p) 
                    inter_seq = [item[0] for item in hyps]
                    seq_len = [len(item[0]) for item in hyps]
                    inter_len = torch.tensor(seq_len)
                    inter_seq = [torch.tensor(i) for i in inter_seq]
                    inter_seq = pad_sequence(inter_seq, batch_first=True, padding_value=6).to(speech.device)

                    # prepare forward and reverse input
                    r_inter_seq = reverse_pad_list(inter_seq, inter_len, float(self.pad))
                    ys_in, ys_out = add_sos_eos(inter_seq, self.sos, self.eos, self.pad)
                    r_ys_in, r_ys_out = add_sos_eos(r_inter_seq, self.sos, self.eos, self.pad)
                    ys_in = self.emb(ys_in)
                    ys_in = self.position2(ys_in)
                    r_ys_in = self.embr(r_ys_in)
                    r_ys_in = self.position2r(r_ys_in)
                    inter_len = inter_len + 1
                    tgt_pad_mask = make_pad_mask(inter_len).bool().to(speech.device)
                    tgt_mask = generate_lower_triangular_mask(ys_in.shape[1]).bool().to(speech.device)

                    # decode
                    decoder_out = self.decoder(tgt=ys_in, memory=encoder_inter[i].unsqueeze(0).repeat(self.beam_size, 1, 1), 
                                            tgt_mask=~tgt_mask, tgt_key_padding_mask=tgt_pad_mask, 
                                            memory_key_padding_mask=speech_masks[i].unsqueeze(0).repeat(self.beam_size, 1))
                    decoder_out = self.fc2(decoder_out)
                    decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
                    decoder_out = decoder_out.cpu().numpy()

                    r_decoder_out = self.r_decoder(tgt=r_ys_in, memory=encoder_inter[i].unsqueeze(0).repeat(self.beam_size, 1, 1), 
                                            tgt_mask=~tgt_mask, tgt_key_padding_mask=tgt_pad_mask, 
                                            memory_key_padding_mask=speech_masks[i].unsqueeze(0).repeat(self.beam_size, 1))
                    r_decoder_out = self.fc2r(r_decoder_out)
                    r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
                    r_decoder_out = r_decoder_out.cpu().numpy()

                    pred_seq, pred_score = self.attention_rescoring(hyps, decoder_out, r_decoder_out)
                    seq.append(pred_seq)
                    score.append(pred_score)
                    i = i+1
                return seq
    
    def attention_rescoring(self, hyps, decoder_out, r_decoder_out, reverse_weight = 0.5):
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
                score = 0.0
                for j, w in enumerate(hyp[0]):
                    score += decoder_out[i][j][w]
                score += decoder_out[i][len(hyp[0])][self.eos]
                # add right to left decoder score
                if reverse_weight > 0:
                    r_score = 0.0
                    for j, w in enumerate(hyp[0]):
                        r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                    r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                    score = score * (1 - reverse_weight) + r_score * reverse_weight
                # add ctc score
                score += hyp[1] * self.weight
                if score > best_score:
                    best_score = score
                    best_index = i
        return hyps[best_index][0], best_score