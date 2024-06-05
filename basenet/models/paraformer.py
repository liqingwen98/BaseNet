import torch

from typing import Dict
from typing import Tuple

import torch
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.predictor.cif import mae_loss
from funasr.models.predictor.cif import CifPredictorV2
from funasr.models.encoder.sanm_encoder import SANMEncoder
from funasr.models.decoder.sanm_decoder import ParaformerSANMDecoder
from ..utils.mask_utils import make_pad_mask, add_sos_eos

class Model(torch.nn.Module):
    def __init__(
            self,
            encoder = SANMEncoder(input_size = 2,
                                output_size = 256,
                                attention_heads = 4,
                                linear_units = 1024,
                                num_blocks = 4,
                                dropout_rate = 0.1,
                                positional_dropout_rate = 0.1,
                                attention_dropout_rate = 0.1,
                                input_layer = 'pe',
                                normalize_before = True,
                                kernel_size = 11,
                                sanm_shfit = 0),
            decoder = ParaformerSANMDecoder(
                                vocab_size = 5,
                                encoder_output_size = 256,
                                attention_heads = 4,
                                linear_units = 1024,
                                num_blocks = 2,
                                dropout_rate = 0.1,
                                positional_dropout_rate = 0.1,
                                self_attention_dropout_rate = 0.1,
                                src_attention_dropout_rate = 0.1,
                                att_layer_num = 2,
                                kernel_size = 11,
                                sanm_shfit = 0,
            ),
            stride = 5,
            vocab_size = 5,
            ignore_id: int = 6,
            sos: int = 5,
            eos: int = 0,
            lsm_weight: float = 0.1,
            length_normalized_loss: bool = True,
            predictor = CifPredictorV2(idim = 256,
                                     l_order = 1, 
                                     r_order = 1),
            predictor_weight: float = 1.0,
            sampling_ratio: float = 0.75,
    ):
        super().__init__()
        self.alphabet = [ "N", "A", "C", "G", "T" ]
        self.encoder = encoder
        self.decoder = decoder
        self.sos = sos
        self.eos = eos 
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.stride = stride

        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.sampling_ratio = sampling_ratio
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.step_cur = 0

    def forward(
            self,
            signal: torch.Tensor,
            signal_lengths: torch.Tensor=None,
            base: torch.Tensor=None,
            base_lengths: torch.Tensor=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        self.step_cur += 1
        base = base[:, : base_lengths.max()]
        if signal_lengths is not None:
            signal = signal[:, :, :signal_lengths.max()]
        else:
            signal_lengths = torch.tensor([signal.shape[-1] for _ in range(signal.shape[0])])

        signal = signal.permute(0, 2, 1)

        encoder_out, encoder_out_lens, _ = self.encoder(signal, signal_lengths)
        # encoder_out_lens = torch.ceil(signal_lengths/self.stride).long()
        # speech_mask = generate_mask(encoder_out_lens, max_len = encoder_out_lens.max()).to(signal.device)
        # # 1. Encoder
        # encoder_out = self.sub_model.encode(signal, speech_mask)

        loss_att = None
        loss_pre = None

        # 2. Attention decoder branch
        # base -> raw without BOS and EOS
        if self.training:
            loss_att, loss_pre = self._calc_att_loss(
                encoder_out, encoder_out_lens, base, base_lengths
                    )
            # 3. CTC-Att loss definition
            loss = loss_att + loss_pre * self.predictor_weight
            return loss
        else:
            decode_out = self._calc_att_loss(
                encoder_out, encoder_out_lens, base, base_lengths
                    )
            return decode_out

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # ys_pad -> raw + BOS
        # pre_acoustic_embeds -> raw + BOS
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        if self.training:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + 1
            pre_acoustic_embeds, pre_token_length, _, _ = self.predictor(encoder_out, ys_pad, encoder_out_mask,
                                                                                  ignore_id=self.ignore_id)
        else:
            pre_acoustic_embeds, pre_token_length, _, _ = self.predictor(encoder_out, None, encoder_out_mask,
                                                                                  ignore_id=self.ignore_id)

        # 0. sampler
        if self.training:
            # ys_pad -> raw + BOS
            sematic_embeds = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                                                               pre_acoustic_embeds)
        else:
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
        )
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        # 2. Compute attention loss
        # ys_pad -> raw + EOS
        if self.training:
            loss_att = self.criterion_att(decoder_out, ys_pad)
            loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
            return loss_att, loss_pre
        else:
            return decoder_out

    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds, chunk_mask=None):
        # ys_pad_embed = self.sub_model.tgt_embed(ys_pad[:, :-1].long())
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].to(ys_pad.device), value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds

    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        return loss_ctc
    
