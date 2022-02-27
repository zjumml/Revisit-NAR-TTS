import torch
import torch.distributions as dist
from torch import nn
from ...commons.normalizing_flow.glow_modules import Glow
from ..fs import FastSpeech


class GlowTTS(FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None, n_blocks_dec=12,
                 kernel_size_dec=5, dilation_rate=1, n_block_layers=4, p_dropout_dec=0.0,
                 n_split=4, n_sqz=2, sigmoid_scale=False, glow_hidden=128):
        super().__init__(dict_size, hparams, out_dims)
        del self.mel_out
        self.n_sqz = n_sqz
        glow_hidden = glow_hidden
        self.mean_only = True
        self.mean_proj = nn.Linear(self.hidden_size, self.out_dims)
        self.logstd_proj = nn.Linear(self.hidden_size, self.out_dims)
        self.decoder = Glow(
            self.out_dims,
            glow_hidden,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_dec,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=self.hidden_size,
            inv_conv_type=hparams['inv_conv_type'],
            share_cond_layers=hparams['share_cond_layers'])
        self.prior_dist = dist.Normal(0, 1)

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None,
                ref_mels=None, f0=None, uv=None, infer=False, **kwargs):
        return super(GlowTTS, self).forward(
            txt_tokens, mel2ph, spk_embed, spk_id, f0, uv,
            ref_mels=ref_mels, infer=mel2ph is None or infer, **kwargs)

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        noise_scale = self.hparams['noise_scale']
        decoder_inp = decoder_inp.transpose(1, 2)  # [B, H, T]
        tgt_nonpadding = tgt_nonpadding.transpose(1, 2)  # [B, H, T]
        if infer:
            z = torch.randn_like(decoder_inp[:, :80, :]) * noise_scale * tgt_nonpadding  # [B, H, T]
            mel_out, logdet = self.decoder(z, tgt_nonpadding, g=decoder_inp, reverse=True)
            ret['z'], ret['logdet'] = z, logdet
        else:
            tgt_mels = kwargs['ref_mels'].transpose(1, 2)  # [B, 80, T]
            ret['z'], ret['logdet'], ret['hs'] = \
                self.decoder(tgt_mels, tgt_nonpadding, g=decoder_inp, reverse=False,
                             return_hiddens=True)
            mel_out = tgt_mels
        return mel_out.transpose(1, 2)

    def store_inverse(self):
        self.decoder.store_inverse()
