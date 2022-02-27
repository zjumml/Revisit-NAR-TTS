import torch
import torch.optim
import torch.utils.data
import torch.distributions
import torch.distributions as dist

from modules.tts.glow_tts.glow import GlowTTS
from tasks.tts.fs import FastSpeechTask
from utils.commons.hparams import hparams


class GlowTTSTask(FastSpeechTask):
    def build_tts_model(self):
        self.model = GlowTTS(
            len(self.token_encoder),
            hparams,
            n_sqz=hparams['n_sqz'],
            n_split=hparams['n_split'],
            n_blocks_dec=hparams['n_blocks_dec'],
            n_block_layers=hparams['n_block_layers'],
            glow_hidden=hparams['glow_hidden'],
            kernel_size_dec=hparams['dec_kernels']
        )
        self.prior_dist = dist.Normal(0, 1)

    def on_epoch_start(self):
        super(GlowTTSTask, self).on_epoch_start()
        if self.global_step == 0:
            for f in self.modules():
                if getattr(f, "set_ddi", False):
                    f.set_ddi(True)

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        mels = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        spk_embed = sample.get('spk_embed')
        spk_ids = sample.get('spk_ids')
        if not infer:
            output = self.model(txt_tokens=txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_ids=spk_ids,
                                ref_mels=mels, f0=f0, uv=uv)
            losses = {}
            self.add_glow_loss(output, sample['mel_lengths'], losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            use_gt_f0 = kwargs.get('infer_use_gt_f0', hparams['use_gt_f0'])
            mel2ph, uv, f0 = None, None, None
            if use_gt_dur:
                mel2ph = sample['mel2ph']
            if use_gt_f0:
                f0 = sample['f0']
                uv = sample['uv']
            output = self.model(txt_tokens=txt_tokens, mel2ph=mel2ph,
                                spk_embed=spk_embed, spk_ids=spk_ids, infer=True, f0=f0, uv=uv)
            return output

    def add_glow_loss(self, output, y_lengths, losses):
        logdet = output['logdet']
        z = output['z']
        losses['nlogpz'] = -self.prior_dist.log_prob(z).mean()
        losses['nlogdet'] = -(logdet / y_lengths / 80).mean()
        losses['mle'] = losses['nlogpz'].detach() + losses['nlogdet'].detach()

    def build_scheduler(self, optimizer):
        if hparams.get('use_step_lr', False):
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=500, gamma=0.998)
        else:
            return super(GlowTTSTask, self).build_scheduler(optimizer)

    def test_start(self):
        super(GlowTTSTask, self).test_start()
        self.model.store_inverse()
