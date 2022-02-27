from torch import nn
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from modules.tts.discrimators.mw_disc import Discriminator
from tasks.tts.fs import FastSpeechTask
from tasks.tts.fs2_orig import FastSpeech2OrigTask
from utils.commons.hparams import hparams
from utils.nn.model_utils import print_arch


def build_disc(h, c_in=80):
    mel_disc = Discriminator(
        time_lengths=[64, 128, 256],
        freq_length=c_in,
        hidden_size=h['disc_hidden_size'], kernel=h['disc_kernel'],
        cond_size=h['hidden_size'] if h['use_cond_disc'] else 0,
        norm_type=h['disc_norm_type'], reduction='stack',
    )
    return mel_disc


class FastSpeechAdvTask(FastSpeech2OrigTask):
    def build_model(self):
        super(FastSpeechAdvTask, self).build_model()
        self.build_disc_model()
        if not hasattr(self, 'gen_params'):
            self.gen_params = list(self.model.parameters())
        return self.model

    def build_disc_model(self):
        self.disc_params = []
        self.mel_disc = build_disc(hparams, hparams['audio_num_mel_bins'])
        self.disc_params += list(self.mel_disc.parameters())
        print_arch(self.mel_disc, model_name='Mel Disc')

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        mel_g = sample['mels']
        disc_loss_type = hparams['disc_loss_type']
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            disc_start = self.global_step >= hparams["disc_start_steps"] + 100 \
                         and hparams['lambda_mel_adv'] > 0  # warmup disc by 100 steps
            log_outputs, model_out = self.run_model(sample)
            self.model_out = {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            self.disc_cond = self.model_out['decoder_inp'].detach() if hparams['use_cond_disc'] else None
            if disc_start:
                mel_p = model_out['mel_out']
                o_ = self.mel_disc(mel_p, self.disc_cond)
                p_, h_p_, start_frames = o_['y'], o_.get('h'), o_.get('start_frames')
                if disc_loss_type == 'LSGAN':
                    self.add_LSGAN_losses(p_, 1, log_outputs, 'A')
                if disc_loss_type == 'RaLSGAN':
                    with torch.no_grad():
                        o = self.mel_disc(mel_g, self.disc_cond, start_frames)
                        p = o['y']
                    self.add_RaLSGAN_losses_gen(p, p_, log_outputs)
                for i, _ in enumerate(p_):
                    loss_weights[f'A{i}'] = hparams['lambda_mel_adv']
                if hparams['lambda_fm'] > 0:
                    o = self.mel_disc(mel_g, self.disc_cond, start_frames)
                    p, h_p = o['y'], o['h']
                    log_outputs["fm"] = 0
                    for h, h_ in zip(h_p, h_p_):
                        log_outputs["fm"] += F.l1_loss(h, h_)
                    log_outputs["fm"] = log_outputs["fm"] / len(h_p) * hparams['lambda_fm']
        else:
            #######################
            #    Discriminator    #
            #######################
            disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                if hparams['disc_rerun_gen']:
                    with torch.no_grad():
                        _, model_out = self.run_model(sample)
                else:
                    model_out = self.model_out
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g, self.disc_cond)
                p, start_frames = o['y'], o.get('start_frames')
                o_ = self.mel_disc(mel_p, self.disc_cond, start_frames)
                p_ = o_['y']
                if disc_loss_type == 'LSGAN':
                    self.add_LSGAN_losses(p, 1, log_outputs, 'R')
                    self.add_LSGAN_losses(p_, 0, log_outputs, 'F')
                if disc_loss_type == 'RaLSGAN':
                    self.add_RaLSGAN_losses_disc(p, p_, log_outputs)
            if len(log_outputs) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        log_outputs['bs'] = sample['mels'].shape[0]
        return total_loss, log_outputs

    def add_LSGAN_losses(self, p, target, ret, name='A'):
        for i, p_i in enumerate(p):
            ret[f'{name}{i}'] = F.mse_loss(p_i, p_i.new_ones(p_i.size()) * target)

    def add_RaLSGAN_losses_gen(self, p_real, p_fake, ret):
        name = 'A'
        for i, (p_real_i, p_fake_i) in enumerate(zip(p_real, p_fake)):
            ones_ = torch.ones_like(p_real_i)
            ret[f'{name}{i}'] = (F.mse_loss(p_real_i - p_fake_i.mean(), -1 * ones_) +
                                 F.mse_loss(p_fake_i - p_real_i.mean(), 1 * ones_)) / 2

    def add_RaLSGAN_losses_disc(self, p_real, p_fake, ret):
        name = 'D'
        for i, (p_real_i, p_fake_i) in enumerate(zip(p_real, p_fake)):
            ones_ = torch.ones_like(p_real_i)
            ret[f'{name}{i}'] = (F.mse_loss(p_real_i - p_fake_i.mean(), 1 * ones_) +
                                 F.mse_loss(p_fake_i - p_real_i.mean(), -1 * ones_)) / 2

    def build_optimizer(self, model):
        if not hasattr(self, 'gen_params'):
            self.gen_params = list(self.model.parameters())
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return FastSpeechTask.build_scheduler(self, optimizer[0])

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["discriminator_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler.step(self.global_step)
