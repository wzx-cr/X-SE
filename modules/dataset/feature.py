import sys
import math
import torch
from torchaudio.compliance.kaldi import fbank
import functools
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from .common import build_fbank_extractor, feature_splice, apply_skip_frame, feature_splice_tensor, VarData
from omegaconf import OmegaConf
from pathlib import Path
from modules.utils.logging import logger
from modules.utils.common import torch_int32, get_pool_executor, EPS


class FeatExtract(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        feats_conf = conf['data']['feats']
        transform = feats_conf['transform']
        if transform:
            transform = Path(conf['data']['data_dir']).joinpath(transform)
            logger.info('Use feature transform from {}'.format(transform))
            self.transform = np.recfromtxt(transform).astype('float32')
            self.register_buffer('transform_tensor', torch.from_numpy(self.transform))
        else:
            self.transform = None
        self.splice = feats_conf['splice']
        self.skip_frame = feats_conf['skip_frame']
        # mask_padding is only applied to post_process_tensor, e.g., online feature extraction
        self.mask_padding = feats_conf.get('mask_padding', False)
        self.padding_value = feats_conf.get('padding_value', 0)
        fbank_param = OmegaConf.load(conf.get('fbank_conf', 'conf/fbank.yaml'))
        self.use_torchaudio = feats_conf.get('extractor', 'torchaudio') == 'torchaudio'
        if self.use_torchaudio:
            self.fbank_extractor = functools.partial(fbank, **fbank_param)
        else:
            self.fbank_extractor = build_fbank_extractor(**fbank_param)
        self.padding = functools.partial(pad_sequence, batch_first=True)
        self.wav_coeff = conf.get('wav_coeff', 32768) # compatible with kaldi !!!
        self.pool = get_pool_executor(**conf['pool'])

    def get_fbank(self, wav):
        if self.use_torchaudio:
            wav = wav.unsqueeze(0)
        return self.fbank_extractor(wav)

    def post_process(self, feat):
        # feat: shape of [n_frame, n_feats], torch.Tensor or numpy.ndarray
        if isinstance(feat, torch.Tensor):
            feat_trans = feature_splice_tensor(feat.unsqueeze(0), self.splice).squeeze(0)
            if self.transform is not None:
                feat_trans = (feat_trans + self.transform_tensor[0]) * self.transform_tensor[1]
        else:
            feat_trans = feature_splice(feat, self.splice)
            if self.transform is not None:
                feat_trans = (feat_trans + self.transform[0]) * self.transform[1]
        return apply_skip_frame(feat_trans, self.skip_frame)

    def post_process_tensor(self, feat):
        """ WARNING, deprecated !
        Post-process the input feature tensor by splicing, transforming, and applying skip-frame.
        feat (torch.Tensor): Input feature tensor of shape [n_batch, n_frame, n_feats]
        """
        feat_splice = feature_splice_tensor(feat, self.splice)
        if self.transform is not None:
            feat_trans = (feat_splice + self.transform_tensor[0]) * self.transform_tensor[1]
            if self.mask_padding:
                # mask = torch.ne(feat, 0).int() # Create mask where non-zero elements are 1
                mask = torch.where(feat == 0, 
                                   self.padding_value, torch.tensor(1.0, dtype=feat.dtype))
                mask_splice = feature_splice_tensor(mask, self.splice)
                feat_trans = mask_splice * feat_trans
        return apply_skip_frame(feat_trans, self.skip_frame)

    def get_feat(self, wav):
        '''
        wav: n_samples
        '''
        wav = wav * self.wav_coeff
        fbank_data = self.get_fbank(wav)
        return self.post_process(fbank_data)

    def get_feat_batch(self, wav_batch, wav_lengths=None, pl_module=None):
        """
        wav_batch: size of (n_batch, n_samples)
        wav_lengths: length of each wav
        """
        if wav_lengths is not None:
            # remove padding
            wav_list = [wav[:length] for wav, length in zip(wav_batch, wav_lengths)]
        else:
            wav_list = list(torch.unbind(wav_batch, dim=0))
        if pl_module:
            pl_module.save_data(wav_list, "wav_list")
        feat_list = list(self.pool.map(self.get_feat, wav_list))
        feat_padded = self.padding(feat_list, padding_value=self.padding_value)
        feat_lengths = torch_int32([feat.shape[0] for feat in feat_list], 
                                   device=wav_batch.device)
        return VarData(feat_padded, feat_lengths)

        # wav_batch = wav_batch * self.wav_coeff
        # # remove padding
        # wav_list = [wav[:length] for wav, length in zip(wav_batch, wav_lengths)]
        # # get feat
        # # feat_list = self.fbank_extractor(wav_list) # about 2h
        # if pl_module:
        #     pl_module.save_data(wav_list, 'wav_list')
        # feat_list = list(self.pool.map(self.get_fbank, wav_list)) # about 50min
        # # padding
        # feat_padded = self.padding(feat_list, padding_value=0)
        # # the post process will apply skip frame
        # feat_lengths = torch_int32([math.ceil(
        #     feat.shape[0]/self.skip_frame) for feat in feat_list], device=wav_batch.device)
        # # feat_lengths are required
        # return VarData(self.post_process_tensor(feat_padded), feat_lengths)
