import numpy as np
import pandas as pd
import random
import torch
from typing import Optional
from multiprocessing import Manager
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import time
import math
import os
import re
import getpass
import shutil
import torch.nn.functional as F

from modules.utils.common import cost_time
from modules.utils.logging import logger


@dataclass
class VarData:
    data: torch.Tensor
    lengths: Optional[torch.Tensor] = None # a list of torch.Tensor


@cost_time
def create_mem_cache(index_map, wav_store):
    clean_data, noisy_data = {}, {}
    for idx in tqdm(index_map, desc='loading'):
        wavid, _, _ = index_map[idx]
        if wavid not in clean_data:
            clean_data[wavid] = wav_store['clean'][wavid][:]
            noisy_data[wavid] = wav_store['noisy'][wavid][:]
    return {'clean': clean_data, 'noisy': noisy_data}


@cost_time
def create_shared_dict(index_map, wav_store):
    manager = Manager()
    shared_dict = manager.dict()
    for idx in tqdm(index_map, desc='loading'):
        wavid, _, _ = index_map[idx]
        if wavid not in shared_dict:
            shared_dict[wavid] = (wav_store['clean'][wavid]
                                  [:], wav_store['noisy'][wavid][:])
    return shared_dict


def create_cache(h5_file, dest='/dev/shm'):
    dest_path = Path(dest).joinpath(getpass.getuser())    
    cache_h5_file = dest_path.joinpath(h5_file)
    cache_h5_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_h5_file.exists() is False:
        logger.info(f'{cache_h5_file} not exists, copying...')
        t = time.perf_counter()
        shutil.copy2(h5_file, cache_h5_file)
        logger.info('cost time: {:.3f} s'.format(time.perf_counter() - t))
    else:
        size_ok = math.isclose(os.path.getsize(cache_h5_file), 
                               os.path.getsize(h5_file))
        time_ok = math.isclose(os.path.getmtime(cache_h5_file),
                               os.path.getmtime(h5_file))  # mtime !
        if size_ok is False or time_ok is False:
            logger.info(f'{cache_h5_file} not match, copying...')
            t = time.perf_counter()
            shutil.copy2(h5_file, cache_h5_file)
            logger.info('cost time: {:.3f} s'.format(time.perf_counter() - t))
    return cache_h5_file


def clear_cache(dest='/dev/shm'):
    user_cache_path = Path(dest).joinpath(getpass.getuser())
    safe_remove(user_cache_path)


def safe_remove(target, safe_path='/dev/shm'):
    if not Path(target).relative_to(safe_path):
        logger.info(f'{target} is not relative to {safe_path}')
        return
    logger.info(f'remove {target}')
    if os.path.isfile(target) or os.path.islink(target):
        os.remove(target)  # remove the file
    elif os.path.isdir(target):
        shutil.rmtree(target)  # remove dir and all contains
    else:
        logger.info(f'{target} is not a file or dir.')
            

def read_csv(csv_file, deny_list=None):
    dataframe = pd.read_csv(csv_file, index_col=0)
    deny_list = []
    if deny_list:
        with open(deny_list) as f:
            deny_list = [item.strip() for item in f.readlines()]
    dataframe = dataframe[~dataframe.index.isin(deny_list)]    
    return dataframe


def get_chunk_info(data_info, chunk_samples=-1, min_chunk=0.5):
    '''get chunk info
    data_info: dataframe or dict
    return: [(wav_id, start, end), ...]
    '''
    if data_info is None:
        return None
    if isinstance(data_info, pd.DataFrame):
        data_info = data_info.to_dict(orient='index')
    chunk_info = []
    for wav_id in data_info:
        if chunk_samples == -1:  # read all samples
            chunk_info.append((wav_id, 0, None))
            continue
        duration = data_info[wav_id]['duration']
        if duration//chunk_samples == 0:
            chunk_info.append((wav_id, 0, None))
            continue
        for j in range(duration//chunk_samples):
            chunk_info.append((wav_id, j*chunk_samples, (j+1)*chunk_samples))
        if duration % chunk_samples > min_chunk * chunk_samples:  # the remainder
            chunk_info.append((wav_id, (j+1)*chunk_samples, None))    
    return chunk_info


def random_choice(id_list, deny_list=None, n=1):
    chosen_list = []
    for _ in range(max(len(id_list), 200)):
        chosen_id = random.choice(id_list)
        if deny_list and chosen_id in deny_list:
            continue
        chosen_list.append(chosen_id)
        if len(chosen_list) == n:
            break
    assert len(chosen_list) == n
    return chosen_list[0] if n == 1 else chosen_list


def build_fbank_extractor(**kwargs):
    """Build a fbank feature extractor for extracting features.
    Ref: https://kaldifeat.readthedocs.io/en/latest/usage/fbank_options.html
    Returns:
        Return a fbank feature extractor.
    """
    import kaldifeat
    opts = kaldifeat.FbankOptions()
    opts.frame_opts.window_type = kwargs['window_type']        
    opts.frame_opts.dither = kwargs['dither']
    opts.frame_opts.samp_freq = kwargs['sample_frequency']
    opts.frame_opts.frame_length_ms = kwargs['frame_length']
    opts.frame_opts.frame_shift_ms = kwargs['frame_shift']
    opts.mel_opts.low_freq = kwargs['low_freq']
    opts.mel_opts.high_freq = kwargs['high_freq']        
    opts.mel_opts.num_bins = kwargs['num_mel_bins']
    opts.htk_compat = kwargs['htk_compat']
    fbank = kaldifeat.Fbank(opts)
    return fbank


def get_phone_info(text_info, units):
    '''Get phone_id (i.e., CTC lable) from text_scp
    text_info: text scp or dict, wav_id -> phone
    units: phone -> phone_id
    return: {wav_id: phone_id}
    '''
    text_dict = {}
    if isinstance(text_info, dict):
        text_dict = text_info
    else:
        with open(text_info) as f:
            for line in f.readlines():
                utt_name, phone, n_concat = line.strip().split(' ')
                utt_name = utt_name.replace('.wav', '')
                text_dict[utt_name] = phone
    phone_dict = {}  # {phone: phone_id}
    with open(units) as f:
        for line in f.readlines():
            phone, phone_id = line.strip().split(' ')
            phone_dict[phone] = int(phone_id)
    phone_info = {}  # {wav_id: phone_id_list}
    for wav_id in text_dict:
        phone = text_dict[wav_id]
        phone_list = sum([item.split('_')for item in phone.split('|')], [])
        phone_id_list = [int(phone) if re.match(r'^\d', phone) else phone_dict[phone] for phone in phone_list]
        phone_info[wav_id] = np.array(phone_id_list, dtype='int32')
    return phone_info


def transcript_to_phone_id(utterance, cmd, lexicon, units):
    '''convert utterance transcript to phone_id (i.e., CTC lable)
    utterance: utt_id -> utt_text
    cmd: cmd or cmd -> phone
    lexicon: text -> phone 
    units: phone -> phone_id
    return: {utt_id: phone_id}
    '''
    phone_id = {}
    return phone_id


def feature_splice(xs, splice):
    '''Do frame expansion for longer context
    xs: shape of [n_frame, n_feats]
    splice: left, right
    return: shape of [n_frame, n_feats*(left+1+right)]
    '''
    left, right = splice
    assert left >= 0 and right >= 0, 'The splice {} is invalid'.format(splice)
    if left == 0 and right == 0:
        return xs

    padded_xs = np.pad(xs, [(left, right), (0, 0)], mode='edge')

    def sliding_window(a, window, step_size):
        '''Reshape a numpy array 'a' of shape (n, x) to form shape((n - window_size + 1), window_size, x))'''
        shape = a.shape[:-1] + (a.shape[-1] - window + 2 - step_size, window)
        shape = (a.shape[0] - window + 2 - step_size, window) + a.shape[1:]
        strides = (a.strides[0] * step_size,) + a.strides
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    window_ys = sliding_window(padded_xs, (left + right + 1), 1)
    flatten_ys = window_ys.reshape(xs.shape[0], -1)
    return flatten_ys


def feature_splice_tensor(xs, splice):
    '''Do frame expansion for longer context
    xs: shape of [batch_size, n_frame, n_feats]
    splice: (left, right)
    return: shape of [batch_size, n_frame, n_feats*(left+1+right)]
    '''
    left, right = splice
    assert left >= 0 and right >= 0, 'The splice {} is invalid'.format(splice)
    if left == 0 and right == 0:
        return xs

    # Pad the tensor along the frame dimension (dimension 1)
    # The padding mode 'replicate' duplicates the border frames
    padded_xs = F.pad(xs, (0, 0, left, right), mode='replicate')    
    batch_size, n_frame, n_feats = xs.shape
    context_size = left + right + 1

    # Use unfold to create the spliced features. Unfold generates a sliding window view on the padded tensor.
    spliced_xs = padded_xs.unfold(dimension=1, size=context_size, step=1)

    # Permute and reshape the tensor to get the final spliced feature tensor
    spliced_xs = spliced_xs.permute(0, 1, 3, 2).reshape(batch_size, n_frame, -1) 

    return spliced_xs


def apply_skip_frame(xs, skip_frame=1):
    '''Apply skip frame
    xs: shape of [n_frame, n_feats], or [n_batch, n_frame, n_feats]
    '''
    if xs.ndim == 2:
        xs = xs[::skip_frame]
    elif xs.ndim == 3:
        xs = xs[:,::skip_frame]
    else:
        raise ValueError(f'xs.ndim error: {xs.ndim}')
    return xs
