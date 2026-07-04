import os
import json
import torch
import functools
import numpy as np
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from rich import print as rprint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as LR_scheduler
from modules.utils.logging import logger
from modules.utils.compute_score import ComputeMOS
import modules.augment as augment
import modules.dataset as dataset
from modules.dataset.feature import FeatExtract
from .utils import metrics, audio, torch_signal as signal
from .utils.common import (INT16_MAX, is_unique_job, compact_dict,
                           get_pool_executor, DataSaver, NumpyEncoder)
from .dataset.common import clear_cache
from . import lr_scheduler as custom_scheduler


class BaseModule(pl.LightningModule):
    '''Base module'''

    def __init__(self, conf):
        super().__init__()
        self.save_conf(deepcopy(conf))
        self.conf = conf
        self.batch_size = conf['data']['dataloader']['batch_size']
        self.sample_rate = conf['data']['sample_rate']
        self.stft_conf = conf['stft']
        self.dataloader_conf = conf['data']['dataloader']        
        self.padding = functools.partial(
            signal.padding, 
            frame_length=conf['stft']['win_length'], 
            hop_length=conf['stft']['hop_length'])
        self.pool = get_pool_executor(**conf['pool'])
        self.augment = self.get_data_augment()
        self.feat_extractor = FeatExtract(conf) if conf['data'].get('feats') else None

        # some default configurations
        self.mask = self.get_mask_conf(conf)
        self.valid_batch_size = conf['data'].get('valid_batch_size', 1)
        self.valid_metric = conf['system'].get('valid_metric', False)
        self.valid_MOS = conf['system'].get('valid_MOS', False)
        if self.valid_MOS:
            self.MOS_worker = ComputeMOS(conf)
        self.test_metric = conf['system'].get('test_metric', True)
        self.test_MOS = self.valid_MOS
        self.debug = conf['system'].get('debug', False)
        self.debug_data = conf['system'].get('debug_data', False)
        self.init_save_wav_conf()
        self.valid_log = functools.partial(
            self.log, on_epoch=True, sync_dist=True, batch_size=self.valid_batch_size)
        self.prepare_data_per_node = True

    def prepare_data(self):
        self.prepare_cache()

    def setup(self, stage):
        save_path = 'exp/debug/pkl'
        if os.getenv('SLURM_JOB_ID'):
            save_path = save_path + '_' + os.getenv('SLURM_JOB_ID')
        self.data_saver = DataSaver(self.global_rank, save_path) if self.debug_data else None

    def save_data(self, data, name):
        if self.data_saver:
            file_name = f"epoch={self.current_epoch}-step={self.global_step}-{name}.pkl"
            self.data_saver.save(data, file_name)

    def copy_data(self, dst='exp/debug'):
        if self.data_saver:
            self.data_saver.copytree(dst)

    def save_conf(self, conf):
        conf.pop('n_gpu', None)
        conf.pop('gpu_mem', None)
        conf.pop('a_wait', None)
        conf.trainer.pop('num_nodes', None)
        conf.trainer.pop('devices', None)
        conf.trainer.pop('accelerator', None)
        conf.trainer.pop('progress_logger', None)
        conf.trainer.pop('enable_progress_bar', None)
        conf.data.pop('cache', None)
        self.save_hyperparameters(conf)

    def get_mask_conf(self, conf):
        model_conf = OmegaConf.to_container(conf['model'], resolve=True)
        if isinstance(model_conf, list):
            return model_conf[0].get('mask', True)
        else:
            return model_conf.get('mask', True)
        
    def init_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def init_save_wav_conf(self):
        save_wav = self.conf.get('save_wav', False)
        # Default configuration
        default_conf = {
            'n_wav': self.conf.get('n_wav', INT16_MAX),
            'est_dir': self.conf.get('est_dir', 'est'),
            'clean_dir': self.conf.get('clean_dir', None),
            'noisy_dir': self.conf.get('noisy_dir', None)
        }
        # Assign configuration based on save_wav
        self.save_wav_conf = default_conf if save_wav else None

    def get_data_augment(self):
        param = self.conf['data'].get('data_augment', None)
        if param:
            augments = []
            if param.remix:
                augments.append(augment.Remix())
            if param.bandmask:
                augments.append(augment.BandMask(**param.bandmask))
            if param.shift:
                augments.append(augment.Shift(**param.shift))
            if param.revecho:
                augments.append(augment.RevEcho(**param.revecho))
            return torch.nn.Sequential(*augments)
        else:
            return None

    def data_augment(self, noisy, clean):
        if self.augment is not None:
            sources = torch.stack([noisy - clean, clean])
            sources = self.augment(sources)
            noise, clean = sources
            noisy = noise + clean
        return noisy, clean
    
    def get_optimizer(self, opt_conf, parameters):
        conf = deepcopy(opt_conf)
        optimizer_name = conf.pop('name', 'AdamW')
        monitor = conf.pop('monitor', 'valid_loss')
        lr_scheduler_conf = conf.pop('lr_scheduler', {})
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(parameters, **conf)
        if lr_scheduler_conf:
            lr_scheduler_name = lr_scheduler_conf.pop('name', 'ReduceLROnPlateau')
            if hasattr(LR_scheduler, lr_scheduler_name):
                lr_scheduler_class = getattr(LR_scheduler, lr_scheduler_name)
            elif hasattr(custom_scheduler, lr_scheduler_name):
                lr_scheduler_class = getattr(custom_scheduler, lr_scheduler_name)
            else:
                logger.error(lr_scheduler_name)
            scheduler = lr_scheduler_class(optimizer, **lr_scheduler_conf)
            return {'optimizer': optimizer, 
                    'lr_scheduler': {'scheduler': scheduler, 'monitor': monitor}}
        else:
            return optimizer
        
    def configure_optimizers(self):
        opt_conf = OmegaConf.to_container(self.conf['optimizer'])
        # by default, there is only one optimizer
        assert isinstance(opt_conf, dict) and 'name' in opt_conf
        return self.get_optimizer(opt_conf, self.parameters())

    def prepare_cache(self):
        cache = self.conf['data'].get('cache', None) # e.g. /dev/shm, /data0
        if cache is None:
            return
        logger.info('prepare_cache --------------')
        if self.conf['data'].get('clear_cache', True):
            if is_unique_job():
                clear_cache(cache)
        self.train_dataloader()
        self.val_dataloader()
        # self.test_dataloader()
        logger.info('prepare_cache end ----------')

    def teardown(self, stage):
        cache = self.conf['data'].get('cache', None) # e.g. /dev/shm, /data0
        if cache is None:
            return
        if not self.trainer.is_global_zero:
            return
        logger.info('teardown -------------------')
        if self.conf['data'].get('clear_cache', True):
            if is_unique_job():
                clear_cache(cache)
        logger.info('teardown end ---------------')

    def train_dataloader(self):
        subset = 'train'        
        name = self.conf['data'][subset]['name']
        p_data = getattr(dataset, name)(self.conf, subset)
        logger.info(f'[{subset}] {type(p_data).__name__}')
        kwargs = OmegaConf.to_container(self.dataloader_conf, resolve=True) # dict
        if 'collate_fn' in kwargs and kwargs['collate_fn'] is not None:
            kwargs['collate_fn'] =  getattr(dataset, kwargs['collate_fn'])(self.conf)
        return DataLoader(p_data, **kwargs)

    def val_dataloader(self):
        subset = 'valid'
        name = self.conf['data'][subset]['name']
        p_data = getattr(dataset, name)(self.conf, subset)
        logger.info(f'[{subset}] {type(p_data).__name__}')
        kwargs = OmegaConf.to_container(self.dataloader_conf, resolve=True) # dict
        kwargs.pop('shuffle', False)
        kwargs['batch_size'] = self.valid_batch_size
        if self.valid_metric is True:
            assert kwargs['batch_size'] == 1
        # if self.valid_batch_size == 1:
        #     kwargs['collate_fn'] = None
        # else:
        if 'collate_fn' in kwargs and kwargs['collate_fn'] is not None:
            kwargs['collate_fn'] = getattr(dataset, kwargs['collate_fn'])(self.conf)
        return DataLoader(p_data, **kwargs)

    def test_dataloader(self):
        subset = 'test'        
        name = self.conf['data'][subset]['name']
        p_data = getattr(dataset, name)(self.conf, subset)
        logger.info(f'[{subset}] {type(p_data).__name__}')
        kwargs = {'num_workers': self.dataloader_conf['num_workers'],
                  'collate_fn': self.dataloader_conf.get('collate_fn')}
        if kwargs['collate_fn'] is not None:
            kwargs['collate_fn'] =  getattr(dataset, kwargs['collate_fn'])(self.conf)
        return DataLoader(p_data, **kwargs)

    def to_numpy(self, data):
        if torch.is_tensor(data):
            return data.squeeze().detach().cpu().numpy()
        return data        
    
    def save_wav_file(self, est_wav, clean_wav, noisy_wav, wavid_list):        
        if not self.save_wav_conf or self.save_wav_conf['n_wav'] < 0:
            return
        root_path = Path(self.conf.get('root_dir', self.trainer.default_root_dir))
        est_dir = self.save_wav_conf['est_dir']
        clean_dir = self.save_wav_conf['clean_dir']
        noisy_dir = self.save_wav_conf['noisy_dir']
        for idx, wavid in enumerate(wavid_list):
            if idx < self.save_wav_conf['n_wav']:
                if est_dir:
                    self.pool.submit(audio.audiowrite, root_path.joinpath(est_dir, f'{wavid}.wav'),
                                     self.to_numpy(est_wav[idx, :]), self.sample_rate)
                if noisy_dir is not None:
                    self.pool.submit(audio.audiowrite, root_path.joinpath(noisy_dir, f'{wavid}.wav'),
                                     self.to_numpy(noisy_wav[idx, :]), self.sample_rate)
                if clean_dir is not None and clean_wav is not None:
                    self.pool.submit(audio.audiowrite, root_path.joinpath(clean_dir, f'{wavid}.wav'),
                                     self.to_numpy(clean_wav[idx, :]), self.sample_rate)

    def unpack_wav_batch(self, batch):
        return batch.get('clean_wav'), batch['noisy_wav'], batch['wav_id']
    
    def training_step(self, batch, batch_idx):
        clean_wav, noisy_wav, _ = self.unpack_wav_batch(batch)
        noisy_wav, clean_wav = self.data_augment(noisy_wav, clean_wav)
        loss, _  = self.forward(noisy_wav, clean_wav)
        self.log('train_loss', loss)
        return loss
    
    def prepare_valid_test(self):
        # prepare validation or test
        self.future_tasks = []
        self.MOS_list = []
    
    def process_valid_test(self, batch, metric=False, MOS=False):
        # the common step for validation an test
        clean_wav, noisy_wav, wavid_list = self.unpack_wav_batch(batch)
        if clean_wav is None:
            est_wav = self.forward(self.padding(noisy_wav), train=False)
            if isinstance(est_wav, tuple):
                est_wav = est_wav[0]
            loss = torch.zeros((), device=noisy_wav.device)
        else:
            loss, est_wav = self.forward(self.padding(noisy_wav), self.padding(clean_wav))
        est_wav = est_wav[:, 0:noisy_wav.shape[1]]  # remove the padding
        if metric is True and clean_wav is not None:
            for idx in range(clean_wav.shape[0]): # the batch dimension
                self.future_tasks.append(
                    self.pool.submit(metrics.eval, clean_wav[idx, :].cpu(), est_wav[idx, :].cpu()))
        if MOS is True:
            self.MOS_list.append(self.MOS_worker.batch_scores(est_wav.cpu()))
        self.save_wav_file(est_wav, clean_wav, noisy_wav, wavid_list)
        return loss
    
    def on_validation_epoch_start(self):
        self.prepare_valid_test()
        
    def validation_step(self, batch, batch_idx):
        loss = self.process_valid_test(batch, self.valid_metric, self.valid_MOS)
        self.valid_log('valid_loss', loss)

    def on_validation_epoch_end(self):
        score_list = []
        for f in self.future_tasks:
            _score = f.result()
            score_list.append((_score['PESQ'], _score['STOI'], _score['eSTOI'], _score['SI_SNR']))
        if score_list:
            rank_metrics = torch.tensor(np.array(score_list), dtype=torch.float32).mean(dim=0)
            self.log_dict({'valid/pesq': rank_metrics[0], 'valid/stoi': rank_metrics[1],
                           'valid/estoi': rank_metrics[2], 'valid/si_snr': rank_metrics[3]})
        elif self.valid_metric is True:
            logger.warning('Skip valid PESQ/STOI/eSTOI/SI_SNR because clean_wav is unavailable.')
        if self.valid_MOS and self.MOS_list:
            # logger.debug("[{}] {}".format(self.local_rank, torch.tensor(np.array(self.MOS_list)).shape))
            MOS_score = torch.tensor(np.array(self.MOS_list).squeeze(1)).mean(dim=0)
            self.log_dict({"valid/MOS_SIG": MOS_score[0], "valid/MOS_BAK": MOS_score[1],
                           "valid/MOS_OVL": MOS_score[2]})

    def on_test_epoch_start(self):
        self.prepare_valid_test()

    def test_step(self, batch, batch_idx):
        loss = self.process_valid_test(batch, self.test_metric, self.test_MOS)
        return loss

    def on_test_epoch_end(self):
        if self.test_metric is not True and not self.test_MOS:
            return
        data_list = []
        for f in self.future_tasks:
            _score = f.result()
            data_list.append((_score['PESQ'], _score['STOI'], _score['eSTOI'], _score['SI_SNR']))
        root_path = Path(self.conf.get('root_dir', self.trainer.default_root_dir))
        if data_list:
            data_gather = self.all_gather(torch.tensor(data_list)) # to tensor first !
            if self.trainer.is_global_zero:
                mean_dim = list(range(data_gather.ndim-1)) # mean over the first two dim
                scores = data_gather.mean(dim=mean_dim)
                scores = scores.cpu().numpy()
                results = {'PESQ': scores[0], 'STOI': scores[1],
                           'eSTOI': scores[2], 'SI_SNR': scores[3]}
                logger.info(compact_dict(results))
                try:
                    json.dump(results, open(root_path.joinpath('test_results.json'), 'w'))
                except Exception as e:
                    print(e)
        elif self.test_metric is True:
            logger.warning('Skip test PESQ/STOI/eSTOI/SI_SNR because clean_wav is unavailable.')

        if self.trainer.is_global_zero and self.test_MOS and self.MOS_list:
            MOS_score = np.array(self.MOS_list).squeeze(1).mean(axis=0)
            results = {'SIG': MOS_score[0], 'BAK': MOS_score[1], 'OVL': MOS_score[2]}
            logger.info(compact_dict(results))
            try:
                json.dump(results, open(root_path.joinpath('DNSMOS_results.json'), 'w'),
                          cls=NumpyEncoder)
            except Exception as e:
                print(e)
