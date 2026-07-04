import os
import time
import inspect
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from pytorch_lightning.utilities.model_summary import ModelSummary
from omegaconf import OmegaConf
from rich import print as rprint
import alpha.enh.system as enh_system
from .utils.init_conf import get_exp_name
from modules.utils.logging import logger
from modules.utils.common import format_time


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, monitor, patience=3, verbose=False, mode='min', **kwargs):
        super().__init__(monitor=monitor, patience=patience, verbose=verbose, 
                         mode=mode, **kwargs)

    def load_state_dict(self, callback_state):
        self.wait_count = callback_state['wait_count']
        self.stopped_epoch = callback_state['stopped_epoch']
        self.best_score = callback_state['best_score']
        # self.patience = callback_state['patience'] # don't load patience


class LoggingCallback(pl.Callback):
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            metrics = trainer.callback_metrics
            if metrics:
                info = ', '.join([f'{k}: {v:.2f}' for k, v in metrics.items() if v is not None and 'train' in k])
                logger.info(f'Epoch {trainer.current_epoch}, {batch_idx + 1}/{trainer.num_training_batches}: {info}')

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.perf_counter() - self.epoch_start_time
        time_str = format_time(epoch_time)
        metrics = trainer.callback_metrics
        if metrics:
            info = ', '.join([
                f'{k}: {v:.2e}' if k.startswith('lr') else f'{k}: {v:.2f}' 
                for k, v in metrics.items() if v is not None])
            logger.info(f'Epoch {trainer.current_epoch}/{trainer.max_epochs}: {info} | time: {time_str}')

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:  # Sanity checking phase, don't logging
            return
        if trainer.val_check_interval == 1.0: # float [0.0, 1.0] or int (default: 1.0, don't logging)
            return
        metrics = trainer.callback_metrics
        if metrics:
            info = ', '.join(
                [f'{k}: {v:.2f}' for k, v in metrics.items() if v is not None and 'valid' in k])
            logger.info(f'Epoch {trainer.current_epoch}: {info}')


def _sanitize_trainer_kwargs(conf):
    trainer_kwargs = OmegaConf.to_container(conf['trainer'], resolve=True)
    trainer_kwargs = trainer_kwargs or {}
    router_grpo = conf.get('router_grpo', {})
    device_map = router_grpo.get('device_map', router_grpo.get('multi_gpu', {})) if router_grpo else {}
    model_parallel = (
        conf['system'].get('name') == 'FrozenExpertRouterGRPO'
        and bool(device_map)
        and bool(device_map.get('enabled', True))
    )
    if model_parallel:
        trainer_kwargs['devices'] = 1
        trainer_kwargs['strategy'] = 'auto'
        trainer_kwargs['use_distributed_sampler'] = False
        logger.info(
            '[FrozenExpertRouterGRPO] model-parallel device_map is enabled; '
            'forcing Lightning trainer to single-process devices=1 strategy=auto.'
        )
    if (conf.get('cmd') == 'test'
            and conf['system'].get('name') == 'FrozenExpertRouterGRPO'
            and conf.get('router_grpo', {}).get('adapt_in_test', True)):
        trainer_kwargs['inference_mode'] = False

    # Backward compatibility: some old configs put early_stop under trainer.
    legacy_early_stop = trainer_kwargs.pop('early_stop', None)
    if legacy_early_stop:
        callbacks_conf = conf.get('trainer_callbacks')
        if callbacks_conf is not None:
            if callbacks_conf.get('early_stop') is None:
                callbacks_conf['early_stop'] = legacy_early_stop
                logger.warning('Found deprecated `trainer.early_stop`; moved to `trainer_callbacks.early_stop`.')
            else:
                logger.warning(
                    'Found deprecated `trainer.early_stop`, but `trainer_callbacks.early_stop` is already set. '
                    'Ignoring deprecated field.')
        else:
            logger.warning('Found deprecated `trainer.early_stop`, but `trainer_callbacks` is missing. Ignoring it.')

    valid_keys = set(inspect.signature(pl.Trainer.__init__).parameters.keys()) - {'self'}
    invalid_keys = [k for k in list(trainer_kwargs.keys()) if k not in valid_keys]
    if invalid_keys:
        for key in invalid_keys:
            trainer_kwargs.pop(key, None)
        logger.warning(f'Ignoring unsupported trainer args: {invalid_keys}')

    return trainer_kwargs


def get_model(conf):
    name = conf['system']['name']
    if hasattr(enh_system, name):
        model = getattr(enh_system, name)(conf=conf)
    else:
        raise ValueError(f'system name error: {name}')
    # Try to initialize model if ckpt None
    if conf['ckpt'] is None:
        model.init_model(OmegaConf.to_container(conf['model']))
    return model


def get_callbacks(conf):
    callbacks_conf = conf['trainer_callbacks']
    checkpoint = ModelCheckpoint(Path(conf['root_dir'])/'checkpoints',
                                 **callbacks_conf['model_checkpoint'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint, lr_monitor]
    early_stop = callbacks_conf.get('early_stop')
    if early_stop:
        if OmegaConf.is_dict(early_stop) or isinstance(early_stop, dict):
            callbacks.append(CustomEarlyStopping(**early_stop))
        else:
            logger.warning('`trainer_callbacks.early_stop` must be a dict config. Ignoring invalid value.')
    progress_bar = callbacks_conf.get('progress_bar')
    if progress_bar:
        callbacks.append(TQDMProgressBar(**progress_bar))
    progress_logger = callbacks_conf.get('progress_logger')
    if progress_logger:
        callbacks.append(LoggingCallback())
    return callbacks


# train model
def train(conf):
    trainer_kwargs = _sanitize_trainer_kwargs(conf)
    exp_name = get_exp_name(conf)
    tb_logger = TensorBoardLogger(save_dir=conf['system']['log_dir'], name=exp_name, 
                                  version=conf['version'], default_hp_metric=False)
    root_dir = conf['root_dir']
    logger.info('root_dir: ' + root_dir)
    callbacks = get_callbacks(conf)
    model = get_model(conf)
    trainer = pl.Trainer(default_root_dir=root_dir, logger=tb_logger, 
                         callbacks=callbacks, **trainer_kwargs)
    trainer.fit(model, ckpt_path=conf['ckpt'])
    if os.path.exists(root_dir):
        checkpoint = callbacks[0]
        checkpoint.to_yaml(os.path.join(root_dir, 'best_k_models.yaml'))
    if conf['system'].get('train_and_test', False):
        trainer.test(model)
    if trainer.is_global_zero and conf['system'].get('dingtalk', False):
        try:
            from tools.dingtalk import dingtalk_metric
            dingtalk_metric(root_dir)
        except Exception as exc:
            logger.warning(f'DingTalk notification skipped: {exc}')
    logger.info('[End] {} {} {}'.format('-'*10, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], '-'*10))


def test(conf):
    ckpt_optional_systems = {'FrozenExpertRouterGRPO', 'FrozenExpertOracleBaseline'}
    if conf['system'].get('name') not in ckpt_optional_systems:
        assert conf['ckpt']
    trainer_kwargs = _sanitize_trainer_kwargs(conf)
    trainer = pl.Trainer(logger=False, **trainer_kwargs)
    model = get_model(conf)
    trainer.test(model, ckpt_path=conf['ckpt'])


def valid(conf):
    assert conf['ckpt']
    trainer_kwargs = _sanitize_trainer_kwargs(conf)
    trainer = pl.Trainer(logger=False, **trainer_kwargs)
    model = get_model(conf)
    trainer.validate(model, ckpt_path=conf['ckpt'])  


def model_summary(conf):
    model = get_model(conf)
    rprint(ModelSummary(model, max_depth=conf.get('max_depth', 1)))
    
