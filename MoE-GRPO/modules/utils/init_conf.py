import os
import torch
from pathlib import Path
from platform import node
from multiprocessing import cpu_count

# Torch / third-party compatibility shims
_pytree = getattr(torch.utils, "_pytree", None)
if _pytree is not None and not hasattr(_pytree, "register_pytree_node"):
    register_impl = getattr(_pytree, "_register_pytree_node", None)

    if register_impl is not None:

        def register_pytree_node(*args, **kwargs):
            return register_impl(*args, **kwargs)

        setattr(_pytree, "register_pytree_node", register_pytree_node)

import pytorch_lightning as pl
from copy import deepcopy
from datetime import datetime
from omegaconf import OmegaConf
from .common import get_ckpt, find_available_gpu, set_resource, is_kubernetes
from modules.utils.logging import setup_root_logger, logger
torch.set_float32_matmul_precision('high')

def get_exp_name(conf):
    if 'exp_name' in conf['system']:
        return conf['system']['exp_name']
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    if isinstance(model_conf, list):
        model_name = "_".join(item['name'] for item in model_conf)
    else:
        model_name = model_conf['name']
    return '{}_{}'.format(conf['system']['name'], model_name)

def set_cuda_visible_devices(conf):
    '''
    Try to find available GPU and set CUDA_VISIBLE_DEVICES    
    '''
    # Slurm will set CUDA_VISIBLE_DEVICES, or use all gpu in kubernetes
    if 'CUDA_VISIBLE_DEVICES' in os.environ or is_kubernetes():
        return
    n_gpu = conf.get('n_gpu', 1)
    gpu_mem = conf.get('gpu_mem', 4)
    a_wait = conf.get('a_wait', 60 if conf.get('a_wait')is True else conf.get('a_wait', -1))
    gpus = find_available_gpu(n_gpu=n_gpu, gpu_mem=gpu_mem, a_wait=a_wait)
    if not gpus:
        logger.error('CUDA_VISIBLE_DEVICES not set, and find_available_gpu() failed.')
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def cuda_is_usable():
    """
    torch.cuda.is_available() may be True even when the runtime is unusable
    (e.g., invalid CUDA_VISIBLE_DEVICES or driver mismatch). Do a minimal
    health check to decide whether to run on GPU.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.current_device()
        # tiny allocation to ensure cudaSetDevice works
        torch.tensor([0], device='cuda')
        return True
    except Exception as e:
        logger.warning(f'CUDA reported available but failed health check: {e}')
        return False

def init_conf(conf):
    # root_dir
    if 'root_dir' not in conf:
        exp_name = get_exp_name(conf)
        conf['root_dir'] = str(Path(conf['system']['log_dir']).joinpath(exp_name, conf['version']))
    Path(conf['root_dir']).mkdir(parents=True, exist_ok=True)
    setup_root_logger(conf['system']['debug'], Path(conf['root_dir']).joinpath('log.log'), conf['cmd'])
    logger.info('[Start] {} {} {}'.format('-'*10, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], '-'*10))
    if os.getenv('SLURM_JOB_ID'):
        logger.info('SLURM_JOB_ID: ' + os.getenv('SLURM_JOB_ID'))
    
    # if ckpt not specified, try to find ckpt in roor_dir
    ckpt = conf.get('ckpt', get_ckpt(conf['root_dir'], conf['cmd']))
    if ckpt is not None:
        conf = OmegaConf.load(Path(conf['root_dir']).joinpath('hparams.yaml'))
        conf.merge_with_cli()  # merge again
    conf['ckpt'] = ckpt  # set ckpt
    logger.info('ckpt: {}'.format(conf['ckpt']))
    
    set_cuda_visible_devices(conf)
    try:
        device_count = torch.cuda.device_count()
    except Exception as e:
        logger.warning(f'Failed to query CUDA device count: {e}')
        device_count = 0
    cuda_ok = cuda_is_usable()
    logger.info('Hostname: {}, cpu_count: {}, CUDA_VISIBLE_DEVICES: [{}], HIP_VISIBLE_DEVICES: [{}], device_count: {}, cuda_ok: {}'.format(
        node(), cpu_count(), os.environ.get('CUDA_VISIBLE_DEVICES', None),  os.environ.get('HIP_VISIBLE_DEVICES', None),
        device_count, cuda_ok))
    
    if not cuda_ok:
        # prevent downstream CUDA placement when driver is unusable
        os.environ['CUDA_UNUSABLE'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # hard-disable CUDA entry points so downstream code can't trigger lazy init
        torch.cuda.is_available = lambda: False  # type: ignore
        torch.cuda.device_count = lambda: 0      # type: ignore
        torch.cuda.current_device = lambda: 0    # type: ignore
        torch.cuda.get_device_properties = lambda *a, **k: None  # type: ignore
        torch.cuda.manual_seed = lambda *a, **k: None  # type: ignore
        torch.cuda.manual_seed_all = lambda *a, **k: None  # type: ignore
        torch.cuda.get_rng_state_all = lambda: []  # type: ignore
        torch.cuda.set_rng_state_all = lambda *a, **k: None  # type: ignore
        torch.cuda._lazy_init = lambda *a, **k: None  # type: ignore
        torch.cuda._initialized = True  # type: ignore
        # pin_memory offers no benefit on CPU and can touch CUDA runtime; disable it
        try:
            conf['data']['dataloader']['pin_memory'] = False
        except Exception:
            pass
    router_grpo = conf.get('router_grpo', {})
    device_map = router_grpo.get('device_map', router_grpo.get('multi_gpu', {})) if router_grpo else {}
    model_parallel = (
        conf['system'].get('name') == 'FrozenExpertRouterGRPO'
        and bool(device_map)
        and bool(device_map.get('enabled', True))
    )
    if model_parallel and cuda_ok:
        # Model-parallel FrozenExpertRouterGRPO needs one process that can see
        # all selected GPUs; launching Lightning DDP would duplicate the router
        # and every expert in each worker.
        conf['trainer']['devices'] = 1
        if conf['trainer'].get('strategy') in (None, 'ddp', 'ddp_find_unused_parameters_true'):
            conf['trainer']['strategy'] = 'auto'
    else:
        conf['trainer']['devices'] = -1 if cuda_ok else 'auto'
    conf['trainer']['accelerator'] = 'gpu' if cuda_ok else 'cpu'    
    set_resource()  # set resource limit
    pl.seed_everything(conf['system'].get('seed', 1234))  # seed
    # when validation is diable, disable ckpt_monitor and early_stop
    if conf['trainer'].get('limit_val_batches', 1.0) == 0:
        callbacks_conf = conf['trainer_callbacks']
        callbacks_conf['model_checkpoint']['monitor'] = None  # disable monitor
        callbacks_conf['model_checkpoint']['save_top_k'] = 1
        callbacks_conf['model_checkpoint']['save_last'] = False
        callbacks_conf['early_stop'] = None  # disable early_stop
    logger.debug(conf['trainer'])
    logger.debug(conf['trainer_callbacks'])
    return conf
