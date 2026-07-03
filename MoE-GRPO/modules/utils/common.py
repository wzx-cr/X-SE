import shutil
import getpass
from copy import deepcopy
import uuid
import pickle
from copy import deepcopy
import concurrent.futures
try:
    from nvitop import Device
except ModuleNotFoundError:
    Device = None
from collections import deque, Counter
from itertools import chain
from sys import getsizeof, stderr
from omegaconf import OmegaConf, DictConfig
import re
import argparse
import librosa
from torch.utils.tensorboard import SummaryWriter
from rich import print as rprint
import sys
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn.functional as F
from pathlib import Path
from platform import node
import soundfile as sf
import wave
try:
    import resource
except ModuleNotFoundError:
    resource = None
import time
import seaborn
seaborn.set_theme(
    style='whitegrid',
    font_scale=1.2,
)


EPS = torch.finfo(torch.float32).eps
TINY = torch.finfo(torch.float32).tiny
INT16_MAX = torch.iinfo(torch.int16).max


def read_scp(file):
    regex = re.compile('.CH[0-6]')
    content = dict()
    with open(file) as f:
        all = f.readlines()
        for line in all:
            line_split = line.strip().split(' ')
            wav_id = line_split[0]
            if '.CH' in wav_id:
                wav_id = regex.sub('', wav_id)
            content[wav_id] = ' '.join(line_split[1:])
    return content


class DualOutput(object):
    def __init__(self, terminal, log_file, mode='a'):
        self.terminal = terminal
        self.log = open(log_file, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def str2bool(v):
    if v == None:
        return v
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_resource():
    if resource is None:
        return
    _RLIMIT_NOFILE = resource.getrlimit(resource.RLIMIT_NOFILE)
    __RLIMIT_NOFILE = (
        max(min(_RLIMIT_NOFILE[1]//2, 8192), _RLIMIT_NOFILE[0]), _RLIMIT_NOFILE[1])
    resource.setrlimit(resource.RLIMIT_NOFILE, __RLIMIT_NOFILE)
    # rprint('RLIMIT_NOFILE: {} --> {}'.format(_RLIMIT_NOFILE, resource.getrlimit(resource.RLIMIT_NOFILE)))


def set_proxy(url=None):
    url = url or os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    if not url:
        return
    os.environ['http_proxy'] = url
    os.environ['https_proxy'] = url
    os.environ.setdefault('no_proxy', '10.*,192.*,127.*')
    print(f'set_proxy: {url}')


class compact_dict(dict):
    # print in a compact format
    def __str__(self):
        return str({k: round(v, 3) \
            if isinstance(v, float) or isinstance(v, np.float32) \
                else v for k, v in self.items()})


def tb_plot(x, log_dir='./debug/tensorboard/'):
    x = x.squeeze()
    assert x.ndim == 1
    writer = SummaryWriter(log_dir=log_dir)
    for i, item in enumerate(x):
        writer.add_scalar('x', item, i)
    writer.close()


def torch_float32(x, device=None):
    '''Ensure array/tensor is a float32 tf.Tensor.'''
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    return x


def torch_int32(x, device=None):
    '''Ensure array/tensor is a int32 tf.Tensor.'''
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.int32, device=device)
    return x


def plot_data(x, file_name, title=None, out_dir='debug', width=8):
    x = x.squeeze()
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    fig = plt.figure(figsize=(width, width//2))
    if len(x.shape) == 1:
        plt.plot(x)
    else:
        plt.imshow(x, aspect='auto', interpolation='nearest', cmap=('jet'))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if file_name is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, file_name), dpi=150)
        plt.close()
    return fig


def plot_multi_data(x_list, file_name=None, out_dir='debug', width=8):
    '''
    x_list: [x, title]
    '''
    N = len(x_list)
    for (x, title) in x_list:
        if 'loss' in title.lower():
            N = N + 1
    fig = plt.figure(figsize=(width, 2 * N))
    row_idx = 0
    for (x, title) in x_list:
        x = x.squeeze()
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if 'loss' in title.lower():
            ax = plt.subplot2grid((N, 1), (row_idx, 0), rowspan=2, xmargin=0)
            row_idx = row_idx + 2
        else:
            ax = plt.subplot2grid((N, 1), (row_idx, 0), xmargin=0)
            row_idx = row_idx + 1
        if len(x.shape) == 1:
            ax.plot(x)
        else:
            im = ax.imshow(x, aspect='auto',
                           interpolation='nearest', cmap=('jet'))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05)
            fig.colorbar(im, ax=ax, cax=cax)
            ax.invert_yaxis()
        ax.set_title(title)
    plt.tight_layout()
    if file_name is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, file_name), dpi=150)
        plt.close()
    return fig


def write_wav(wave, file_name, fs=16000, out_dir='debug'):
    sf.write(os.path.join(out_dir, file_name),
        wave.detach().cpu().numpy().reshape(-1), fs)


def merge_dicts(dict1, dict2):
    """
    递归合并两个字典。
    
    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :return: 合并后的字典
    """
    if isinstance(dict1, DictConfig):
        dict1 = OmegaConf.to_container(dict1, resolve=True)
    if isinstance(dict2, DictConfig):
        dict2 = OmegaConf.to_container(dict2, resolve=True)
    merged = dict1.copy()  # 创建 dict1 的副本以避免修改原始字典
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)  # 递归合并嵌套的字典
        else:
            merged[key] = value  # 覆盖或添加新值
    return merged


def yaml_load(cfg='config.yaml'):
    with open(cfg) as stream:
        param = yaml.safe_load(stream)
    return param


def get_wav(wav_dir, wav_csv):
    if not os.path.exists(wav_csv):
        wavlist = list(Path(wav_dir).expanduser().rglob('*.wav'))
        wavidlist = [os.path.basename(item)[:-4] for item in wavlist]
        df = pd.DataFrame(data={'wav': wavlist}, index=wavidlist)
        df.to_csv(wav_csv, sep=' ', header=None)
    else:
        # print(wav_csv, 'exist')
        df = pd.read_csv(wav_csv, index_col=0, sep=' ',
                         names=['wav'], dtype='str')
    return df


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print('{}, cost time: {:.3f} s'.format(
            func.__name__, time.perf_counter() - t))
        return result

    return fun


def read_wav(wav_file):
    '''Wav file should be: 16bits, 16000Hz, mono
    update: equal to sf.read(wav_file, dtype='int16')
    '''
    with wave.open(wav_file, 'rb') as f:
        assert f.getsampwidth() == 2, 'Sample width error'
        assert f.getframerate() == 16000, 'sampling frequency error'
        assert f.getnchannels() == 1, 'Channel error'
        wave_data = f.readframes(f.getnframes())
    return np.frombuffer(wave_data, dtype=np.short)


def get_size(x, unit='M'):
    n_Bytes = sys.getsizeof(x.storage())
    if unit == None:
        pass
    if unit == 'M':
        print('{:.2f} M Bytes'.format(n_Bytes/1024/1024))
    elif unit == 'K':
        print('{:.2f} K Bytes'.format(n_Bytes/1024))
    else:
        print('{:.2f} Bytes'.format(n_Bytes))
    return n_Bytes


# https://code.activestate.com/recipes/577504/
try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    ''' Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    '''
    def dict_handler(d): return chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    # estimate sizeof object without __sizeof__
    default_size = getsizeof(0)

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def get_edge_win(window, n):
    '''get edge window for waveform reconstruction

    Args:
        window (tensor): original window
        N (int): desired window length
    '''
    edge_win = torch.ones(n).type_as(window)
    mid = len(window)//2
    edge_win_temp = 1/(window**2)
    edge_win[:mid] = edge_win_temp[:mid]
    edge_win[-mid:] = edge_win_temp[mid:]
    return edge_win


def formatSize(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print('传入的字节格式不对')
        return 'Error'

    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return '%.2fG' % (G)
        else:
            return '%.2fM' % (M)
    else:
        return '%.2fkb' % (kb)


def getFileSize(path):
    try:
        size = os.path.getsize(path)
        return formatSize(size)
    except Exception as err:
        print(err)


def getDirSize(path):
    sumsize = 0
    try:
        filename = os.walk(path)
        for root, dirs, files in filename:
            for fle in files:
                size = os.path.getsize(path + fle)
                sumsize += size
        return formatSize(sumsize)
    except Exception as err:
        print(err)


def get_ckpt(model_dir, cmd='train'):
    if cmd in ['train', 'last']:
        # sort the newest to the head
        ckpt_list = sorted(Path(model_dir).rglob('*.ckpt'),
                           key=os.path.getmtime, reverse=True)
        ckpt_path = None if len(ckpt_list) == 0 else str(
            ckpt_list[0])  # the latest
    else:
        best_k = Path(model_dir).joinpath('best_k_models.yaml')
        if best_k.exists():
            with open(best_k, 'r') as f:
                ckpt_dict = yaml.load(f, Loader=yaml.FullLoader)
            # sort the best to the head
            ckpt_list = sorted(ckpt_dict.items(), key=lambda item: item[1])
            ckpt_path = None if len(ckpt_list) == 0 else str(
                ckpt_list[0][0])  # the best
            if ckpt_path is None:
                # call self to return the latest
                return get_ckpt(model_dir, 'train')
            ckpt_path = ckpt_path[ckpt_path.find('exp'):]
        else:
            # call self to return the latest
            return get_ckpt(model_dir, 'train')
    return ckpt_path


def check_model_isnan(model_weight):
    for key in model_weight:
        if torch.isnan(model_weight[key]).any(): 
            return True
    return False


def average_model(conf, dest="/dev/shm"):
    ckpt_list = sorted(Path(conf["root_dir"]).rglob("epoch*.ckpt"), key=os.path.getmtime, reverse=True)
    ckpt_list = ckpt_list[:conf.get('n_last', 3)]
    if len(ckpt_list) == 0:
        return None
    rprint(str([item.name for item in ckpt_list]))
    avg = None
    num = len(ckpt_list)
    for path in ckpt_list:        
        model = torch.load(path, map_location=torch.device('cpu'))
        states = model['state_dict']
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    model['state_dict'] = avg # the last model
    if dest:
        dest_path = Path(dest).joinpath(getpass.getuser())
        dest_path.mkdir(exist_ok=True)
        avg_ckpt = dest_path.joinpath('avg_model.ckpt')
    else:
        avg_ckpt = ckpt_list[0].parent.joinpath('avg_model.ckpt')
    rprint('Saving to {}'.format(avg_ckpt))
    torch.save(model, avg_ckpt)
    return avg_ckpt


def refine_state_dict(state_dict, prefix='model.'):
    state_dict_new = {}
    for key in state_dict:
        if key.startswith(prefix):
            new_key = key.replace(prefix, '')
            state_dict_new[new_key] = deepcopy(state_dict[key])
    return state_dict_new


def copy_state_dict(dest_dict, src_dict, prefix='model.'):
    state_dict = deepcopy(dest_dict)
    keys = set(dest_dict.keys()) & set(src_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            state_dict[key] = deepcopy(src_dict[key]).to(state_dict[key].device)
    return state_dict


def load_ckpt(ckpt_path, pre_train_model=None, cache='/tmp'):
    # try to load ckpt with pre_train_model
    if pre_train_model is None:
        return ckpt_path
    save_ckpt = Path(cache).joinpath(getpass.getuser(), 'ckpt', f"{uuid.uuid1()}.ckpt")
    save_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model_ckpt = torch.load(ckpt_path)
    model_state_dict = torch.load(pre_train_model)
    assert 'state_dict' in model_state_dict
    model_ckpt['state_dict'] = copy_state_dict(model_ckpt['state_dict'], 
                                               model_state_dict['state_dict']) # set state_dict
    torch.save(model_ckpt, save_ckpt)
    return save_ckpt


def truncate(est, ref):
    # shape [n_batch, n_samples]
    est = est[:, :ref.shape[1]]
    ref = ref[:, :est.shape[1]]
    return est, ref


def freeze_model(model, check=False):
    for param in model.parameters():
        param.requires_grad = False
    if check is True:
        freezed_num, pass_num = 0, 0
        for param in model.parameters():
            if param.requires_grad == False:
                freezed_num += 1
            else:
                pass_num += 1
        print('Total {} params, miss {}'.format(
            freezed_num + pass_num, pass_num))
    return model


def read_wav_scp(wav_scp, root_path=None):
    if root_path is not None and Path(wav_scp).exists() is False:
        wav_scp = root_path.joinpath(wav_scp)
    with open(wav_scp) as f:
        all_lines = f.readlines()
    wav_scp_dict = {}
    for item in all_lines:
        wav_id, wav_file = item.strip().split()
        if root_path is not None and Path(wav_file).exists() is False:
            wav_path = root_path.joinpath(wav_file)
        else:
            wav_path = Path(wav_file)
        wav_scp_dict[wav_id] = wav_path
    return wav_scp_dict


def is_slurm():
    slurm_env_vars = ['SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_NODELIST', 'SLURM_NTASKS']
    return any(var in os.environ for var in slurm_env_vars)


def is_docker():
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return any('docker' in line for line in f)
    except FileNotFoundError:
        return False


def is_kubernetes():
    if 'KUBERNETES_SERVICE_HOST' in os.environ:
        return True
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return any('kubepods' in line for line in f)
    except FileNotFoundError:
        return False
    
    
def is_unique_job():
    # check whether the current job is the unique job on the node
    if not is_slurm():
        return False
    _node = node()
    cmd = "squeue -u `whoami`|grep 'kshdnormal'| awk '{print $8}'"
    p = os.popen(cmd)
    cmd_ret = p.read()
    p.close()
    node_list = cmd_ret.split()
    # print('Host list:', node_list)
    node_dict = Counter(node_list)
    if _node in node_dict:
        if node_dict[_node] == 1:
            return True
    return False


def find_available_gpu(n_gpu=1, gpu_mem=8, a_wait=-1):
    # gpu_mem: xx G, -1 找没有计算任务的卡
    if Device is None:
        rprint('nvitop is not installed; skip automatic GPU selection.')
        return ''
    mem_G = 1024**3
    devices = Device.all()

    def _get_process_num(device, p_type='C'):
        cnt = 0
        processes = device.processes()
        for pid in processes:
            if processes[pid].type == p_type:
                cnt += 1
        return cnt

    def _get_valiable_list():
        avaiable_list = []
        for device in devices:
            if gpu_mem != -1:
                if device.memory_free()/mem_G > gpu_mem:
                    avaiable_list.append(str(device.index))
            else:
                if _get_process_num(device) == 0:
                    avaiable_list.append(str(device.index))
        return avaiable_list
    
    avaiable_list = _get_valiable_list()
    if a_wait > 0:
        while True:
            if len(avaiable_list) >= n_gpu:
                break
            rprint(f'Querying GPU per {a_wait} s ...')
            time.sleep(a_wait)
            avaiable_list = _get_valiable_list()
    return ','.join(avaiable_list[:n_gpu])


def fix_length(data, size, axis=-1, **kwargs):
    '''Fix the length an array ``data`` to exactly ``size`` along a target axis.
    If ``data.shape[axis] < n``, pad according to the provided kwargs.
    By default, ``data`` is padded with trailing zeros.
    '''
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]
    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return F.pad(data, lengths, **kwargs)
    return data


def get_batch_samples():
    y1, sr = librosa.load(librosa.ex('libri1'), sr=16000)
    y2, sr = librosa.load(librosa.ex('libri2'), sr=16000)
    y3, sr = librosa.load(librosa.ex('libri3'), sr=16000)
    n_samples = min(len(y1), len(y2), len(y3))
    y = np.vstack([y1[:n_samples], y2[:n_samples], y3[:n_samples]])
    return y, sr


def get_samples():
    return librosa.load(librosa.ex('libri1'), sr=16000)


def wave_pad_sample(wave, n_sample, mode='reflect'):
    # pad or sample the wave with given n_sample
    n_pad = n_sample - len(wave)
    if n_pad >= 0:
        wave_out = np.pad(wave[:], (0, n_pad), mode=mode)  # pad the wave
    else:
        s = np.random.randint(0, -n_pad)
        wave_out = wave[s: s+n_sample]  # sample the wave
    return wave_out


def get_pool_executor(name='ThreadPoolExecutor', max_workers=None):
    # name: ProcessPoolExecutor, ThreadPoolExecutor
    return getattr(concurrent.futures, name)(max_workers)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return -1


def format_time(seconds):
    # Decompose the total seconds into hours, minutes, seconds, and milliseconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the time string
    formatted_time = f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"
    return formatted_time


class DataSaver:
    def __init__(self, rank=0, save_path=None, k=3):
        self.rank = rank
        if self.rank != 0:
            return
        if save_path is None:
            self.save_path = Path("/tmp") / getpass.getuser() / "pkl"
        else:
            self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.saved_files = {}  # {varname: deque(maxlen=k)}
        self.pool = get_pool_executor()

    def save_data(self, data, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def get_varname(self, file_name):
        # file_name: epoch=1-step=96016-varname.pkl
        start = file_name.rfind("-") + 1
        end = file_name.rfind(".pkl")
        return file_name[start:end]

    def save(self, data, file_name):
        if self.rank != 0:
            return        
        name = self.get_varname(file_name)
        if name not in self.saved_files:
            self.saved_files[name] = deque(maxlen=self.k)
        filepath = self.save_path / file_name
        if len(self.saved_files[name]) == self.k:
            self.saved_files[name].popleft().unlink(missing_ok=True)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        self.pool.submit(self.save_data, data, filepath)
        self.saved_files[name].append(filepath)

    def copytree(self, dst="exp/debug"):
        # copy save_path to dst
        if self.rank != 0:
            return
        dst_path = Path(dst) / self.save_path.name
        dst_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.save_path, dst_path, dirs_exist_ok=True)

        
import numpy 
import json
class NumpyEncoder(json.JSONEncoder):  
    def default(self, obj):  
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,  
            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,  
            numpy.uint16, numpy.uint32, numpy.uint64)):  
            return int(obj)  
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,numpy.float64)):  
            return float(obj)  
        elif isinstance(obj, (numpy.ndarray,)):  
            return obj.tolist()  
        return json.JSONEncoder.default(self, obj)


def resolve_path(file_path):
    # Check if the path is absolute or relative to the current directory
    path = Path(file_path)
    if path.exists():
        return path
    
    # Check if the path is relative to the MAIN_ROOT environment variable
    main_root = Path(os.environ.get('MAIN_ROOT', ''))
    path = main_root / file_path
    if path.exists():
        return path
    
    raise ValueError(f'{file_path} does not exist')
