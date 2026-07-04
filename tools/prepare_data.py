from omegaconf import OmegaConf
import shutil
import hashlib
import soundfile as sf
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import io
from modules.utils.logging import logger
import functools

pd_read_csv = functools.partial(pd.read_csv, index_col=0, sep=' ', 
                                names=['id', 'wav_path', 'dur'])

def get_wav_id_list(csv_file, ):
    df = pd_read_csv(csv_file)
    return [str(item) for item in df.index.to_list()]


def get_wav(wav_path, flag, wav_id=None, path_sep=None):
    if path_sep is None:
        wav, _ = sf.read(wav_path, dtype='int16')
        if wav_id is None:
            wav_id = Path(wav_path).stem
        return wav_id, wav, flag
    # read multi-path and concate the wave
    wav_path_list = wav_path.split(path_sep)
    wav_list = []
    for item in wav_path_list:
        wav, _ = sf.read(item, dtype='int16')
        wav_list.append(wav)
    wav = np.concatenate(wav_list)
    if wav_id is None:
        id_list = [Path(item).stem for item in wav_path_list]
        wav_id = path_sep.join(id_list)
    return wav_id, wav, flag


def _get_wav_dur(wav_path, wav_id=None, ret="sample", fs=None):
    try:
        info = sf.info(wav_path)
    except Exception as e:
        print(e)
        return None
    duration = info.duration if ret == "time" else info.frames
    if fs is not None:
        assert info.samplerate == fs
    if wav_id is None:
        wav_id = wav_path.stem
    return wav_id, wav_path, duration


def get_wav_dur(wav_list, pool, fs=None, id_list=None):
    data_list = []
    future_tasks = []
    if id_list is None:
        for _item in wav_list:
            future_tasks.append(pool.submit(_get_wav_dur, _item, fs=fs))
    else:
        for _wav_path, _id in zip(wav_list, id_list):
            future_tasks.append(pool.submit(_get_wav_dur, _wav_path, _id, fs=fs))
    for _ in tqdm(range(len(future_tasks))):
        f = future_tasks.pop(0)
        result = f.result()
        if result is not None:
            data_list.append(result)
    df = pd.DataFrame(data=data_list, columns=['wav_id', 'wav_path', 'duration'])
    return df


def export_hdf5_group(out_store_file, in_store_file, wavid_list, group, 
                      sample_rate=16000, desc=''):
    '''
    export hdf5 group    
    '''    
    out_store_file = Path(out_store_file).expanduser()
    in_store_file = Path(in_store_file).expanduser()
    assert in_store_file.suffix == '.hdf5'
    assert out_store_file.suffix == '.hdf5'
    try:
        _, in_format, _ = in_store_file.name.rsplit('.', 2)
    except:
        in_format = 'wav'
    _, out_format, _ = out_store_file.name.rsplit('.', 2)
    out_store = h5py.File(out_store_file, 'w')
    in_store = h5py.File(in_store_file, 'r')
    logger.info(f'format: {in_format} -> {out_format}')
    for wav_id in tqdm(wavid_list, desc=desc):
        wav_data = in_store[group][wav_id][()]  # binary data
        if out_format == in_format:
            out_store[wav_id] = wav_data
        elif in_format == 'wav' and out_format == 'flac':
            try:
                with io.BytesIO() as f:
                    sf.write(f, wav_data, sample_rate, format='flac')
                    out_store[wav_id] = np.void(f.getvalue())
            except Exception as e:
                logger.error(f"{wav_id}: {e}")
        elif in_format == 'flac' and out_format == 'wav':
            try:
                with io.BytesIO() as f:
                    data, _ = sf.read(io.BytesIO(np.void(wav_data)), dtype='int16')
                    out_store[wav_id] = data
            except Exception as e:
                logger.error(f"{wav_id}: {e}")
        else:
            logger.error(f'format error: {in_format} -> {out_format}')
    out_store.close()
    
    
def get_md5_code(name):
    md = hashlib.md5()
    md.update(name.encode('utf-8'))
    return md.hexdigest()


def merge_hdf5(conf):
    def _merge_files(file_list, output_file):
        with open(output_file, 'wb') as out_file:
            for file in file_list:
                with open(file, 'rb') as f:
                    shutil.copyfileobj(f, out_file)

    flag = conf.get('flag', 'noise')
    _hdf5_list = conf.get('hdf5_list', '').split(',')
    hdf5_list = [Path(item) for item in _hdf5_list]
    out_dir = Path(conf.get('out_dir', hdf5_list[0].parent))
    out_name = conf.get('out_name', 'merge')
    out_csv_file = out_dir.joinpath(f'{out_name}.csv')
    out_hdf5_file = out_dir.joinpath(f'{out_name}.hdf5')
    if out_csv_file.exists() or out_hdf5_file.exists():
        print(out_csv_file, 'or', out_hdf5_file, 'exists')
        return
    csv_list = [x.with_suffix('.csv') for x in hdf5_list]
    _merge_files(csv_list, out_csv_file)
    shutil.copy(hdf5_list[0], out_hdf5_file)
    writer = h5py.File(out_hdf5_file, "r+")
    for hdf5_file in hdf5_list[1:]:
        # hdf5_file = hdf5_list[1]
        df_noise = pd_read_csv(hdf5_file.with_suffix('.csv'))
        noise_list = df_noise.index.to_list()
        with h5py.File(hdf5_file, "r") as reader:
            for wav_id in tqdm(noise_list):
                wav_data = reader[f'{flag}/{wav_id}'][:]
                writer[f'{flag}/{wav_id}'] = wav_data
    writer.close()


def export_wav(conf):
    root_path = Path(conf['data_dir']).joinpath(conf['name'])
    subset = conf.get('subset', 'aispeech_valid')
    wav_store = h5py.File(root_path.joinpath(f'{subset}.hdf5'), "r")    
    id_list = get_wav_id_list(root_path.joinpath(f'{subset}.csv'))
    group_list = ['clean', 'noisy']
    for group in group_list:
        dest = root_path.joinpath(subset, group)
        dest.mkdir(parents=True, exist_ok=True)
        for wav_id in tqdm(id_list):
            wave = wav_store[f'{group}/{wav_id}'][:]
            sf.write(dest.joinpath(f"{wav_id}.wav"),
                     wave, samplerate=conf.sample_rate)
    wav_store.close()


if __name__ == "__main__":
    conf = OmegaConf.create({
        "yaml": "conf/config.yaml",
        "cmd": "prepare_aispeech_noise",
        "num_workers": 8
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
