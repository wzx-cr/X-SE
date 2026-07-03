# pytorch map-style dataset
from torch.utils.data import Dataset
import json
import torch
from scipy import signal
import random
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
from modules.utils.logging import logger
from modules.utils import audio
from modules.utils.common import wave_pad_sample
from .common import read_csv, get_chunk_info, get_phone_info
from .feature import FeatExtract
from .data_store import WavStore, FeatStore, LabelStore


class DatasetBase(Dataset):
    def __init__(self, conf, subset):
        '''
        subset: ['train', 'valid', 'test']
        '''
        super().__init__()
        random.seed(conf['system'].get('seed', 0))
        self.conf = conf
        self.subset = subset
        self.pairs = conf['data'].get('pairs', 'noisy2clean')
        self.unpaired = conf['data'].get('unpaired', False)        
        self.mix_param = conf['data'].get('mix_param')
        self.add_rir = self.conf['data'].get('add_rir', 0)
        self.split_wav = conf['data'][subset]['split_wav']
        self.sample_rate = conf['data']['sample_rate']
        self.cache = conf['data'].get('cache', None)
        self.random_chunk_begin = bool(conf['data'][subset].get(
            'random_chunk_begin', conf['data'].get('random_chunk_begin', True)
        ))
        noisy_csv, noisy_store = self.resolve_audio_source('noisy_csv', 'noisy_dir', 'noisy_store', subset)
        clean_csv, clean_store = self.resolve_audio_source('clean_csv', 'clean_dir', 'clean_store', subset)
        noise_csv, noise_store = self.resolve_audio_source('noise_csv', 'noise_dir', 'noise_store')
        self.data_info = {
            'noisy': self.read_data_table(noisy_csv, role='noisy'),
            'clean': self.read_data_table(clean_csv, role='clean'),
            'noise': self.read_data_table(noise_csv, role='noise')
        }
        self.data_store = {
            'noisy': WavStore(noisy_store, self.data_info['noisy'], 
                              cache=self.cache) if noisy_store or noisy_csv else None,
            'clean': WavStore(clean_store, self.data_info['clean'], 
                              cache=self.cache) if clean_store or clean_csv else None,
            'noise': WavStore(noise_store, self.data_info['noise'],
                              cache=self.cache) if noise_store or noise_csv else None,
        }
        if self.split_wav:
            self.chunk_samples = (conf['data']['chunk_size'] - 1) * conf['stft']['hop_length'] + \
                conf['stft']['win_length']
        else:
            self.chunk_samples = -1
        self.chunk_info = {
            'noisy': get_chunk_info(self.data_info['noisy'], self.chunk_samples),
            'clean': get_chunk_info(self.data_info['clean'], self.chunk_samples),
            'noise': get_chunk_info(self.data_info['noise'], self.chunk_samples)
        }
        if self.add_rir > 0:
            rir_csv = self.resolve_path('rir_csv')
            rir_store = self.resolve_path('rir_store')
            self.data_info['rir'] = read_csv(rir_csv) if rir_csv else None
            self.data_store['rir'] = WavStore(
                rir_store, self.data_info['rir'], cache=self.cache) if rir_store or rir_csv else None
            self.update_rir_info(rir_csv.with_name('RIR_table_simple.csv'))
        self.label_dict = None
        if isinstance(self.data_info['noisy'], pd.DataFrame):
            if 'text' in self.data_info['noisy'].columns:
                self.label_dict = get_phone_info(self.data_info['noisy']['text'].to_dict(), 
                                                 self.resolve_path('units'))
            if 'mos' in self.data_info['noisy'].columns:
                self.label_dict = self.data_info['noisy']['mos']
        else:
            label_csv = self.resolve_path('label_csv', subset)
            label_store = self.resolve_path('label_store', subset)
            self.data_info['label'] = read_csv(label_csv) if label_csv else None
            self.data_store['label'] = LabelStore(
                label_store, self.data_info['label']) if label_store or label_csv else None
            # read all label data to label_dict
            if self.data_store['label']:
                self.label_dict = {}
                for wav_id in self.data_store['label'].id_list:
                    self.label_dict[wav_id] = self.data_store['label'].get_data(wav_id)
        self.feat_extractor = FeatExtract(conf) if conf['data'].get('feats') else None
        logger.info(self.get_msg())
        self.filter_data()

    @staticmethod
    def _is_dir_path(path):
        return path is not None and not isinstance(path, list) and Path(path).is_dir()

    def resolve_audio_source(self, csv_key, dir_key, store_key, subset=None):
        audio_dir = self.resolve_path(dir_key, subset)
        store_path = self.resolve_path(store_key, subset)
        csv_path = self.resolve_path(csv_key, subset)

        if self._is_dir_path(audio_dir):
            return audio_dir, None
        if self._is_dir_path(store_path):
            logger.warning(f'{store_key} points to a directory; treating it as raw audio dir: {store_path}')
            return store_path, None
        if self._is_dir_path(csv_path):
            return csv_path, None
        return csv_path, store_path

    def read_data_table(self, file_path, role='noisy'):
        """
        Read csv/tsv/json data list. For json (DNS3 style), expect:
            {
              "123": {"mix": "noisy/xxx.wav", "clean": "clean/xxx.wav", "file_len": 80000, ...},
              ...
            }
        `role` chooses which field to use as wav path.
        """
        if file_path is None:
            return None
        file_path = Path(file_path)
        if file_path.is_dir():
            return self.read_audio_dir(file_path)
        if file_path.suffix.lower() != '.json':
            return read_csv(file_path)

        with open(file_path, 'r') as f:
            meta = json.load(f)

        base_dir = file_path.parent
        records = {}
        wav_key_map = {
            'noisy': ['mix', 'noisy', 'wav'],
            'clean': ['clean', 'target', 'ref'],
            'noise': ['noise'],
        }
        for wav_id, info in meta.items():
            wav_rel = None
            for k in wav_key_map.get(role, ['wav']):
                if k in info:
                    wav_rel = info[k]
                    break
            if wav_rel is None:
                continue  # skip if missing
            wav_path = (base_dir / wav_rel).as_posix()
            record = {'wav_path': wav_path}
            dur = info.get('file_len') or info.get('duration') or info.get('length')
            if dur is not None:
                record['duration'] = int(dur)
            if 'snr' in info:
                record['snr'] = info['snr']
            records[wav_id] = record

        if len(records) == 0:
            logger.warning(f'{file_path} is empty or no usable entries for role={role}')
            return None
        return pd.DataFrame.from_dict(records, orient='index')

    def read_audio_dir(self, wav_dir):
        audio_exts = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}
        records = {}
        wav_dir = Path(wav_dir).expanduser()
        for wav_path in sorted(p for p in wav_dir.rglob('*') if p.is_file() and p.suffix.lower() in audio_exts):
            rel_id = wav_path.relative_to(wav_dir).with_suffix('').as_posix().replace('/', '__')
            try:
                duration = sf.info(str(wav_path)).frames
            except Exception as exc:
                logger.warning(f'skip unreadable audio: {wav_path} ({exc})')
                continue
            records[rel_id] = {
                'wav_path': str(wav_path),
                'duration': int(duration),
            }
        if len(records) == 0:
            raise ValueError(f'No audio files found under: {wav_dir}')
        logger.info(f'Loaded {len(records)} audio files from directory: {wav_dir}')
        return pd.DataFrame.from_dict(records, orient='index')
        
    def get_msg(self):
        msg = f'[{self.subset}] chunk_samples: {self.chunk_samples}; '
        len_info = [len(self.chunk_info[key]) if self.chunk_info[key] else 0 for key in self.chunk_info]
        msg = msg + '|n_chunk|' + f' noisy: {len_info[0]}, clean: {len_info[1]}, noise: {len_info[2]}'
        dur_msg = ''
        for key in self.data_info:
            if key not in ['noisy', 'clean', 'noise', 'rir']:
                continue
            if self.data_info[key] is not None:
                dur_msg = dur_msg + ' [{}] max: {:.2f} s, min: {:.2f} s, mean: {:.2f} s'.format(
                    key, 
                    self.data_info[key]['duration'].max()/self.sample_rate,
                    self.data_info[key]['duration'].min()/self.sample_rate,
                    self.data_info[key]['duration'].mean()/self.sample_rate)
        if len(dur_msg) > 0:
            msg = msg + '; |duration|' + dur_msg
        if self.unpaired:
            msg = msg + f'; unpaired: {self.unpaired}'
        return msg
    
    def filter_data(self):
        # Filter data, and update data_info and data_store
        subset_conf = OmegaConf.to_container(self.conf['data'][self.subset], resolve=True)
        filter_config = subset_conf.get('filter')
        mos_filter = subset_conf.get('mos_filter')
        if not filter_config and mos_filter is None:
            return

        df = self.data_info['noisy']
        if df is None:
            df = self.data_info['clean']
        if df is None:
            return

        df_filtered = df
        filter_msg = []

        # Duration-based filtering
        if filter_config:
            assert isinstance(df, pd.DataFrame) and 'duration' in df.columns, \
                "DataFrame must contain 'duration' column."
            if isinstance(filter_config, list):
                lower_bound, upper_bound = filter_config
            elif isinstance(filter_config, float):
                mid_quantile = filter_config
                lower_q = (1 - mid_quantile) / 2
                upper_q = mid_quantile + lower_q
                lower_bound = df['duration'].quantile(lower_q)
                upper_bound = df['duration'].quantile(upper_q)
            else:
                raise ValueError(f'Invalid filter configuration: {filter_config}')
            df_filtered = df_filtered[(df_filtered['duration'] >= lower_bound) &
                                      (df_filtered['duration'] <= upper_bound)]
            filter_msg.append(f'duration:[{lower_bound:.0f},{upper_bound:.0f}]')

        # MOS-based filtering (only valid when noisy metadata contains "mos")
        if mos_filter is not None:
            noisy_df = self.data_info.get('noisy')
            if noisy_df is None or 'mos' not in noisy_df.columns:
                logger.warning(
                    f'[{self.subset}] mos_filter is set but no "mos" column exists in noisy metadata. '
                    'Skip MOS filter.')
            else:
                noisy_df = noisy_df[noisy_df.index.isin(df_filtered.index)]
                before_mos_n = len(noisy_df)
                if isinstance(mos_filter, list):
                    assert len(mos_filter) == 2, 'mos_filter list must be [lower, upper]'
                    mos_lower, mos_upper = float(mos_filter[0]), float(mos_filter[1])
                    noisy_df = noisy_df[(noisy_df['mos'] >= mos_lower) &
                                        (noisy_df['mos'] <= mos_upper)]
                    filter_msg.append(f'mos:[{mos_lower:.2f},{mos_upper:.2f}]')
                elif isinstance(mos_filter, (int, float)):
                    mos_lower = float(mos_filter)
                    noisy_df = noisy_df[noisy_df['mos'] >= mos_lower]
                    filter_msg.append(f'mos:>={mos_lower:.2f}')
                else:
                    raise ValueError(f'Invalid mos_filter configuration: {mos_filter}')

                after_mos_n = len(noisy_df)
                logger.info(
                    f'[{self.subset}] MOS filtered: {before_mos_n} -> {after_mos_n} '
                    f'(drop {before_mos_n - after_mos_n})')
                df_filtered = df_filtered[df_filtered.index.isin(noisy_df.index)]

        for key in ['noisy', 'clean']:
            _df = self.data_info.get(key)
            if _df is not None:
                self.data_info[key] = _df[_df.index.isin(df_filtered.index)]
                self.chunk_info[key] = get_chunk_info(self.data_info[key], self.chunk_samples)
                self.data_store[key].update_info(self.data_info[key])
        logger.info(f'[Filtered] {self.get_msg()}')
        if filter_msg:
            logger.info(f'[{self.subset}] filter rules -> {", ".join(filter_msg)}')
        
    def resolve_path(self, key, subset=None):
        """
        Resolve the path based on the key and subset.

        Args:
            key (str): The key to lookup in the configuration.
            subset (str, optional): The subset to lookup in the configuration.

        Returns:
            str or list of str: The resolved path(s).
        """
        sub_conf = self.conf['data'][subset] if subset else self.conf['data']
        file_path = sub_conf.get(key)
        if file_path is None:
            logger.debug(f'return None: {key}, {subset}')
            return None
        data_dir = Path(self.conf['data']['data_dir']).expanduser()

        def _resolve_one(fp):
            fp = Path(fp).expanduser()
            if fp.is_absolute():
                return fp
            # If path already contains `data_dir` prefix, do not join again.
            data_dir_parts = data_dir.parts
            if len(fp.parts) >= len(data_dir_parts) and fp.parts[:len(data_dir_parts)] == data_dir_parts:
                return fp
            # Keep existing relative paths (resolved from current working directory).
            if fp.exists():
                return fp
            return data_dir.joinpath(fp)

        if isinstance(file_path, list) or OmegaConf.is_list(file_path):
            return [_resolve_one(item) for item in file_path]
        else:
            return _resolve_one(file_path)
    
    def read_csv(self, csv_file):
        """
        Read one or multiple CSV files into a pandas DataFrame.

        Args:
            csv_file (Path or list of Path): The path(s) to the CSV file(s).

        Returns:
            pd.DataFrame: The data from the CSV file(s).
        """
        if not csv_file:
            return None
        if isinstance(csv_file, list):
            df_list = [read_csv(item).assign(store_idx=i) for i, item in enumerate(csv_file)]            
            return pd.concat(df_list)
        else:
            return read_csv(csv_file)
    
    @property
    def snr(self):
        return random.randint(self.mix_param['snr_lower'], self.mix_param['snr_upper'])

    def get_chunk(self, index, key='noisy', min_chunk=0.5):
        index = index % len(self.chunk_info[key]) # remainder
        wav_id, begin, end = self.chunk_info[key][index]
        if end is None:
            return wav_id, begin, end
        if not self.random_chunk_begin:
            return wav_id, begin, end
        max_begin = begin + min_chunk * self.chunk_samples
        begin_random = random.randint(begin, max_begin)  # random begin
        return wav_id, begin_random, begin_random + self.chunk_samples

    # @logger.catch
    def get_wav_data(self, wav_id, key):
        return self.data_store[key][wav_id]

    def check_wav_data(self, wav_data):
        n_pad = self.chunk_samples - wav_data.shape[-1]
        if n_pad > 0 and 'test' not in self.subset:
            wav_data = np.pad(wav_data, (0, n_pad))
        return wav_data
    
    @property
    def rir(self):
        return self.get_rir_data()

    def update_rir_info(self, rir_csv_ext):
        def wavfile2id(wavfile):
            wav_id = '_'.join(wavfile.split('/')[2:])
            return wav_id.replace('.wav', '')
        ext_info = pd.read_csv(rir_csv_ext)
        msg = 'rir.csv: {}, {}: {}'.format(
            len(self.data_store['rir'].id_list), rir_csv_ext.name, len(ext_info))
        ext_info = ext_info[ext_info['T60_WB'] >= self.mix_param['lower_t60']]
        ext_info = ext_info[ext_info['T60_WB'] <= self.mix_param['upper_t60']]
        ext_info['wav_id'] = ext_info['file'].apply(wavfile2id)
        rir_info = self.data_info['rir']
        rir_info_new = rir_info[rir_info.index.isin(ext_info['wav_id'])]
        logger.info(f'{msg}, filtered: {len(rir_info_new)}')
        self.data_info['rir'] = rir_info_new
        self.data_store['rir'].update_info(rir_info_new)

    def get_rir_data(self, chunk_samples=None):
        rir_data = self.data_store['rir'].sample_data()
        if rir_data.ndim > 1:
            rir_data = rir_data[:, 0] # only use the first channel
        if chunk_samples is not None:
            rir_data = wave_pad_sample(rir_data, chunk_samples)
        return rir_data

    def add_pyreverb(self, clean_speech):
        reverb_speech = signal.fftconvolve(clean_speech, self.rir, mode='full')
        reverb_speech = reverb_speech[0: clean_speech.shape[0]]
        return reverb_speech

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
       
    
class WavDataset(DatasetBase):
    '''map-style dataset'''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noisy']
        
    def __getitem__(self, index):
        wav_id, begin, end = self.get_chunk(index)
        noisy_wav = self.get_wav_data(wav_id, 'noisy')
        noisy_wav = self.check_wav_data(noisy_wav[begin:end])
        data = {'noisy_wav': noisy_wav, 'wav_id': wav_id}
        if self.unpaired is True:
            wav_id = self.data_store['clean'].sample_id(deny_list=[wav_id])
        if self.chunk_info['clean'] is not None:
            clean_wav = self.get_wav_data(wav_id, 'clean')
            clean_wav = self.check_wav_data(clean_wav[begin:end])
            data['clean_wav'] = clean_wav
        if self.label_dict is not None and len(self.label_dict) > 0:
            data['label'] = self.label_dict[wav_id]
        return data

    def __len__(self):
        return len(self.chunk_info['clean'] if self.chunk_info['clean'] else self.chunk_info['noisy'])


class WavDataSetTeacher(DatasetBase):
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        teacher_csv = self.resolve_path('teacher_csv', subset)
        teacher_store = self.resolve_path('teacher_store', subset)
        self.data_info['teacher'] = read_csv(teacher_csv) if teacher_csv else None
        self.data_store['teacher'] = WavStore(teacher_store, self.data_info['teacher'], 
                                              cache=self.cache) if teacher_store or teacher_csv else None
        self.chunk_info['teacher'] = get_chunk_info(self.data_info['teacher'], self.chunk_samples)
        
    def __getitem__(self, index):
        wav_id, begin, end = self.get_chunk(index)
        noisy_wav = self.get_wav_data(wav_id, 'noisy')
        noisy_wav = self.check_wav_data(noisy_wav[begin:end])
        data = {'noisy_wav': noisy_wav, 'wav_id': wav_id}
        if self.unpaired is True:
            wav_id = self.data_store['clean'].sample_id(deny_list=[wav_id])
        if self.chunk_info['clean'] is not None:
            clean_wav = self.get_wav_data(wav_id, 'clean')
            clean_wav = self.check_wav_data(clean_wav[begin:end])
            data['clean_wav'] = clean_wav
        if self.chunk_info['teacher'] is not None:
            teacher_wav = self.get_wav_data(wav_id, 'teacher')
            teacher_wav = self.check_wav_data(teacher_wav[begin:end])
            data['teacher_wav'] = teacher_wav
        if self.label_dict is not None and len(self.label_dict) > 0:
            data['label'] = self.label_dict[wav_id]
        return data

    def __len__(self):
        return len(self.chunk_info['clean'] if self.chunk_info['clean'] else self.chunk_info['noisy'])
        

class NoiseDataset(DatasetBase):
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noise']

    def get_noise(self, index=None):
        if not index:
            index = random.randint(0, len(self))
        wav_id, begin, end = self.get_chunk(index, 'noise')
        noise = self.get_wav_data(wav_id, 'noise')
        return self.check_wav_data(noise[begin:end])

    def __getitem__(self, index):
        noise = self.get_noise()
        return noise
    
    def __len__(self):
        # should larger than batch size
        return len(self.chunk_info['noise'])
        

class OnlineNoisyDataset(DatasetBase):
    '''
    noisy2clean: clean + noise -> clean
    noisy2noisy: clean + noise1 -> clean + noise2
    noisier2noisy: clean + noise1 + noise2 -> clean + noise1
    noisier2noisy_v2: noisy + noise -> noisy
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        if self.pairs == 'noisier2noisy_v2':
            assert (self.data_store['noisy'] and self.data_store['noise'])
        else:
            assert (self.data_store['clean'] and self.data_store['noise'])
        if self.add_rir > 0:
            assert self.data_store['rir']
        logger.info(f'pairs: {self.pairs}')
        
    def get_clean(self, index):
        key = 'noisy' if self.pairs == 'noisier2noisy_v2' else 'clean'
        wav_id, begin, end = self.get_chunk(index, key)
        clean = self.get_wav_data(wav_id, key)
        clean = clean[begin:end]
        clean = self.check_wav_data(clean)
        if random.random() < self.add_rir:
            clean = self.add_pyreverb(clean)
        return wav_id, clean

    def get_noise(self, index=None):
        if not index:
            index = random.randint(0, len(self))
        wav_id, begin, end = self.get_chunk(index, 'noise')
        noise = self.get_wav_data(wav_id, 'noise')
        return self.check_wav_data(noise[begin:end])

    def __getitem__(self, index):
        wav_id, clean = self.get_clean(index)
        noise = self.get_noise()
        clean, _, noisy, _ = audio.snr_mixer(self.mix_param, clean, noise, self.snr)
        if self.pairs in ['noisy2clean', 'noisier2noisy_v2']:
            target = clean
        elif self.pairs == 'noisy2noisy':
            noise = self.get_noise()
            _, _, target, target_rms_level = audio.snr_mixer(self.mix_param, clean, noise, self.snr)
            noisy = audio.normalize(noisy, target_rms_level)
        elif self.pairs == 'noisier2noisy':
            noise = self.get_noise()
            _, _, noisier, noisier_rms_level = audio.snr_mixer(self.mix_param, noisy, noise, self.snr)
            target = noisy
            noisy = noisier
            target = audio.normalize(target, noisier_rms_level)
        else:
            raise ValueError(f'pairs: {self.pairs}')
        data = {'clean_wav': target.astype(np.float32), 
                'noisy_wav': noisy.astype(np.float32), 
                'wav_id': wav_id}
        return data
        # return target.astype(np.float32), noisy.astype(np.float32), wav_id
    
    def __len__(self):
        if self.pairs == 'noisier2noisy_v2':
            length = len(self.chunk_info['noisy'])
        else:
            length = len(self.chunk_info['clean'])
        return length
    
    
class FeatLableDataset(DatasetBase):
    '''FeatLableDataset
    read feat and label from kaldi data (hdf5)
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        feats_csv = self.resolve_path('feats_csv', subset)
        feats_store = self.resolve_path('feats_store', subset)
        self.data_info['feats'] = read_csv(feats_csv) if feats_csv else None
        self.data_store['feats'] = FeatStore(feats_store, self.data_info['feats'], 
                                             cache=self.cache) if feats_store or feats_csv else None
        assert self.label_dict is not None
        assert set(self.data_info['feats'].index) == set(self.label_dict.keys())
        assert self.feat_extractor is not None
        logger.info(self.get_feats_msg())
        self.filter_feats()
        
    def get_feats_msg(self):
        n_frames = self.data_info['feats']['n_frames']
        msg = '[{}] utterance: {:d}, max: {:d} frames, min: {:.2f} frames, mean: {:.2f} frames'.format(
            self.subset, len(n_frames), n_frames.max(), n_frames.min(), n_frames.mean())
        return msg
    
    def filter_feats(self, key='feats', name='n_frames'):
        # Filter feats, and update data_info and data_store
        filter_config = OmegaConf.to_container(self.conf['data'][self.subset]).get('filter')
        if not filter_config:
            return
        df = self.data_info[key]
        assert name in df.columns
        if isinstance(filter_config, list):
            lower_bound, upper_bound = filter_config
        elif isinstance(filter_config, float):
            mid_quantile = filter_config
            lower_q = (1 - mid_quantile) / 2
            upper_q = mid_quantile + lower_q
            lower_bound = df[name].quantile(lower_q)
            upper_bound = df[name].quantile(upper_q)
        else:
            raise ValueError(f'Invalid filter configuration: {filter_config}')
        df_filtered = df[(df[name] >= lower_bound) & (df[name] <= upper_bound)]
        self.data_info[key] = df[df.index.isin(df_filtered.index)]
        self.data_store[key].update_info(self.data_info[key])
        logger.info(f'[Filtered] {self.get_feats_msg()}')
    
    def get_feats(self, index):
        wav_id, data = self.data_store['feats'].get_data_by_index(index)
        feats = self.feat_extractor.post_process(data)
        return wav_id, feats
    
    def __getitem__(self, index):
        wav_id, feats = self.get_feats(index)
        label = self.label_dict[wav_id]
        data = {'wav_id': wav_id, 'feats': feats, 'label': label}
        return data
    
    def __len__(self):
        return len(self.data_info['feats'])


class WavTextDataset(DatasetBase):
    '''WavTextDataset
    Extract FBank from wav, and obtain CTC lable from text
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.label_dict is not None
        assert self.feat_extractor is not None
    
    def __getitem__(self, index):        
        wav_id, begin, end = self.get_chunk(index)
        noisy_wav = self.get_wav_data(wav_id, 'noisy')
        noisy_wav = self.check_wav_data(noisy_wav[begin:end])        
        feats = self.feat_extractor.get_feat(torch.from_numpy(noisy_wav))
        label = self.label_dict[wav_id]
        data = {'wav_id': wav_id, 'noisy_wav': noisy_wav, 'feats': feats, 'label': label}
        return data        
    
    def __len__(self):
        return len(self.data_info['noisy'])
