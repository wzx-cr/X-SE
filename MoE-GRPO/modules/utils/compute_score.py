import torch.nn.functional as F
import torch
import functools
import os
import modules.dataset as dataset
from torch.utils.data import DataLoader
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pesq import pesq
from tqdm import tqdm
from modules.utils.logging import logger
from .common import compact_dict, get_pool_executor, get_rank, resolve_path


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class DNSMOSBase:
    provider_dict = {'CUDA': 'CUDAExecutionProvider', 
                     'ROCM': 'ROCMExecutionProvider', 
                     'CPU': 'CPUExecutionProvider'}
    def __init__(self, conf) -> None:
        model_path = resolve_path(conf['DNSMOS']['model_path'])
        providers = self.get_providers(conf['DNSMOS']['providers'], get_rank())
        logger.info(f"DNSMOS model_path: {model_path}, providers: {providers}")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 8
        self.onnx_sess = self._create_session(model_path, providers, sess_options)
        # keep the effective providers for downstream logic (e.g., io_binding toggling)
        self.providers = self.onnx_sess.get_providers()
        self.mos_raw = conf['DNSMOS'].get('mos_raw', False)
        self.personalized = conf['DNSMOS'].get('personalized', False)

    def _create_session(self, model_path, providers, sess_options):
        """
        Try to build ONNXRuntime session with the requested providers.
        If CUDA/ROCM init fails (common on shared clusters with mismatched drivers),
        fall back to CPU to avoid hard crashes during training.
        """
        try:
            return ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        except Exception as e:
            logger.warning(
                f"DNSMOS InferenceSession init failed with providers={providers}; "
                f"falling back to CPUExecutionProvider. Error: {e}"
            )
            return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)
        
    def get_providers(self, provider, rank=0):
        if not isinstance(provider, str):
            return provider
        if provider == 'auto':
            if os.environ.get("CUDA_UNUSABLE") == "1":
                provider = 'CPU'
            else:
                if torch.backends.cuda.is_built():
                    if torch.version.hip:
                        provider = "ROCM"
                    else:
                        provider = "CUDA"
                else:
                    provider = 'CPU'
        _provider = self.provider_dict.get(provider)
        if _provider is None:
            raise ValueError(f"Invalid provider: {provider}")
        # 优先使用 LOCAL_RANK 映射到实际 GPU，fallback 到传入的 rank
        env_local = os.environ.get("LOCAL_RANK")
        if env_local is not None:
            device_id = max(0, int(env_local))
        else:
            device_id = max(0, int(rank))
        return [(_provider, {'device_id': device_id})]
    
    
# compute DNSMOS score
class ComputeMOS(DNSMOSBase):
    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.pool = conf.get('pool', None)
        if self.pool:
            self.pool_executor = get_pool_executor(**conf['pool'])
        
    def get_polyfit_val(self, sig, bak, ovr):
        if self.personalized:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479,  -0.1644611,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def inference(self, audio, fs=16000):
        if type(audio) == str:
            audio, _ = sf.read(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            oi = {'input_1': input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
        if self.mos_raw:
            SIG, BAK, OVRL = np.mean(predicted_mos_sig_seg_raw), np.mean(predicted_mos_bak_seg_raw), np.mean(predicted_mos_ovr_seg_raw)
        else:
            SIG, BAK, OVRL = np.mean(predicted_mos_sig_seg), np.mean(predicted_mos_bak_seg), np.mean(predicted_mos_ovr_seg)
        return SIG, BAK, OVRL
    
    def batch_scores(self, audio_batch):
        # n_batch, n_samples = audio_batch.shape
        if isinstance(audio_batch, torch.Tensor):
            audio_batch = audio_batch.cpu()
        scores = []
        if self.pool:
            future_tasks = []
            for audio in audio_batch:
                future_tasks.append(self.pool_executor.submit(self.inference, audio))
            for _ in range(len(future_tasks)):
                f = future_tasks.pop(0)
                scores.append(f.result())
        else:
            for audio in audio_batch:
                scores.append(self.inference(audio))
        return np.array(scores, dtype=np.float32)


def evaluate_polynomials(inputs, coeffs):
    degree = coeffs.size(1) - 1
    powers = torch.arange(degree, -1, -1).type_as(inputs)
    x_powers = inputs.unsqueeze(-1) ** powers
    results = torch.sum(x_powers*coeffs.unsqueeze(0), dim=-1)
    return results


class TorchMOS(DNSMOSBase):
    def __init__(self, conf) -> None:
        super().__init__(conf)
        # io_binding only makes sense when CUDA provider is actually available
        use_io_binding = conf['DNSMOS'].get('io_binding', False) and \
            ('CUDAExecutionProvider' in self.providers)
        if use_io_binding:
            self.io_binding = self.onnx_sess.io_binding()
        else:
            self.io_binding = False
        if self.personalized:
            self.coefficients = torch.tensor(
                [[-0.01019296, 0.02751166, 1.19576786, -0.24348726],
                 [-0.04976499, 0.44276479, -0.1644611, 0.96883132],
                 [-0.00533021, 0.005101, 1.18058466, -0.11236046]], dtype=torch.float32)
        else:
            self.coefficients = torch.tensor(
                [[-0.08397278, 1.22083953, 0.0052439],
                 [-0.13166888, 1.60915514, -0.39604546],
                 [-0.06766283, 1.11546468, 0.04602535]], dtype=torch.float32)
        self.fs = conf['data']['sample_rate']
        self.hop_samples = self.fs
        self.n_samples = int(INPUT_LENGTH * self.fs)        
        chunk_samples = (conf['data']['chunk_size'] - 1) * conf['stft']['hop_length'] + \
            conf['stft']['win_length']
        self.padding = functools.partial(F.pad, pad=(0, self.n_samples-chunk_samples), mode='replicate')
        self.score_full_audio = conf['DNSMOS'].get('score_full_audio', False)
    
    def _fit_input_length(self, audio):
        if audio.shape[-1] > self.n_samples:
            return audio[..., :self.n_samples]
        if audio.shape[-1] < self.n_samples:
            pad = self.n_samples - audio.shape[-1]
            if audio.shape[-1] == 0:
                return F.pad(audio, (0, pad))
            return torch.cat([audio, audio[..., -1:].expand(*audio.shape[:-1], pad)], dim=-1)
        return audio

    def _score_fixed_length(self, input):
        input = input.contiguous()
        if self.io_binding and input.is_cuda:
            self.io_binding.bind_input(name='input_1', device_type='cuda', device_id=input.device.index,
                                    element_type=np.float32, shape=tuple(input.shape), buffer_ptr=input.data_ptr())
            # Prepare output buffers
            output_shape = (input.shape[0], 3)
            scores = torch.empty(output_shape, dtype=torch.float32, device='cuda').contiguous()
            self.io_binding.bind_output(name='Identity:0', device_type='cuda', device_id=input.device.index,
                                        element_type=np.float32, shape=output_shape, buffer_ptr=scores.data_ptr())
            # Run the session
            self.onnx_sess.run_with_iobinding(self.io_binding)
        else:
            oi = {'input_1': input.cpu().numpy()}
            scores = self.onnx_sess.run(None, oi)[0]
            scores = torch.tensor(scores).type_as(input)
        if not self.mos_raw:
            scores = evaluate_polynomials(scores, self.coefficients.type_as(scores))
        return scores

    def _batch_scores_full_audio(self, audio):
        score_list = []
        for wav in audio:
            wav = wav.unsqueeze(0)
            n_sample = wav.shape[-1]
            last_start = max(0, n_sample - self.n_samples)
            starts = list(range(0, last_start + 1, self.hop_samples))
            if not starts:
                starts = [0]
            if starts[-1] != last_start:
                starts.append(last_start)
            segments = [self._fit_input_length(wav[..., start:start + self.n_samples]) for start in starts]
            segment_scores = self._score_fixed_length(torch.cat(segments, dim=0))
            score_list.append(segment_scores.mean(dim=0))
        return torch.stack(score_list, dim=0)

    def batch_scores(self, audio, score_full_audio=None):
        # audio: tensor, shape of [n_batch, n_samples]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        score_full = self.score_full_audio if score_full_audio is None else bool(score_full_audio)
        if score_full and audio.shape[-1] > self.n_samples:
            return self._batch_scores_full_audio(audio)
        # Some ONNX DNSMOS models have a static input length (e.g., 144160).
        # Force every sample to exact `self.n_samples` to avoid shape mismatch.
        return self._score_fixed_length(self._fit_input_length(audio))
        

# compute PESQ score                                                                         
def calc_pesq(clean, noisy, sr=16000):    
    clean = clean.cpu().numpy()
    noisy = noisy.cpu().numpy()
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


class ComputePESQ:
    def __init__(self, conf) -> None:
        self.pool = conf.get('pool', None)
        if self.pool:
            self.pool_executor = get_pool_executor(**conf['pool'])
    
    def batch_scores(self, clean_batch, noisy_batch):
        # n_batch, n_samples = clean_batch.shape
        score_list = []
        if self.pool:
            future_tasks = []
            for clean, noisy in zip(clean_batch,  noisy_batch):
                future_tasks.append(self.pool_executor.submit(calc_pesq, clean, noisy))
            for _ in range(len(future_tasks)):
                f = future_tasks.pop(0)
                score_list.append(f.result())
        else:
            for clean, noisy in zip(clean_batch,  noisy_batch):
                score_list.append(calc_pesq(clean, noisy))        
        return np.array(score_list)
    

def test_wav_dir(conf):
    worker = ComputeMOS(conf)
    wav_list = list(Path(conf.wav_dir).expanduser().glob("*.wav"))
    print("wav:", len(wav_list))
    if conf.get('parallel', False):
        score = Parallel(n_jobs=-1)(delayed(worker)(str(wav_path)) for wav_path in wav_list)
        score = np.array(score)
    elif conf.get('pool', None):
        pool = get_pool_executor(**conf['pool'])
        future_tasks = []
        data_list = []
        for _item in tqdm(wav_list):
            future_tasks.append(pool.submit(worker, str(_item)))
        for _ in tqdm(range(len(future_tasks))):
            f = future_tasks.pop(0)
            data_list.append(f.result())
        df = pd.DataFrame(data_list, columns=['SIG', 'BAK', 'OVRL'])
        df_mean = df[['SIG','BAK', 'OVRL']].mean()
        result = compact_dict({"SIG": df_mean["SIG"], "BAK": df_mean["BAK"], "OVRL": df_mean["OVRL"]})
        print(result)
    else:
        data_list = []
        for wav_path in tqdm(wav_list):
            data = worker(str(wav_path))
            data_list.append(data)
        df = pd.DataFrame(data_list, columns=['SIG', 'BAK', 'OVRL'])
        df_mean = df[['SIG','BAK', 'OVRL']].mean()
        result = compact_dict({"SIG": df_mean["SIG"], "BAK": df_mean["BAK"], "OVRL": df_mean["OVRL"]})
        print(result)


def test_Dataset(conf):
    subset = conf.get('subset', 'train')
    kwargs = deepcopy(conf['data']['dataloader'])
    data_key = conf.get('data_key', 'noisy_wav')
    kwargs = OmegaConf.to_container(kwargs, resolve=True) # dict
    dataset_name = conf.pDataset or conf['data'][subset]['name']
    p_dataset = getattr(dataset, dataset_name)(conf, subset)
    if 'collate_fn' in kwargs and kwargs['collate_fn'] is not None:
        kwargs['collate_fn'] =  getattr(dataset, kwargs['collate_fn'])()
    dataloader = DataLoader(p_dataset, **kwargs)
    # batch = next(iter(dataloader))
    score_list = []
    if conf['metric'] == "MOS":
        worker = ComputeMOS(conf)
        for batch in tqdm(dataloader):
            score = worker.batch_scores(batch[data_key].data)
            score_list.append(score)
        print(np.concatenate(score_list, axis=0).mean(axis=0))
        # 3090 耗时3:21, [2.7186  2.4273 2.0514], raw [2.8401 2.2015 2.0948]
    elif conf['metric'] == "TorchMOS":
        worker = TorchMOS(conf)
        for batch in tqdm(dataloader):
            score = worker.batch_scores(batch[data_key].data.cuda())
            score_list.append(score)
        print(torch.cat(score_list, dim=0).mean(dim=0)) 
        # 3090 耗时28s, [2.6576, 2.5862, 2.1242], raw [2.7818, 2.3970, 2.1967]
    elif conf['metric'] == "PESQ":
        worker = ComputePESQ(conf)
        for clean, noisy, _ in tqdm(dataloader):
            score = worker.batch_scores(clean, noisy)
            # score = batch_pesq(clean, noisy)
            print(score)        
    else:
        print(conf['metric'], "error")


if __name__ == "__main__":
    conf = OmegaConf.create({
         "yaml": "conf/UnSE.yaml",
         "cmd": "test_Dataset",
         "subset": "train",
         'pDataset': None,
         "metric": "TorchMOS" # {"PESQ", "MOS"}
         })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
    
