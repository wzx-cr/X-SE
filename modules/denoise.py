from tqdm import tqdm
import librosa
from pathlib import Path
import csv
from omegaconf import OmegaConf
from .utils import audio
from .utils.init_conf import init_conf
from .utils.common import get_pool_executor
from alpha.enh import system


def _save_wav(wav_path, wav, volume=None, volume_type='rms'):
    if volume:
        if volume_type == 'rms':
            audio.audiowrite(wav_path, wav, norm=True, target_level=volume)
        else:
            wav = audio.normalize_with_peak(wav, peak_level=volume)
            audio.audiowrite(wav_path, wav)
    else:
        audio.audiowrite(wav_path, wav)


def _resolve_wav_path(path_str, base_dir):
    wav_path = Path(str(path_str).strip()).expanduser()
    if not wav_path.is_absolute():
        wav_path = base_dir.joinpath(wav_path)
    return wav_path


def _load_wavlist(wavlist_file):
    wavlist_file = Path(wavlist_file).expanduser()
    if not wavlist_file.exists():
        raise FileNotFoundError(f'wavlist not found: {wavlist_file}')

    # Support both plain wav list (one path per line) and csv metadata files.
    if wavlist_file.suffix.lower() == '.csv':
        with open(wavlist_file, newline='') as f:
            sample = f.read(2048)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',\t ')
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if reader.fieldnames is None:
                raise ValueError(f'Invalid wavlist csv (no header): {wavlist_file}')
            field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
            for key in ['wav_path', 'wav', 'path']:
                if key in field_map:
                    wav_col = field_map[key]
                    break
            else:
                # fallback: usually 2nd column is wav path when first is wav_id
                wav_col = reader.fieldnames[1] if len(reader.fieldnames) > 1 else reader.fieldnames[0]
            wav_list = []
            for row in reader:
                wav_path = row.get(wav_col)
                if wav_path is None:
                    continue
                wav_path = wav_path.strip()
                if len(wav_path) == 0:
                    continue
                wav_list.append(_resolve_wav_path(wav_path, wavlist_file.parent))
        if len(wav_list) == 0:
            raise ValueError(f'No wav path found in wavlist csv: {wavlist_file}')
    else:
        with open(wavlist_file) as f:
            content = f.readlines()
        wav_list = []
        for wav in content:
            wav = wav.strip()
            if len(wav) == 0 or wav.startswith('#'):
                continue
            wav_list.append(_resolve_wav_path(wav, wavlist_file.parent))

    if len(wav_list) == 0:
        raise ValueError(f'Empty wav list: {wavlist_file}')
    return wav_list


def denoise(conf):
    device = conf.get('device', 'cuda')
    assert 'denoise' in conf
    w_oa = conf.denoise.get('w_oa', 0.0)  # observation adding
    if w_oa > 0:
        print(f'w_oa: {w_oa}')
    volume_type = conf.get('volume_type', 'rms')  # rms, peak
    volume = conf.get('volume', None)  # volume in dB, e.g., -25, -2.02
    if volume:
        print(f'volume_type:{volume_type}, volume: {volume}')
    pool = get_pool_executor(**conf['pool'])
    future_tasks = []
    out_dir = Path(conf.denoise.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    if conf.denoise.get('wavlist', None):
        wav_list = _load_wavlist(conf.denoise.wavlist)
    else:
        wav_list = list(
            Path(conf.denoise.wav_dir).expanduser().glob('*/*.wav'))
    print('wav_list: ', wav_list[:3], '...')
    conf = init_conf(conf)
    print(conf.ckpt)
    system_class = getattr(system, conf['system']['name'])
    if conf.ckpt is None:
        if conf['model'].get('init') is None:
            return
        model = system_class(conf=conf).to(device)
        model.init_model(OmegaConf.to_container(conf['model'], resolve=True))
    else:
        model = system_class.load_from_checkpoint(conf.ckpt, conf=conf).to(device)
    model.eval()  # evaluation mode
    for _wav in tqdm(wav_list):
        y, _ = librosa.load(_wav, sr=conf.get('sr', None))
        est_wav = model.denoise(y, conf.denoise.get('chunk', -1)).cpu().numpy()
        if w_oa > 0:
            est_wav = (1 - w_oa) * est_wav + w_oa * y
        future_tasks.append(pool.submit(_save_wav, out_dir.joinpath(
            _wav.name), est_wav, volume, volume_type))
    for _ in tqdm(range(len(future_tasks))):
        f = future_tasks.pop(0)
        _ = f.result()
