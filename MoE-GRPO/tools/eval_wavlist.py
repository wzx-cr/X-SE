import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import soundfile as sf
from modules.utils import metrics
from modules.utils.compute_score import ComputeMOS
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from modules.utils.common import compact_dict, read_wav_scp
from copy import deepcopy

def eval_wavfile(ref_file, est_file, fs=16000, truncate=True, mode='wb', extended=True):
    ref, fs1 = sf.read(ref_file)
    est, fs2 = sf.read(est_file)
    assert fs1 == fs
    assert fs2 == fs
    if truncate is True:
        est = est[:ref.shape[0]]
        ref = ref[:est.shape[0]]
    score = metrics.eval(ref, est, fs, mode, extended)
    return (ref_file.stem, score['PESQ'], score['STOI'], score['eSTOI'], score['SI_SNR'])

    
def eval_wavlist(pairlist, n_workers=16, mode='wb', extended=True):
    """evalutate wav list
    Args:
        pairlist (list): pairlist of wav, [(ref, estimated), ...]
    Return:
        [(name, score1, score2, ...), ...]
    """
    future_tasks = []
    data_list = []
    pool = ProcessPoolExecutor(n_workers)
    for (ref_file, est_file) in pairlist:
        future_tasks.append(pool.submit(eval_wavfile, ref_file, est_file, 
                                        mode=mode, extended=extended))
    for f in tqdm(future_tasks):
        data_list.append(f.result())
    return data_list


def attach_dnsmos(df, est_path_list, conf):
    worker = ComputeMOS(conf)
    sample_rate = conf.get('sample_rate', conf.get('sr', 16000))
    mos_data = []
    for est_wav in tqdm(est_path_list, desc='DNSMOS'):
        sig, bak, ovrl = worker.inference(str(est_wav), fs=sample_rate)
        mos_data.append((est_wav.stem, sig, bak, ovrl))
    mos_df = pd.DataFrame(mos_data, columns=['id', 'SIG', 'BAK', 'OVRL'])
    mos_df = mos_df.drop_duplicates(subset=['id'], keep='first')
    return df.merge(mos_df, on='id', how='left')


def run_eval_wavlist(conf):
    pairlist = []
    est_path_list = []
    est_dir = Path(conf.est_dir).expanduser()
    ref_dir = Path(conf.ref_dir).expanduser()
    subset = conf.get('subset', None)
    if subset:
        df = pd.read_csv(subset, index_col=0, sep=" ", names=['id', 'wav_path', 'dur'])
        id_list = [str(item) for item in df.index.to_list()]
        for wav_id in id_list:
            est_wav = est_dir.joinpath(f"{wav_id}.wav")
            pairlist.append([ref_dir.joinpath(f"{wav_id}.wav"), est_wav])
            est_path_list.append(est_wav)
    else:
        for est_wav in est_dir.glob("*.wav"):
            pairlist.append([ref_dir.joinpath(est_wav.name), est_wav])
            est_path_list.append(est_wav)
    data_list = eval_wavlist(pairlist, conf.n_workers, 
                             conf.get('mode', 'wb'), conf.get('extended', True))
    
    metric_list = conf.metric_list.split(",")
    columns = deepcopy(metric_list)
    columns.insert(0, "id")
    df = pd.DataFrame(data=data_list, columns=columns).sort_values(by="id")
    if conf.get('dnsmos', False):
        df = attach_dnsmos(df, est_path_list, conf)
        metric_list = metric_list + ['SIG', 'BAK', 'OVRL']
    out_csv = conf.out_csv
    if out_csv is None:
        name = "{}_{}".format(est_dir.parent.parent.name, est_dir.parent.name)
        out_csv = Path(conf.out_dir).joinpath(f"{name}.csv")
    else:
        out_csv = Path(out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print("to csv:", out_csv)
    df.to_csv(out_csv, index=None)
    
    if conf.key_list is not None:
        for key in conf.key_list.split(","):
            print(key, compact_dict(df[df.id.str.contains(key)][metric_list].mean()))
    print("Average", compact_dict(df[metric_list].mean()))


def run_eval_wav_scp(conf):
    root_path = Path(conf['root_dir']).expanduser()
    ref_wav_dict = read_wav_scp(root_path.joinpath(conf['ref_scp']), root_path)    
    est_wav_dict = read_wav_scp(root_path.joinpath(conf['est_scp']), root_path)
        
    pairlist = []
    est_path_list = []
    for key in ref_wav_dict:
        est_wav_path = Path(est_wav_dict[key])
        if not est_wav_path.exists():
            est_wav_path = root_path.joinpath(est_wav_dict[key])
        pairlist.append([Path(ref_wav_dict[key]), est_wav_path])
        est_path_list.append(est_wav_path)
    data_list = eval_wavlist(pairlist, conf.n_workers)
    
    metric_list = conf.metric_list.split(",")
    columns = deepcopy(metric_list)
    columns.insert(0, "id")
    df = pd.DataFrame(data=data_list, columns=columns).sort_values(by="id")
    if conf.get('dnsmos', False):
        df = attach_dnsmos(df, est_path_list, conf)
        metric_list = metric_list + ['SIG', 'BAK', 'OVRL']
    out_csv = conf.out_csv
    if out_csv is None:
        name = Path(conf['est_scp']).parent.name
        out_csv = Path(conf.out_dir).joinpath(f"{name}.csv")
    else:
        out_csv = Path(out_csv).expanduser()
    print("to csv:", out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=None)
    
    if conf.key_list is not None:
        for key in conf.key_list.split(","):
            print(key, compact_dict(df[df.id.str.contains(key)][metric_list].mean()))
    print("Average", compact_dict(df[metric_list].mean()))
    

def run_eval_wavfile(conf):
    print(eval_wavfile(Path(conf.ref_file), Path(conf.est_file)))


if __name__ == '__main__':
    conf = OmegaConf.create({
         "cmd": "run_eval_wavlist",
         "n_workers": 16,
         "ref_dir": "outputs/reference",
         "est_dir": "outputs/enhanced",
         "out_dir": "exp/results",
         "out_csv": None,
         "metric_list": "PESQ,STOI,eSTOI,SI_SNR",
         "dnsmos": False,
         "sample_rate": 16000,
         "DNSMOS": {
             "model_path": "tools/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
             "providers": "auto",
             "mos_raw": False,
             "personalized": False
         },
         "key_list": None,
         "root_dir": ".",
         "ref_scp": "data/reference.scp",
         "est_scp": "outputs/enhanced.scp",
         })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)
    
