# pytorch iter-style dataset
from torch.utils.data import IterableDataset
from .data_store import kaldi_read_label, KaldiFeatStore

class FeatDatasetIter(IterableDataset):
    def __init__(self, conf, subset):
        '''        
        subset: ['train', 'valid']
        '''
        super().__init__()
        self.conf = conf
        self.subset = subset
        self.label_data = kaldi_read_label(conf['data'][subset]['label_scp'], 
                                           conf['data']['n_split'])
        
class WavDatasetIter():
    def __init__(self) -> None:
        pass