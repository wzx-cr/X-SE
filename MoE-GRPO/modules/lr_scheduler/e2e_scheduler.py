import math
import logging

from .lr_scheduler import decision, add_scheduler
from .kaldi_scheduler import KaldiLRScheduler

logger = logging.getLogger(__name__)

# todo: store lr scheduler params, currently this scheduler DO NOT support resuming
@add_scheduler('e2e')
class E2ELRScheduler(KaldiLRScheduler):
    """Normal E2E-training learning rate scheduler

    It decays the weight once there is no relative improvement
    The training is terminated if model not improves for #patience epochs
    Note that this scheduler only return "REJECT" status if relative improvement less than 'reject_threshold'

    The learning rate can be gradually warmed up linearly by specify non-zero ``warmup_round``
    and ``warmup_batches_per_round``.

    Parameters:
        - warmup_round (int): #rounds the scheduler spends to increase the learning rate
        - warmup_batches_per_round (int): #batches a round contains
        - decay_threshold (float): relative improvement threshold to decay lr
        - reject_threshold (float): relative improvement threshold to return 'REJECT status'
        - patience (int): patience epoch for early stopping
    """

    def __init__(self, optimizer, warmup_round=200, warmup_batches_per_round=20,
                decay_threshold=0.0, reject_threshold=-0.05, patience=3):
        self._last_metric = math.inf
        self.warmup_round = warmup_round
        self.warmup_batches_per_round = warmup_batches_per_round
        self.decay_threshold = decay_threshold
        self.reject_threshold = reject_threshold
        self.patience = patience

        self._no_imprv = 0 # no improve epochs
        self.decay_factor = 1
        # rel_stop & rel_decay is not used
        super().__init__(optimizer, warmup_round, warmup_batches_per_round)

    def _step_epoch(self, metric):
        metric = float(metric)
        if math.isnan(metric):
            metric = math.inf
        rel_improve = (self._last_metric - metric) / self._last_metric
        self.last_epoch += 1

        decisions = []

        if rel_improve <= self.decay_threshold:
            self._no_imprv += 1
            # check if model not improves for 'patience' epoch
            if self._no_imprv >= self.patience: # early stop
                self.lr_decay(factor=0)
                decisions.append(decision.STOP)
                decisions.append(decision.REJECT)
                return decisions, 'Finished, no improvement for {} epochs.'.format(self.patience)

            if rel_improve <= self.reject_threshold: # if performance decay too much, reject
                decisions.append(decision.REJECT)
                self._last_metric = self._last_metric
                self.lr_decay()
                return decisions, 'Too much performance degradation {}, lr decays & reject'.format(rel_improve)
            # continue decay, still return ACCEPT decision
            decisions.append(decision.ACCEPT)
            self._last_metric = self._last_metric
            self.lr_decay()
            return decisions, 'No improvement {}, lr decays'.format(rel_improve)
        else:
            self._no_imprv = 0 # clean up
            decisions.append(decision.ACCEPT)
            self._last_metric = metric # update best metric
            return decisions, f'Model improves, continue training'

