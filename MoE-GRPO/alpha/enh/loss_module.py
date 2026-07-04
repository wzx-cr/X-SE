"""Custom enhancement losses.

The public MoE-GRPO example uses `torch.nn.L1Loss`, resolved by
`alpha.enh.system.system.BaseSE.get_loss`. Add project-specific loss classes
here only when a config references them by name.
"""
