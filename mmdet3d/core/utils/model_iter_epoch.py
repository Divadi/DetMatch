"""We define a simple hook that gives the runner's "iter" and "epoch"
attributes to the model.

For some reason, runner.epoch is 0 over the first two epochs, and 1 for the
third, 2 for the fourth, etc.
However, when resuming the checkpoint after first epoch, runner.epoch is
1. For checkpoint after second epoch, is 2.
    This would be an issue, but when resuming the checkpoint after the second
    epoch, is 2 but is also 2 for the epoch after that. then, it resumes to 3.

My solution is the "-1" below, in before_run. So when starting training,
from scratch would go -1, 0, 1, ...
from checkpoint "i" would go i-1, i, i+1, ...
"""
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ModelIterEpochHook(Hook):

    def before_run(self, runner):
        runner.model.module.iter = runner.iter
        runner.model.module.epoch = runner.epoch - 1

    def after_train_iter(self, runner):
        runner.model.module.iter = runner.iter

    def after_train_epoch(self, runner):
        runner.model.module.epoch = runner.epoch