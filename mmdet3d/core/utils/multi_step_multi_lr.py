import mmcv
from functools import reduce
from mmcv.runner.hooks import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class MultiStepMultiLrUpdaterHook(LrUpdaterHook):
    """Multi-Step LR Schduler with different gamma at each step.

    Step LR scheduler with min_lr clipping.
    Args:
        step (list[int])
        gamma (list[float])
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self, step, gamma, min_lr=None, **kwargs):
        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all([s > 0 for s in step])
        else:
            raise TypeError('"step" must be a list')

        if isinstance(gamma, list):
            assert mmcv.is_list_of(gamma, float)
            assert all([g > 0 for g in gamma])
        else:
            raise TypeError('"gamma" must be a list')

        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super(MultiStepMultiLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate index into self.gamma
        # values will range from 0 ~ len(self.step), inclusive.
        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break

        if exp == 0:
            lr = base_lr
        else:
            lr = base_lr * reduce(lambda a, b: a * b, self.gamma[:exp], 1)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr