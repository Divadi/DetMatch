# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS
from mmcv.runner.hooks import OptimizerHook
from mmcv.utils import _BatchNorm
from torch.nn.utils import clip_grad


@HOOKS.register_module()
class MyOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()


# @HOOKS.register_module()
# class GradientCumulativeOptimizerHook(OptimizerHook):
#     """Optimizer Hook implements multi-iters gradient cumulating.

#     Args:
#         cumulative_iters (int, optional): Num of gradient cumulative iters.
#             The optimizer will step every `cumulative_iters` iters.
#             Defaults to 1.

#     Examples:
#         >>> # Use cumulative_iters to simulate a large batch size
#         >>> # It is helpful when the hardware cannot handle a large batch size.
#         >>> loader = DataLoader(data, batch_size=64)
#         >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
#         >>> # almost equals to
#         >>> loader = DataLoader(data, batch_size=256)
#         >>> optim_hook = OptimizerHook()
#     """

#     def __init__(self, cumulative_iters=1, **kwargs):
#         super(GradientCumulativeOptimizerHook, self).__init__(**kwargs)

#         assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
#             f'cumulative_iters only accepts positive int, but got ' \
#             f'{type(cumulative_iters)} instead.'

#         self.cumulative_iters = cumulative_iters
#         self.divisible_iters = 0
#         self.remainder_iters = 0
#         self.initialized = False

#     def has_batch_norm(self, module):
#         if isinstance(module, _BatchNorm):
#             return True
#         for m in module.children():
#             if self.has_batch_norm(m):
#                 return True
#         return False

#     def _init(self, runner):
#         if runner.iter % self.cumulative_iters != 0:
#             runner.logger.warning(
#                 'Resume iter number is not divisible by cumulative_iters in '
#                 'GradientCumulativeOptimizerHook, which means the gradient of '
#                 'some iters is lost and the result may be influenced slightly.'
#             )

#         if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
#             runner.logger.warning(
#                 'GradientCumulativeOptimizerHook may slightly decrease '
#                 'performance if the model has BatchNorm layers.')

#         residual_iters = runner.max_iters - runner.iter

#         self.divisible_iters = (
#             residual_iters // self.cumulative_iters * self.cumulative_iters)
#         self.remainder_iters = residual_iters - self.divisible_iters

#         self.initialized = True

#     def after_train_iter(self, runner):
#         if not self.initialized:
#             self._init(runner)

#         if runner.iter < self.divisible_iters:
#             loss_factor = self.cumulative_iters
#         else:
#             loss_factor = self.remainder_iters
#         loss = runner.outputs['loss']
#         loss = loss / loss_factor
#         loss.backward()

#         if (self.every_n_iters(runner, self.cumulative_iters)
#                 or self.is_last_iter(runner)):

#             if self.grad_clip is not None:
#                 grad_norm = self.clip_grads(runner.model.parameters())
#                 if grad_norm is not None:
#                     # Add grad norm to the logger
#                     runner.log_buffer.update({'grad_norm': float(grad_norm)},
#                                              runner.outputs['num_samples'])
#             runner.optimizer.step()
#             runner.optimizer.zero_grad()