import mmcv
import time
import torch
import warnings
from mmcv.runner.builder import RUNNERS
from mmcv.runner.iter_based_runner import IterBasedRunner, IterLoader
from mmcv.runner.utils import get_host_info


@RUNNERS.register_module()
class IterBasedSSLRunner(IterBasedRunner):

    def train(self, lab_data_loader, unlab_data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = lab_data_loader
        # TODO: I don't know if the above causes hidden problems
        self._epoch = lab_data_loader.epoch

        lab_data_batch = next(lab_data_loader)
        unlab_data_batch = next(unlab_data_loader)
        data_batch = dict()
        data_batch.update(
            {f'lab_{k}': v for k, v in lab_data_batch.items()})
        data_batch.update(
            {f'unlab_{k}': v for k, v in unlab_data_batch.items()})
        data_batch['img_metas'] = lab_data_batch['img_metas']

        # print(data_batch)

        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        torch.cuda.empty_cache()  # only thing different here

        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        self._inner_iter += 1

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)

        assert len(workflow) == 1
        assert len(data_loaders) == 2  # Labeled & Unlabeled

        lab_data_loader, unlab_data_loader = data_loaders
        lab_iter_loader = IterLoader(lab_data_loader)
        unlab_iter_loader = IterLoader(unlab_data_loader)

        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)

                for _ in range(iters):
                    assert mode == 'train'
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(lab_iter_loader, unlab_iter_loader, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')