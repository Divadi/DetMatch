from mmcv.runner import HOOKS, Hook, WandbLoggerHook
from mmcv.runner.dist_utils import master_only


def add_prefix_no_dot(input_dict, prefix):
    return {f'{prefix}{k}': v for k, v in input_dict.items()}


@HOOKS.register_module()
class WandbVisHook(Hook):
    def __init__(self):
        self.wandb = None

    @master_only
    def after_train_iter(self, runner):
        if self.wandb is None:
            # pull wandb from "WandbLoggerHook"
            # Not sure if necessary, because self.wandb is directly the
            # imported module, but the init code is indeed in the loggerhook.
            for hook in runner.hooks:
                if isinstance(hook, WandbLoggerHook):
                    self.wandb = hook.wandb
                    break
            if self.wandb is None:
                raise Exception('WandbLoggerHook not found in hook list.')

        vis_dict = add_prefix_no_dot(runner.outputs.get('vis', dict()), 'vis/')
        # vis_dict = {
        #     k.rsplit("/", 1)[0]:self.wandb.Image(
        #         v, caption=k.rsplit("/", 1)[1])
        #     for k, v in vis_dict.items()}
        vis_dict = {
            k: self.wandb.Image(
                v, caption=k.rsplit('/', 1)[1])
            for k, v in vis_dict.items()}
        self.wandb.log(vis_dict, step=(runner.iter + 1))