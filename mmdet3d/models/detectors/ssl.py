"""Basic SSL parent class that implements useful SSL methods.

Expected framework is a single model_cfg from which both student & teacher
models are constructed, with teacher model detached. ssl_cfg defines lists of
labeled & unlabeled ssl modules. train_cfg contains ssl parameters as well as
normal model train cfgs.
"""
import copy
import numpy as np
import torch
from collections import OrderedDict

from mmdet.models import DETECTORS
from mmseg.core import add_prefix
from ..builder import build_detector, build_ssl_module
from .base import Base3DDetector


@DETECTORS.register_module()
class SSL(Base3DDetector):

    def __init__(self,
                 model_cfg,
                 ssl_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(SSL, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.train_cfg is not None:
            # train_cfg is None for testing.
            self.ema_params = self.train_cfg['ssl']['ema_params']
            self.ema_decay = self.ema_params['ema_decay']
            self.true_avg_rampup = self.ema_params.get('true_avg_rampup',
                                                       False)
            self.rampup_start_decay = self.ema_params.get(
                'rampup_start_decay', 0.5)
            self.use_student_bn_stats_for_teacher = self.ema_params.get(
                'use_student_bn_stats_for_teacher', False)

            self.ssl_weight_params = self.train_cfg['ssl']['weight_params']
            self.ssl_weight = self.ssl_weight_params['weight']
            self.ssl_weight_rampup_start_iter = \
                self.ssl_weight_params.get('weight_rampup_start_iter', 0)
            self.ssl_weight_rampup_num_iter = \
                self.ssl_weight_params.get('weight_rampup_num_iter', 0)

            self.set_teacher_eval = \
                self.train_cfg['ssl'].get('set_teacher_eval', False)

            # To be filled in by ModelIterEpochHook
            self.epoch, self.iter = None, None

        ### Create Teacher & Student models
        teacher_model_cfg = copy.deepcopy(model_cfg)
        teacher_model_cfg['train_cfg'] = \
            train_cfg['teacher'] if train_cfg is not None else None
        teacher_model_cfg['test_cfg'] = test_cfg['teacher']
        teacher_model_cfg['pretrained'] = pretrained  # Added for mmdetector
        self.teacher = build_detector(teacher_model_cfg)
        for t_param in self.teacher.parameters():
            t_param.requires_grad = False

        student_model_cfg = copy.deepcopy(model_cfg)
        student_model_cfg['train_cfg'] = \
            train_cfg['student'] if train_cfg is not None else None
        student_model_cfg['test_cfg'] = test_cfg['student']
        student_model_cfg['pretrained'] = pretrained  # Added for mmdetector
        self.student = build_detector(student_model_cfg)

        ### Create SSL Modules
        ## Want flexibility to use multiple ssl modules and to use different
        ## ssl modules for labeled & unlabeled samples.
        ## ssl_cfg must have dict keys labeled and unlabeled, the values of
        ## which can be lists or just another dict.
        self.lab_ssl_modules = []
        if 'labeled' in ssl_cfg:
            if not isinstance(ssl_cfg['labeled'], list):
                assert isinstance(ssl_cfg['labeled'], dict)
                ssl_cfg['labeled'] = [ssl_cfg['labeled']]

            self.lab_ssl_modules = [
                build_ssl_module(ssl_single_cfg)
                for ssl_single_cfg in ssl_cfg['labeled']
            ]

        self.unlab_ssl_modules = []
        if 'unlabeled' in ssl_cfg:
            if not isinstance(ssl_cfg['unlabeled'], list):
                assert isinstance(ssl_cfg['unlabeled'], dict)
                ssl_cfg['unlabeled'] = [ssl_cfg['unlabeled']]

            self.unlab_ssl_modules = [
                build_ssl_module(ssl_single_cfg)
                for ssl_single_cfg in ssl_cfg['unlabeled']
            ]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """The key point is to support both initial pre-trained student loading
        into teacher & student, as well as continuation of an SSL checkpoint
        that already has separate weights for teacher & student."""
        loading_pretrained_weights = True
        for k in state_dict.keys():
            if 'teacher' in k:
                loading_pretrained_weights = False

        # this function should only worry about parameters initialized in this
        # class. Otherwise, just adjust names for state_dict passed in.
        if loading_pretrained_weights:
            self.teacher.load_state_dict(state_dict)
            self.student.load_state_dict(copy.deepcopy(state_dict))
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                state_dict['teacher.' + k] = v
                state_dict['student.' + k] = copy.deepcopy(v)

                assert 'teacher.' + k in self.state_dict() and \
                    'student.' + k in self.state_dict()
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def _get_curr_ema_decay(self):
        """(Current) Options for ema updating:

        Constant momentum
            True avg until constant
        TODO:
            Update every so many iters parameter
            True avg from initial momentum until constant momentum
            Higher constant momentum, until some iter, go to constant momentum
        """
        if self.ema_params.get('true_avg_rampup', False):
            ema_start_iter = max(round(1 / (1 - self.rampup_start_decay)), 2)
            return min(1 - 1 / (self.iter + ema_start_iter),
                       self.ema_params['ema_decay'])
        else:
            return self.ema_params['ema_decay']

    def _update_teacher(self):
        curr_ema_decay = self._get_curr_ema_decay()

        student_state_dict = self.student.state_dict()
        new_teacher_state_dict = OrderedDict()
        for k, v in self.teacher.state_dict().items():
            if k in student_state_dict:
                new_teacher_state_dict[k] = (
                    student_state_dict[k] * (1 - curr_ema_decay) +
                    v * curr_ema_decay)
            else:
                raise Exception('{} not found in student model'.format(k))

            if self.use_student_bn_stats_for_teacher:
                if 'running' in k:
                    new_teacher_state_dict[k] = student_state_dict[k]

        self.teacher.load_state_dict(new_teacher_state_dict)

    def _get_curr_ssl_weight(self):
        """This is for ramping up unlabeled weight during training. Start
        ramping from ssl_weight_rampup_start_iter, for
        ssl_weight_rampup_num_iter many iters.

        Before ssl_weight_rampup_start_iter, if ssl_weight_rampup_num_iter is
        not 0, ssl_weight will be 0.
        """
        if self.ssl_weight_rampup_num_iter == 0:
            return self.ssl_weight
        elif self.iter < self.ssl_weight_rampup_start_iter:
            return 0.0
        else:
            current = self.iter - self.ssl_weight_rampup_start_iter
            current = np.clip(current, 0, self.ssl_weight_rampup_num_iter)
            phase = 1.0 - current / self.ssl_weight_rampup_num_iter
            return self.ssl_weight * np.exp(-5.0 * phase * phase)

    def _collapse_losses(self, losses):
        """Adapted from _parse_losses from base.py in mmdet detectors.

        Used to collapse down lists of losses into a single scalar.
        """
        for loss_name, loss_value in list(losses.items()):
            if isinstance(loss_value, torch.Tensor):
                losses[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        return losses

    def _sum_update_losses(self, losses, new_losses):
        """Updates losses with new_losses, adding overlaps.

        Collapses both losses and new_losses. Performs changes in-place
        """
        losses = self._collapse_losses(losses)
        new_losses = self._collapse_losses(new_losses)
        for k in new_losses:
            if k in losses:
                losses[k] += new_losses[k]
            else:
                losses[k] = new_losses[k]

        return losses

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        if self.set_teacher_eval:
            self.teacher.eval()
        losses = self(**data)
        vis = losses.pop('vis', dict())
        log_vars = losses.pop('log_vars', dict())
        loss, log_vars_ = self._parse_losses(losses)
        log_vars.update(log_vars_)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']),
            vis=vis)

        return outputs

    def forward_train(self, lab_stu, lab_tea, unlab_stu, unlab_tea, *args,
                      **kwargs):
        ### Following ssl_dataset, need to collapse unlab_stu and unlab_tea
        ### (which are lists) into a batch
        if isinstance(unlab_stu, list):
            # Required if using "new_ssl_dataset.py" or "ssl_dataset.py"
            # TODO: Double check
            new_unlab_stu = dict()
            for k in unlab_stu[0].keys():
                new_unlab_stu[k] = [
                    sample for batch in unlab_stu for sample in batch[k]
                ]
                if k in ['img']:
                    # some need to be a single tensor, not a list of
                    new_unlab_stu[k] = torch.stack(new_unlab_stu[k], dim=0)
            unlab_stu = new_unlab_stu

            new_unlab_tea = dict()
            for k in unlab_tea[0].keys():
                new_unlab_tea[k] = [
                    sample for batch in unlab_tea for sample in batch[k]
                ]
                if k in ['img']:
                    new_unlab_tea[k] = torch.stack(new_unlab_tea[k], dim=0)
            unlab_tea = new_unlab_tea

        ### Labeled
        ## Assert these so I don't have to worry about them at all
        assert lab_stu.get('gt_bboxes_ignore', None) is None and \
               lab_tea.get('gt_bboxes_ignore', None) is None
        lab_dict = dict(
            stu=lab_stu, tea=lab_tea, sup_losses=dict(), ssl_losses=dict())

        ## Go through SSL Modules
        for ssl_module in self.lab_ssl_modules:
            lab_dict = ssl_module.forward(self, lab_dict)

        ### Unlabeled
        unlab_dict = dict(stu=unlab_stu, tea=unlab_tea, ssl_losses=dict())

        ## Go through SSL Modules
        for ssl_module in self.unlab_ssl_modules:
            unlab_dict = ssl_module.forward(self, unlab_dict)

        ### Aggregate losses
        losses = dict()

        ## Only labeled has sup_losses
        losses.update(add_prefix(lab_dict['sup_losses'], 'sup'))

        ## Aggregate ssl_losses, adding prefixes.
        ssl_losses = dict()
        ssl_losses.update(add_prefix(lab_dict['ssl_losses'], 'lab'))
        ssl_losses.update(add_prefix(unlab_dict['ssl_losses'], 'unlab'))

        ## Pop off visualizations
        vis_dict = dict()
        if 'vis' in lab_dict:
            vis_dict.update({f'lab/{k}': v
                             for k, v in lab_dict['vis'].items()})
        if 'vis' in unlab_dict:
            vis_dict.update({f'unlab/{k}': v
                             for k, v in unlab_dict['vis'].items()})

        ## Pop off log_vars
        log_vars_dict = dict()
        if 'log_vars' in lab_dict:
            log_vars_dict.update({f'lab/{k}': v
                                 for k, v in lab_dict['log_vars'].items()})
        if 'log_vars' in unlab_dict:
            log_vars_dict.update({f'unlab/{k}': v
                                 for k, v in unlab_dict['log_vars'].items()})

        ## Weight ssl_losses
        # Get curr weight
        curr_ssl_weight = self._get_curr_ssl_weight()
        losses['ssl.weight'] = \
            list(losses.values())[0].new_tensor(curr_ssl_weight)

        # Weight ssl losses & put into overal losses
        ssl_losses = self._collapse_losses(ssl_losses)
        for k in ssl_losses.keys():
            if '.metrics' not in k and '.acc' not in k:
                ssl_losses[k] = ssl_losses[k] * curr_ssl_weight
        losses.update(add_prefix(ssl_losses, 'ssl'))

        ## Put vis back in output dict
        losses['vis'] = vis_dict
        losses['log_vars'] = log_vars_dict

        ### Update Teacher
        losses['ssl.ema_decay'] = \
            list(losses.values())[0].new_tensor(self._get_curr_ema_decay())
        self._update_teacher()

        return losses

    def forward_test(self, **kwargs):
        # Does not work with aug_test
        for name in ['img', 'points', 'img_metas']:
            if name in kwargs:
                assert isinstance(kwargs[name], list)
                kwargs[name] = kwargs[name][0]

        return self.simple_test(**kwargs)

    def simple_test(self, **kwargs):
        teacher_bbox_results = self.teacher.simple_test(**kwargs)
        student_bbox_results = self.student.simple_test(**kwargs)

        bbox_results = [
            dict(teacher=teacher_res,
                 student=student_res) for teacher_res, student_res in zip(
                     teacher_bbox_results, student_bbox_results)
        ]

        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        raise Exception('To be implemented by child classes')

    def extract_feat(self, points, img_metas=None):
        raise Exception('Irrelevant')

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        raise Exception('Irrelevant')