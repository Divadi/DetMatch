"""MM stands for multi-modal A very simple wrapper class that contains both a
detector_2d and a detector_3d.

Nothing besides __init__ is even implemented, because detector_2d and 3d will
always be used independently.
"""
import torch

from mmdet.models import DETECTORS
from functools import partial
from collections import OrderedDict
import copy
from ..builder import build_detector


@DETECTORS.register_module()
class MMDetector(torch.nn.Module):

    def __init__(self,
                 detector_2d,
                 detector_3d,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__()

        detector_2d['train_cfg'] = \
            train_cfg['detector_2d'] if train_cfg is not None else None
        detector_2d['test_cfg'] = test_cfg['detector_2d']
        self.detector_2d = build_detector(detector_2d)

        detector_3d['train_cfg'] = \
            train_cfg['detector_3d'] if train_cfg is not None else None
        detector_3d['test_cfg'] = test_cfg['detector_3d']
        self.detector_3d = build_detector(detector_3d)

        self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained):
        if pretrained is not None:
            assert isinstance(pretrained, dict)
            self.detector_2d.load_state_dict(
                torch.load(pretrained['detector_2d'], map_location='cpu')['state_dict'])
            self.detector_3d.load_state_dict(
                torch.load(pretrained['detector_3d'], map_location='cpu')['state_dict'])

    def simple_test(self, **kwargs):
        do_2d = 'img' in kwargs

        points = kwargs.pop('points')
        if do_2d:
            results_2d = self.detector_2d.simple_test(**kwargs)
            kwargs.pop('img')


        # all_in_outs = OrderedDict()
        # def save_output(name, module, input, output):
        #     all_in_outs[name] = (copy.deepcopy(input), copy.deepcopy(output))
        # for n, m in self.detector_3d.named_modules():
        #     if hasattr(m, 'register_forward_hook'):
        #         m.register_forward_hook(partial(save_output, n))
        #     else:
        #         print(n)


        # all_ins = OrderedDict()
        # def save_input(name, module, input):
        #     all_ins[name] = copy.deepcopy(input)
        # for n, m in self.detector_3d.named_modules():
        #     if hasattr(m, 'register_forward_pre_hook'):
        #         m.register_forward_pre_hook(partial(save_input, n))
        #     else:
        #         print(n)

        kwargs['points'] = points
        results_3d = self.detector_3d.simple_test(**kwargs)

        if do_2d:
            bbox_results = [
                dict(results_2d=res_2d,
                    results_3d=res_3d) for res_2d, res_3d in zip(
                        results_2d, results_3d)
            ]
        else:
            bbox_results = results_3d


        # print(bbox_results[0]['results_3d'])
        # # import pickle
        # # pickle.dump((kwargs, bbox_results[0]['results_3d'], all_ins, all_in_outs), open("/home/msc-auto/src/new_code/DetMatch/outputs/detmatch_good_ordered.pkl", "wb+"))
        # assert False

        return bbox_results
"""

        all_in_outs = OrderedDict()
        def save_output(name, module, args, kwargs, output):
            all_in_outs[name] = (copy.deepcopy((args, kwargs)), copy.deepcopy(output))
        for n, m in self.detector_3d.named_modules():
            if hasattr(m, 'register_forward_hook'):
                m.register_forward_hook(partial(save_output, n), with_kwargs=True)
            else:
                print(n)


        all_ins = OrderedDict()
        def save_input(name, module, args, kwargs):
            all_ins[name] = copy.deepcopy((args, kwargs))
        for n, m in self.detector_3d.named_modules():
            if hasattr(m, 'register_forward_pre_hook'):
                m.register_forward_pre_hook(partial(save_input, n), with_kwargs=True)
            else:
                print(n)

        kwargs['points'] = points
        results_3d = self.detector_3d.simple_test(**kwargs)

        if do_2d:
            bbox_results = [
                dict(results_2d=res_2d,
                    results_3d=res_3d) for res_2d, res_3d in zip(
                        results_2d, results_3d)
            ]
        else:
            bbox_results = results_3d


        print(bbox_results[0]['results_3d'])
        import pickle
        pickle.dump((kwargs, bbox_results[0]['results_3d'], all_ins, all_in_outs), open("/home/msc-auto/src/new_code/DetMatch/outputs/detmatch_bad_ordered.pkl", "wb+"))
        assert False

        return bbox_results
"""