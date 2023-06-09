import copy
import torch
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from mmdet3d.datasets import build_dataset
from mmdet.datasets import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class TS_SSL_Dataset(Dataset):
    def __init__(self,
                 dataset,
                 student_pipeline=[],
                 teacher_pipeline=[]):

        self.dataset = build_dataset(dataset)
        self.CLASSES = self.dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag

        self.student_pipeline = Compose(student_pipeline)
        self.teacher_pipeline = Compose(teacher_pipeline)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        student_data = self.student_pipeline(copy.deepcopy(data))
        teacher_data = self.teacher_pipeline(data)

        return dict(
            stu=student_data, tea=teacher_data, img_metas=DC(torch.tensor([])))

    def __len__(self):
        return len(self.dataset)