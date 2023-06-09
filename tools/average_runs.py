import argparse
import json
import numpy as np
import os
# import sys
from collections import defaultdict
from os import path as osp

metrics_dict = {
    '3d': ['KITTI/Overall_3D_moderate',
           'KITTI/Pedestrian_3D_moderate_strict',
           'KITTI/Cyclist_3D_moderate_strict',
           'KITTI/Car_3D_moderate_strict'],
    '2d': ['KITTI/Overall_2D_moderate',
           'KITTI/Pedestrian_2D_moderate_strict',
           'KITTI/Cyclist_2D_moderate_strict',
           'KITTI/Car_2D_moderate_strict'],
    'ssl3d': ['tea.3d.KITTI/Overall_3D_moderate',
              'stu.3d.KITTI/Overall_3D_moderate',
              'tea.3d.KITTI/Pedestrian_3D_moderate_strict',
              'stu.3d.KITTI/Pedestrian_3D_moderate_strict',
              'tea.3d.KITTI/Cyclist_3D_moderate_strict',
              'stu.3d.KITTI/Cyclist_3D_moderate_strict',
              'tea.3d.KITTI/Car_3D_moderate_strict',
              'stu.3d.KITTI/Car_3D_moderate_strict'],
    'ssl2d': ['tea.2d.KITTI/Overall_2D_moderate',
              'stu.2d.KITTI/Overall_2D_moderate',
              'tea.2d.KITTI/Pedestrian_2D_moderate_strict',
              'stu.2d.KITTI/Pedestrian_2D_moderate_strict',
              'tea.2d.KITTI/Cyclist_2D_moderate_strict',
              'stu.2d.KITTI/Cyclist_2D_moderate_strict',
              'tea.2d.KITTI/Car_2D_moderate_strict',
              'stu.2d.KITTI/Car_2D_moderate_strict'],
    'fusion': ['tea.3d.KITTI/Overall_3D_moderate',
               'stu.3d.KITTI/Overall_3D_moderate',
               'tea.3d.KITTI/Pedestrian_3D_moderate_strict',
               'stu.3d.KITTI/Pedestrian_3D_moderate_strict',
               'tea.3d.KITTI/Cyclist_3D_moderate_strict',
               'stu.3d.KITTI/Cyclist_3D_moderate_strict',
               'tea.3d.KITTI/Car_3D_moderate_strict',
               'stu.3d.KITTI/Car_3D_moderate_strict',
               'tea.2d.KITTI/Overall_2D_moderate',
               'stu.2d.KITTI/Overall_2D_moderate',
               'tea.2d.KITTI/Pedestrian_2D_moderate_strict',
               'stu.2d.KITTI/Pedestrian_2D_moderate_strict',
               'tea.2d.KITTI/Cyclist_2D_moderate_strict',
               'stu.2d.KITTI/Cyclist_2D_moderate_strict',
               'tea.2d.KITTI/Car_2D_moderate_strict',
               'stu.2d.KITTI/Car_2D_moderate_strict']
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_jsons', metavar='N', type=str, nargs='+')
    parser.add_argument('--type', type=str, default='3d')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    res = defaultdict(list)
    for log_json in args.log_jsons:
        if os.path.isdir(log_json):
            # Find the final *.log.json in this directory
            filtered = list(filter(lambda s: '.log.json' in s,
                                   os.listdir(log_json)))
            log_json = osp.join(log_json, sorted(filtered)[-1])

        last_val_json = json.loads(open(log_json).readlines()[-1].strip())

        for metric in metrics_dict[args.type]:
            if metric not in last_val_json:
                raise Exception('Desired metric {} not in json'.format(metric))

            res[metric].append(last_val_json[metric])

    for k, v in list(res.items()):
        print('{:<40}: {:.02f}(\u00B1{:.02f})'.format(
            k, np.mean(v), np.std(v)))


if __name__ == '__main__':
    main()
