# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import random
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose
from .basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class UltrasoundVideoDataset(BaseDataset):
    METAINFO = dict(classes=('background', 'thyroid nodule'),
                    palette=[[89, 239, 8], [239, 29, 7]])
                    # palette=[[0], [1]])

    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None,
                 train: bool = False) -> None:

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None
        self.data_root = data_root
        self.train = train
        self.data_index = []

        img_path_list = []
        seg_map_path_list = []
        for video in os.listdir(os.path.join(data_root, data_prefix['img_path'])) :
            img_path_list.append(os.path.join(data_prefix['img_path'], video))
        for video in os.listdir(os.path.join(data_root, data_prefix['seg_map_path'])) :
            seg_map_path_list.append(os.path.join(data_prefix['seg_map_path'], video))
        real_dataprefix = {'img_path' : img_path_list,
                           'seg_map_path' : seg_map_path_list}

        # self.data_prefix = copy.copy(data_prefix)
        self.data_prefix = copy.copy(real_dataprefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    def _join_prefix(self):
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, list):
                raise TypeError('prefix should be a list, but got '
                                f'{type(prefix)}')
            a = []
            for prefixa in self.data_prefix[data_key]:
                a.append(os.path.join(self.data_root,prefixa))
            self.data_prefix[data_key] = a


    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """Require label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_map` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette

    '''
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img_dir_num in range(len(img_dir)) :
                for img_ann_num in range(len(os.listdir(img_dir[img_dir_num]))):
                    img = os.listdir(osp.join(self.data_prefix['img_path'][img_dir_num]))[img_ann_num]
                    ann = os.listdir(osp.join(self.data_prefix['seg_map_path'][img_dir_num]))[img_ann_num]
                    data_info = dict(img_path=osp.join(self.data_prefix['img_path'][img_dir_num], img))
                    # data_info['seg_map_path'] = osp.join(self.data_prefix['seg_map_path'][img_dir_num], ann)
                    data_info['seg_map_path'] = data_info['img_path'].replace('img_dir', 'ann_dir')
                    data_info['seg_map_path'] = data_info['seg_map_path'].replace('.jpg','.png')
                    data_info['label_map'] = self.label_map
                    data_info['reduce_zero_label'] = self.reduce_zero_label
                    data_info['seg_fields'] = []
                    data_list.append(data_info)
                    data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
'''
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        img_dir.sort(key=lambda x: int(x[-4:]))



        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            if self.train:
                index = 0
                data_index = []
                for img_dir_num in range(len(img_dir)):
                    number_all = []
                    all_path = sorted(os.listdir(img_dir[img_dir_num]), key=lambda name: int(name[-8:-4]))
                    index_one = 0
                    for tt in range(100):
                        if(index_one >= len(all_path)):
                            break
                        else:
                            if(tt == 0):
                                index_temp = random.randint(0, 10)
                                if (index_one + index_temp <= len(all_path)):
                                    data_index.append([index, index + index_temp])
                                    index_one += index_temp
                                    index += index_temp
                                    number_all = number_all + list(range(0, index_temp))[:]
                                else:
                                    data_index.append([index, index + len(all_path) - index_one])
                                    index += len(all_path) - index_one
                                    number_all = number_all + list(range(0, len(all_path) - index_one))
                                    index_one += len(all_path) - index_one
                            else:
                                index_temp = 10#10
                                if (index_one + index_temp <= len(all_path)):
                                    data_index.append([index, index + index_temp])
                                    index_one += index_temp
                                    index += index_temp
                                    number_all = number_all + list(range(0, index_temp))[:]
                                else:
                                    data_index.append([index, index + len(all_path) - index_one])
                                    index += len(all_path) - index_one
                                    number_all = number_all + list(range(0, len(all_path) - index_one))
                                    index_one += len(all_path) - index_one

                            #index_temp = random.randint(5, 15)
                            #if(index_one + index_temp <= len(all_path)):
                            #    data_index.append([index, index + index_temp])
                            #    index_one += index_temp
                            #    index += index_temp
                            #    number_all = number_all + list(range(0, index_temp))[:]
                            #else:
                            #    data_index.append([index, index + len(all_path) - index_one])
                            #    index  += len(all_path) - index_one
                            #    number_all = number_all + list(range(0, len(all_path) - index_one))
                            #    index_one += len(all_path) - index_one


                    # self.data_index.append([index, index + len(all_path)])
                    # index += len(all_path)
                    augment = np.zeros([6, 2])
                    for img_ann_num in range(len(all_path)):
                        img = all_path[img_ann_num]
                        # ann = os.listdir(osp.join(self.data_prefix['seg_map_path'][img_dir_num]))[img_ann_num]
                        data_info = dict(img_path=osp.join(img_dir[img_dir_num], img))
                        # data_info['seg_map_path'] = osp.join(self.data_prefix['seg_map_path'][img_dir_num], ann)
                        data_info['seg_map_path'] = data_info['img_path'].replace('img_dir', 'ann_dir')
                        data_info['seg_map_path'] = data_info['seg_map_path'].replace('.jpg', '.png')
                        data_info['label_map'] = self.label_map
                        data_info['reduce_zero_label'] = self.reduce_zero_label
                        data_info['seg_fields'] = []
                        data_info['number'] = number_all[img_ann_num]
                        if number_all[img_ann_num] == 0:
                            augment[:, 0] = np.random.randint(0, 10, size=(6))
                            augment[0, 1] = np.random.uniform(0.8, 1.2)
                            augment[1, 1] = np.random.uniform(0.8, 1.2)
                            augment[2, 1] = np.random.uniform(0.8, 1.2)
                            augment[3, 1] = np.random.uniform(-0.1, 0.1)
                            augment[4, 1] = np.random.randint(0, 3)
                            data_info['augment'] = augment
                        else:
                            data_info['augment'] = augment
                        data_list.append(data_info)
                        data_list = sorted(data_list, key=lambda x: x['img_path'])
                self.data_index = data_index
            else:
                for img_dir_num in range(len(img_dir)):
                    all_path = sorted(os.listdir(img_dir[img_dir_num]), key=lambda name: int(name[-8:-4]))
                    # self.data_index.append([index, index + len(all_path)])
                    # index += len(all_path)
                    for img_ann_num in range(len(all_path)):
                        img = all_path[img_ann_num]
                        # ann = os.listdir(osp.join(self.data_prefix['seg_map_path'][img_dir_num]))[img_ann_num]
                        data_info = dict(img_path=osp.join(img_dir[img_dir_num], img))
                        # data_info['seg_map_path'] = osp.join(self.data_prefix['seg_map_path'][img_dir_num], ann)
                        data_info['seg_map_path'] = data_info['img_path'].replace('img_dir', 'ann_dir')
                        data_info['seg_map_path'] = data_info['seg_map_path'].replace('.jpg', '.png')
                        data_info['label_map'] = self.label_map
                        data_info['reduce_zero_label'] = self.reduce_zero_label
                        data_info['seg_fields'] = []
                        data_info['number'] = (int(all_path[img_ann_num][:-4]) -  int(all_path[0][:-4])) % 10
                        data_list.append(data_info)
                        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list



@DATASETS.register_module()
class UltrasoundImageDataset(BaseSegDataset):
    METAINFO = dict(classes = ('background', 'thyroid nodule'),
    palette = [[89,239,8], [239,29,7]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs):
        super().__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
