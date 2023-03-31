import os
import pdb

import numpy as np
import imageio
import json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import PIL


class ImageDataset(Dataset):
    def __init__(self, hparams):
        self.hparams = hparams
        self._path = hparams.path

        # assert hparams.use_labels
        assert isinstance(self._path, list)
        self.num_domain = len(self._path)
        # self.json_name = hparams.json_name if isinstance(hparams.json_name, list) else [hparams.json_name for d in range(self.num_domain)]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(hparams.resolution),
        ])

        self._name = hparams.dataset_name #os.path.splitext(os.path.basename(self._path))[0]
        self.use_domain_list = self.hparams.get('use_domain_list', [i for i in range(self.num_domain)])
        self.use_domain_list.sort()
        assert 0 <= self.use_domain_list[-1] < self.num_domain
        self.use_domain = hparams.use_domain

        self._all_fnames = []
        for index_d, _path in enumerate(self._path):
            if index_d in self.use_domain_list:
                if os.path.isdir(_path):
                    self._type = 'dir'
                    all_fnames = [[os.path.abspath(os.path.join(root, fname)), index_d] for root, _dirs, files in
                                    os.walk(_path) for fname in files]
                    self._all_fnames.extend(all_fnames)
                else:
                    raise IOError('Path must point to a directory')
        # if os.path.isdir(self._path):
        #     self._type = 'dir'
        #     self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        # else:
        #     raise IOError('Image Path must point to a directory')
        

        PIL.Image.init()
        # self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname[0]) in PIL.Image.EXTENSION)
        np.random.RandomState(hparams.random_seed).shuffle(self._image_fnames)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        # self._raw_idx = np.arange(len(self._image_fnames), dtype=np.int64)
        self._init_data()
        self._raw_idx = np.arange(len(self._image_fnames), dtype=np.int64)
        print(f"Number of images : {self.__len__()}")

    def _init_data(self):
        self._init_data_sub()

    def _init_data_sub(self):
        self._raw_domains = [fname[1] for fname in self._image_fnames]
        self._image_fnames = [fname[0] for fname in self._image_fnames]

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_i = self._raw_idx[idx]
        image = self.get_image(raw_i)
        # label = self.get_label(raw_i)
        # res = {
        #     'image': image,
        #     'label': label,
        # }
        res = {
            'image': image,
        }
        
        return res

    def get_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image = imageio.imread(fname)

        if self.transform is not None:
            image = self.transform(image)
            image = image * 255

        return image

    # def get_label(self, idx):
    #     label = self._raw_labels[idx]
    #     if label.dtype == np.int64:
    #         onehot = np.zeros(self.label_shape, dtype=np.float32)
    #         onehot[label] = 1
    #         label = onehot
    #     return torch.from_numpy(label)

    # def get_domain(self, idx):
    #     index = self._raw_domains[idx]
    #     domain = np.zeros(self.num_domain, dtype=np.float32)
    #     domain[index] = 1
    #     return torch.from_numpy(domain)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    # def get_label_std(self):
    #     return self._raw_labels_std

    @property
    def name(self):
        return self._name