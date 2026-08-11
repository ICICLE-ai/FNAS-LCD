'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
# Compatibility for legacy bytecode-only autoaugment.pyc under NumPy >= 1.24.
if not hasattr(np, 'int'):
    np.int = int
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import math

from . import autoaugment

_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}
lighting_param = 0.1


params_dict = {
    'auto_imagefolder': {
        # Generic user-supplied ImageFolder dataset (train/val/test under opt.data_dir).
        # No fixed dirs/class-count here; everything is resolved from opt.data_dir
        # and the dataset itself at load time (see _get_data_'s 'auto_imagefolder' branch).
        'num_train_samples': 0,  # Will be determined from dataset
        'num_val_samples': 0,
        'num_test_samples': 0,
        'num_classes': 0,  # Caller supplies --num_classes explicitly
    },
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def load_imagenet_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_ImageFolderClass,
                       dataloader_testing, frac=1, img_per_class=None):
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train_augment == False:
        assert random_erase == False and auto_augment == False
        transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC), transforms.CenterCrop(input_image_size),
                          transforms.ToTensor(), transforms_normalize]

    else:
        if auto_augment:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              autoaugment.ImageNetPolicy(),
                              transforms.ToTensor(),
                              Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        else:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(0.4, 0.4, 0.4),
                              transforms.ToTensor(),
                              Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        pass

        if random_erase:
            transform_list.append(transforms.RandomErasing())
    pass

    transformer = transforms.Compose(transform_list)
    the_dataset = dataset_ImageFolderClass(data_dir, transformer)

    if set_name == 'train' and img_per_class is not None and img_per_class != -1:
        print(f'[DataLoader] Using {img_per_class} images per class')
        np.random.seed(42)
        idxs = []
        for class_idx in np.unique(the_dataset.targets):
            targets = torch.tensor(the_dataset.targets)
            indices_of_class_b = (targets == class_idx).nonzero(as_tuple=True)[0]
            print(f'[DataLoader] Using {img_per_class} images for class {class_idx}')
            class_idxs = np.random.choice(indices_of_class_b, img_per_class, replace=True)
            idxs.extend(class_idxs.tolist())
        the_dataset = torch.utils.data.Subset(the_dataset, indices=idxs)
        print(f'[DataLoader] Subsampled dataset to {len(the_dataset)} images')

    elif 'train' in data_dir and frac > 1:
        print('[DataLoader] Shuffling idxs')
        np.random.seed(42)

        idxs = np.random.choice(range(len(the_dataset)),round(len(the_dataset)/frac), replace=False)
        the_dataset = torch.utils.data.Subset(the_dataset, idxs)


    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        kk = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, kk)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

        pass
    pass

    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }


def _get_data_(dataset_name=None, set_name=None, batch_size=None, train_augment=False, random_erase=False, auto_augment=False,
             input_image_size=224, input_image_crop=0.875, rank=0, world_size=1, shuffle=False,
             num_workers=6, drop_last=False, dataset_ImageFolderClass=None, dataloader_testing=False, argv=None, frac=1, img_per_class=None,
             data_dir=None):

    if dataset_name == 'auto_imagefolder':
        # Generic user-supplied ImageFolder dataset: train/val/test subdirs under data_dir.
        assert data_dir is not None, "auto_imagefolder requires data_dir"
        subdir = set_name if set_name in ('train', 'val', 'test') else 'val'
        resolved_dir = os.path.join(data_dir, subdir)
        if not os.path.isdir(resolved_dir):
            # Fall back to val/ if the requested split doesn't exist (e.g. no test/)
            resolved_dir = os.path.join(data_dir, 'val')

        if dataset_ImageFolderClass is None:
            dataset_ImageFolderClass = datasets.ImageFolder

        return load_imagenet_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                                  random_erase=random_erase, auto_augment=auto_augment,
                                  data_dir=resolved_dir,
                                  input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                                  world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=drop_last,
                                  dataset_ImageFolderClass=dataset_ImageFolderClass,
                                  dataloader_testing=dataloader_testing,
                                  frac=frac,
                                  img_per_class=img_per_class)

    raise ValueError(f"Unknown dataset_name={dataset_name!r}; only 'auto_imagefolder' is supported")


def get_data(opt, argv):
    dataset_name = opt.dataset
    batch_size = opt.batch_size_per_gpu
    random_erase = opt.random_erase
    auto_augment = opt.auto_augment
    input_image_size = opt.input_image_size
    input_image_crop = opt.input_image_crop
    rank = opt.rank
    world_size = opt.world_size
    num_workers = opt.workers_per_gpu

    # check if independent training
    if opt.independent_training:
        rank = 0
        world_size = 1


    # load train set
    set_name = 'train'
    if opt.no_data_augment:
        train_augment = False
    else:
        train_augment = True
    shuffle = True
    drop_last = True

    frac = getattr(opt, "frac", 1)

    data_dir = getattr(opt, "data_dir", None)

    train_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
             input_image_size, input_image_crop, rank, world_size, shuffle,
             num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv, frac=frac,
             img_per_class=getattr(opt, "img_per_class", -1), data_dir=data_dir)

    print('train info got')
    # load val set
    set_name = 'val'
    train_augment = False
    random_erase = False
    auto_augment = False
    shuffle = False
    drop_last = False

    val_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
                                    input_image_size, input_image_crop, rank, world_size, shuffle,
                                    num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv, frac=1, img_per_class=None, data_dir=data_dir)

    # load test set (only if an ImageFolder test/ split exists; otherwise reuse val)
    set_name = 'test'
    train_augment = False
    random_erase = False
    auto_augment = False
    shuffle = False
    drop_last = False
    if dataset_name == 'auto_imagefolder' and os.path.isdir(os.path.join(data_dir, 'test')):
        test_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
                                        input_image_size, input_image_crop, rank, world_size, shuffle,
                                        num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv, data_dir=data_dir)
    else:
        test_dataset_info = val_dataset_info

    return {
        'train_loader' : train_dataset_info['data_loader'],
        'val_loader' : val_dataset_info['data_loader'],
        'test_loader' : test_dataset_info['data_loader'],
        'train_sampler': train_dataset_info['sampler'],
        'val_sampler': val_dataset_info['sampler'],
        'test_sampler': test_dataset_info['sampler'],
    }
