import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import warnings
import bisect
from os.path import join, dirname, exists
from random import random, sample

dataset = {}
dataset["PACS"] = ["art_painting", "cartoon", "photo", "sketch"]

available_datasets = dataset["PACS"]

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class MyDataset(Dataset):
    def __init__(self, names, labels, img_transformer=None, data_dir='./'):
        self.names = names
        self.labels = labels
        self.data_dir = data_dir

        self._image_transformer = img_transformer
    
    def get_image(self, index):
        framename = self.data_dir + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        
        img = self.get_image(index)
        
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)

def get_random_subset(names, labels, percent):
    """
    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val

def _dataset_info(txt_labels, num_classes=10000):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        if int(row[1]) >= num_classes:
            continue
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)

def get_train_dataloader(args):

    dataset_list = args.source
    assert isinstance(dataset_list, list)
    val_datasets = []
    img_transformer = get_train_transformers(args)
    img_num_per_domain = []
    train_loader_list = []
    for dname in dataset_list:
        name_train, labels_train = _dataset_info(join(args.datalist_dir, args.dataset, '%s_train_kfold.txt' % dname))
        name_val, labels_val = _dataset_info(join(args.datalist_dir, args.dataset, '%s_crossval_kfold.txt' % dname))
        train_dataset = MyDataset(name_train, labels_train, img_transformer=img_transformer, data_dir=args.data_dir)
        val_datasets.append(MyDataset(name_val, labels_val, img_transformer=get_val_transformer(args), data_dir=args.data_dir))
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=False)
        img_num_per_domain.append(len(name_train))
        train_loader_list.append(train_loader)

    val_dataset = ConcatDataset(val_datasets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True, drop_last=False)
    return train_loader_list, val_loader, img_num_per_domain

def get_val_dataloader(args):
    
    if isinstance(args.target, list):
        img_tr = get_val_transformer(args)
        val_datasets = []
        for dname in args.target:
            names, labels = _dataset_info(join(args.datalist_dir, args.dataset, '%s_test.txt' % dname))
            val_datasets.append(MyDataset(names, labels, img_transformer=img_tr, data_dir=args.data_dir))
        
        dataset = ConcatDataset(val_datasets)

    else:        
        names, labels = _dataset_info(join(args.datalist_dir, args.dataset, '%s_test.txt' % args.target))
        img_tr = get_val_transformer(args)
        val_dataset = MyDataset(names, labels, img_transformer=img_tr, data_dir=args.data_dir)
        
        dataset = ConcatDataset([val_dataset])

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True, drop_last=False)
    return loader

def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(int(args.image_size), (args.min_scale, args.max_scale))]
    if args.flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)

def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)
