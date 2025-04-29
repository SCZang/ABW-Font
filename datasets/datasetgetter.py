import torch
from torchvision.datasets import ImageFolder
import os
import random
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap, CrossdomainFolder, \
    ImageFolerRemapPairCF, ImageFolerRemapUnpairCF, ImageFolerRemapPairbasis, \
    ImageFolerRemapPair, TwoDataset

class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img


def get_dataset(args, data_dir=None, class_to_use=None, with_path=False):
    #data/imgs/Seen400_S80F50_TRAIN800 \
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    transform_val = Compose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.ToTensor(),
                                       normalize])

    class_to_use = class_to_use or args.att_to_use
    remap_table = {k: i for i, k in enumerate(class_to_use)}
    # if args.local_rank == 0:
    #     print(f'USE CLASSES: {class_to_use}\nLABEL MAP: {remap_table}')

    img_dir = data_dir or args.data_dir

    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table, with_path=with_path)
    valdataset = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table, with_path=with_path)
    # parse classes to use
    tot_targets = torch.tensor(dataset.targets)

    if True: # my implement
        train_dataset = {'TRAIN': dataset, 'FULL': dataset}
        subset_indices = random.sample(range(len(valdataset)), args.val_num)
        val_dataset = torch.utils.data.Subset(valdataset, subset_indices)

    return train_dataset, val_dataset

def get_full_dataset_ft(args, data_dir=None, class_to_use=None, with_path=False,
        ft_ignore_target=-1):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    class_to_use = class_to_use or args.att_to_use
    remap_table = {k: i for i, k in enumerate(class_to_use)}
    if args.local_rank == 0:
        print(f'USE CLASSES: {class_to_use}\nLABEL MAP: {remap_table}')

    img_dir = data_dir or args.data_dir
    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table, with_path=with_path)
    dataser_ft = ImageFolerRemapPair(img_dir, transform=transform, remap_table=remap_table, ignore_target=ft_ignore_target)
    # parse classes to use
    # tot_targets = torch.tensor(dataset.targets)

    # min_data, max_data = 99999999, 0
    # train_idx, val_idx = None, None
    # for k in class_to_use:
    #     tmp_idx = (tot_targets == k).nonzero(as_tuple=False)
    #     min_data = min(min_data, len(tmp_idx))
    #     max_data = max(max_data, len(tmp_idx))
    full_dataset = {'FULL': dataset, 'FT': dataser_ft}
    # args.min_data, args.max_data = min_data, max_data
    # if args.local_rank == 0:
    #     print(f"MIN/MAX DATA: {args.min_data}/{args.max_data}")
    return full_dataset

def get_full_dataset_cfft(args, data_dir=None, base_dir=None, base_ft_dir=None, class_to_use=None, with_path=False,
        ft_ignore_target=-1, class_to_use_base=None):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    class_to_use = class_to_use or args.att_to_use
    class_to_use_base = class_to_use_base or args.att_to_use_base
    remap_table = {k: i for i, k in enumerate(class_to_use)}
    remap_table_base = {k: i for i, k in enumerate(class_to_use_base)}
    if args.local_rank == 0:
        print(f'USE CLASSES: {class_to_use}\nLABEL MAP: {remap_table}\nBASE LABEL MAP: {remap_table_base}')

    img_dir = data_dir or args.data_dir
    img_base_dir = base_dir or args.base_dir
    img_base_ft_dir = base_ft_dir or args.base_ft_dir

    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table, with_path=with_path)
    #    --data_path ${target_style} \args.data_dir = args.data_path\Unseen100_S80F50_FS16
    dataset_base = ImageFolerRemapPair(img_base_dir, transform=transform, remap_table=remap_table_base, with_path=with_path)
    #    --basis_folder ${basis_content_folder} \args.base_dir = args.basis_folder\400BASIS_S80F50_TEST5646
    dataser_full_ft = ImageFolerRemapPair(img_dir, transform=transform, remap_table=remap_table, ignore_target=ft_ignore_target)
    dataser_base_ft = ImageFolerRemapPair(img_base_ft_dir, transform=transform, remap_table=remap_table_base, ignore_target=ft_ignore_target)
    dataser_ft = TwoDataset(dataser_full_ft, dataser_base_ft)
    #    --basis_ft_folder ${basis_style_ft_folder} \args.base_ft_dir = args.basis_ft_folder\400BASIS_S80F50_FS16

    # parse classes to use
    tot_targets = torch.tensor(dataset.targets)

    # min_data, max_data = 99999999, 0
    # train_idx, val_idx = None, None
    # for k in class_to_use:
    #     tmp_idx = (tot_targets == k).nonzero(as_tuple=False)
    #     min_data = min(min_data, len(tmp_idx))
    #     max_data = max(max_data, len(tmp_idx))
    full_dataset = {'FULL': dataset, 'BASE': dataset_base, 'FT': dataser_ft}
    # args.min_data, args.max_data = min_data, max_data
    # if args.local_rank == 0:
    #     print(f"MIN/MAX DATA: {args.min_data}/{args.max_data}")
    return full_dataset

def get_full_dataset(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    dataset = ImageFolerRemap(args.data_dir, transform=transform, remap_table=args.remap_table)
    return dataset


def get_cf_dataset(args):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])

    class_to_use = args.att_to_use

    # if args.local_rank == 0:
    #     print('USE CLASSES', class_to_use)

    # remap labels
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1

    # if args.local_rank == 0:
    #     print("LABEL MAP:", remap_table)

    img_dir = args.data_dir

    # cf_dataset = ImageFolerRemapPairCF(img_dir, base_idxs=args.base_idxs, base_ws=args.base_ws,transform=transform, remap_table=remap_table, \
    #     sample_skip_base=True, sample_N=args.sample_N)
    cf_dataset = ImageFolerRemapUnpairCF(img_dir, base_idxs=args.base_idxs, base_ws=args.base_ws, transform=transform, remap_table=remap_table, top_n=args.base_top_n)
    cf_basis_dataset = ImageFolerRemapPairbasis(img_dir, base_idxs=args.base_idxs, base_ws=args.base_ws,transform=transform, remap_table=remap_table) # keep bs 1
    
    return cf_dataset, cf_basis_dataset
