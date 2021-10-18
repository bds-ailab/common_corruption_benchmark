"""Dataset constructers"""

from torch.utils.data import Dataset, ConcatDataset
import torchvision
from torchvision import transforms
import AlbumentationsImageFolder
from candidate_corruptions import dict_corruptions
import os
from albumentations.pytorch import ToTensor, ToTensorV2
import albumentations as album


""" Adapt the preprocessing of images depending on 1) the mode (train/test) 2) the albumentations corruptions used in the data augmentation process 3) the preprocessing mode : the torchvision or the albumentations one (set 'album_mode' to True when an albumentations corruption is used)"""
def set_preprocess(mode, corruption_names=['clean'], album_mode=False):
    if corruption_names != ['clean']:
        corruptions = []
        for corrup in corruption_names:
            corruptions.append(dict_corruptions[corrup])
    else:
        corruptions = corruption_names

    if album_mode == True:
        ImageFolder = AlbumentationsImageFolder.ImageFolder
        if mode=='train':
            if corruptions!=['clean']:
                pre_process = album.Compose([album.RandomResizedCrop(224, 224),album.HorizontalFlip(), album.OneOf(corruptions,p=1),album.Normalize(),ToTensorV2()])
            elif corruptions == ['clean']:
                pre_process = album.Compose([album.RandomResizedCrop(224, 224),album.HorizontalFlip(),album.Normalize(),ToTensorV2()])
        elif mode=='test':
            if corruptions!=['clean']:
                pre_process = album.Compose([album.Resize(256, 256,interpolation=3),album.CenterCrop(224, 224), album.OneOf(corruptions,p=1),album.Normalize(),ToTensorV2()])
            elif corruptions == ['clean']:
                pre_process = album.Compose([album.Resize(256, 256,interpolation=3),album.CenterCrop(224, 224), album.Normalize(),ToTensorV2()])

    elif album_mode == False:
        ImageFolder = torchvision.datasets.ImageFolder
        if mode=='train':
            pre_process = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip() ,transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        elif mode=='test':
            pre_process = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    return pre_process, ImageFolder

""" Load ImageNet-100 corrupted with 'corruption' and using the preprocessing 'album_mode' """
def get_Inet100(dataset_path, mode, corruptions=['clean'], album_mode=False):
    pre_transform, ImageFolder = set_preprocess(mode, corruptions, album_mode)
    if mode == "train":
        dataset = ImageFolder(os.path.join(dataset_path, "train"),transform=pre_transform)
    elif mode == "test":
        dataset = ImageFolder(os.path.join(dataset_path, "validation"),transform=pre_transform)
    return dataset

""" Load ImageNet-1K corrupted with 'corruption' and using the preprocessing 'album_mode' """
def get_Inet(dataset_path, mode, corruptions=['clean'], album_mode=False):
    pre_transform, ImageFolder = set_preprocess(mode, corruptions, album_mode)
    if mode == "train":
        dataset = ImageFolder(os.path.join(dataset_path, "train"),transform=pre_transform)
    elif mode == "test":
        dataset = ImageFolder(os.path.join(dataset_path, "validation"),transform=pre_transform)
    return dataset

""" Load ImageNet-V2 """
def get_Inet1K_V2(dataset_path, album_mode):
    pre_transform, ImageFolder = set_preprocess('test', ['clean'], album_mode)
    dataset = ImageFolder(dataset_path,transform=pre_transform)
    return dataset

""" Load ImageNet-Sketch """
def get_Inet_Sketch(dataset_path, album_mode):
    pre_transform, ImageFolder = set_preprocess('test', ['clean'], album_mode)
    dataset = ImageFolder(dataset_path, transform=pre_transform)
    return dataset

""" Load ImageNet-R """
def get_Inet_r(dataset_path, album_mode):
    pre_transform, ImageFolder = set_preprocess('test', ['clean'], album_mode)
    dataset = ImageFolder(dataset_path, transform=pre_transform)
    return dataset

""" Load ImageNet-Video """
def get_Inet_vid(dataset_path, album_mode):
    pre_transform, ImageFolder = set_preprocess('test', ['clean'], album_mode)
    dataset = ImageFolder(dataset_path,transform=pre_transform)
    return dataset

""" Load ImageNet-A """
def get_Inet_a(dataset_path, album_mode):
    pre_transform, ImageFolder = set_preprocess('test', ['clean'], album_mode)
    dataset = ImageFolder(dataset_path, transform=pre_transform)
    return dataset

""" Load ObjectNet """
def get_ObjectNet(dataset_path, album_mode):
    pre_transform, ImageFolder = set_preprocess('test', ['clean'], album_mode)
    dataset = ImageFolder(dataset_path, transform=pre_transform)
    return dataset

""" Load the ImageNet validation set corrupted 'corruption', with 'corruption' a corruption of ImageNet-C  """
def get_Inet_c(dataset_path, corruption):
    pre_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    dataset1 = torchvision.datasets.ImageFolder(os.path.join(dataset_path,corruption,"1"), transform=pre_transform)
    dataset2 = torchvision.datasets.ImageFolder(os.path.join(dataset_path,corruption,"2"), transform=pre_transform)
    dataset3 = torchvision.datasets.ImageFolder(os.path.join(dataset_path,corruption,"3"), transform=pre_transform)
    dataset4 = torchvision.datasets.ImageFolder(os.path.join(dataset_path,corruption,"4"), transform=pre_transform)
    dataset5 = torchvision.datasets.ImageFolder(os.path.join(dataset_path,corruption,"5"), transform=pre_transform)
    dataset = ConcatDataset([dataset1,dataset2,dataset3,dataset4,dataset5])
    return dataset
