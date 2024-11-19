import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import torch
import evaluate_utils
from dataset.image_folder_dataset import CustomImageFolderDataset
from dataset.record_dataset import AugmentRecordDataset

class LFWDataset(Dataset):
    def __init__(self, lfw_data):
        """
        Args:
            lfw_data: (lfw_images, lfw_issame) のタプル
        """
        lfw_images, lfw_issame = lfw_data
        # LFWのインデックスは2（FiveValidationDatasetと同じ）
        self.dataname_to_idx = {"lfw": 2}
        
        # データの準備
        self.images = lfw_images
        # issameの長さを画像数と合わせる
        self.issame = []
        for same in lfw_issame:
            self.issame.extend([same, same])  # 各ラベルを2回追加
        self.issame = np.array(self.issame)
        # データセット名のインデックスを画像数分作成
        self.dataname = np.array([self.dataname_to_idx["lfw"]] * len(self.images))
        
        # データの整合性チェック
        assert len(self.images) == len(self.issame)
        assert len(self.issame) == len(self.dataname)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        x_np = self.images[index].copy()
        x = torch.tensor(x_np)
        y = self.issame[index]
        dataname = self.dataname[index]
        return x, y, dataname, index

class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.output_dir = kwargs['output_dir']
        self.data_root = kwargs['data_root']
        self.train_data_path = kwargs['train_data_path']
        self.val_data_path = kwargs['val_data_path']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.train_data_subset = kwargs.get('train_data_subset', False)

        self.low_res_augmentation_prob = kwargs.get('low_res_augmentation_prob', 0.0)
        self.crop_augmentation_prob = kwargs.get('crop_augmentation_prob', 0.0)
        self.photometric_augmentation_prob = kwargs.get('photometric_augmentation_prob', 0.0)
        self.swap_color_channel = kwargs.get('swap_color_channel', False)
        self.use_mxrecord = kwargs.get('use_mxrecord', False)

    def prepare_data(self):
        # LFWのmemfileが存在しない場合のみ作成
        lfw_memfile_path = os.path.join(self.data_root, self.val_data_path, 'lfw', 'memfile')
        if not os.path.isdir(lfw_memfile_path):
            print('Making LFW validation data memfile')
            evaluate_utils.get_val_pair(os.path.join(self.data_root, self.val_data_path), 'lfw')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print('Creating train dataset')
            self.train_dataset = train_dataset(
                self.data_root,
                self.train_data_path,
                self.low_res_augmentation_prob,
                self.crop_augmentation_prob,
                self.photometric_augmentation_prob,
                self.swap_color_channel,
                self.use_mxrecord,
                self.output_dir
            )

            if 'faces_emore' in self.train_data_path and self.train_data_subset:
                with open('assets/ms1mv2_train_subset_index.txt', 'r') as f:
                    subset_index = [int(i) for i in f.read().split(',')]
                    self.subset_ms1mv2_dataset(subset_index)

            print('Creating LFW validation dataset')
            val_data = evaluate_utils.get_val_data(os.path.join(self.data_root, self.val_data_path))
            _, _, lfw, _, _, lfw_issame, _, _, _, _ = val_data
            self.val_dataset = LFWDataset((lfw, lfw_issame))

        if stage == 'test' or stage is None:
            lfw, lfw_issame = evaluate_utils.get_val_pair(os.path.join(self.data_root, self.val_data_path), 'lfw')
            self.test_dataset = LFWDataset((lfw, lfw_issame))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    
    def subset_ms1mv2_dataset(self, subset_index):
        # remove too few example identites
        self.train_dataset.samples = [self.train_dataset.samples[idx] for idx in subset_index]
        self.train_dataset.targets = [self.train_dataset.targets[idx] for idx in subset_index]
        value_counts = pd.Series(self.train_dataset.targets).value_counts()
        to_erase_label = value_counts[value_counts<5].index
        e_idx = [i in to_erase_label for i in self.train_dataset.targets]
        self.train_dataset.samples = [i for i, erase in zip(self.train_dataset.samples, e_idx) if not erase]
        self.train_dataset.targets = [i for i, erase in zip(self.train_dataset.targets, e_idx) if not erase]

        # label adjust
        max_label = np.max(self.train_dataset.targets)
        adjuster = {}
        new = 0
        for orig in range(max_label+1):
            if orig in to_erase_label:
                continue
            adjuster[orig] = new
            new += 1

        # readjust class_to_idx
        self.train_dataset.targets = [adjuster[orig] for orig in self.train_dataset.targets]
        self.train_dataset.samples = [(sample[0], adjuster[sample[1]]) for sample in self.train_dataset.samples]
        new_class_to_idx = {}
        for label_str, label_int in self.train_dataset.class_to_idx.items():
            if label_int in to_erase_label:
                continue
            else:
                new_class_to_idx[label_str] = adjuster[label_int]
        self.train_dataset.class_to_idx = new_class_to_idx


def train_dataset(data_root, train_data_path,
                  low_res_augmentation_prob,
                  crop_augmentation_prob,
                  photometric_augmentation_prob,
                  swap_color_channel,
                  use_mxrecord,
                  output_dir):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if use_mxrecord:
        train_dir = os.path.join(data_root, train_data_path)
        train_dataset = AugmentRecordDataset(root_dir=train_dir,
                                             transform=train_transform,
                                             low_res_augmentation_prob=low_res_augmentation_prob,
                                             crop_augmentation_prob=crop_augmentation_prob,
                                             photometric_augmentation_prob=photometric_augmentation_prob,
                                             swap_color_channel=swap_color_channel,
                                             output_dir=output_dir)
    else:
        train_dir = os.path.join(data_root, train_data_path, 'imgs')
        train_dataset = CustomImageFolderDataset(root=train_dir,
                                                 transform=train_transform,
                                                 low_res_augmentation_prob=low_res_augmentation_prob,
                                                 crop_augmentation_prob=crop_augmentation_prob,
                                                 photometric_augmentation_prob=photometric_augmentation_prob,
                                                 swap_color_channel=swap_color_channel,
                                                 output_dir=output_dir
                                                 )

    return train_dataset
