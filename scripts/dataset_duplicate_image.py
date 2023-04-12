from typing import Dict, List, Tuple

import os
import pathlib

from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision.io import read_image
import imagehash
import numpy as np
import pandas as pd
import torch

class DuplicateImageDataset(Dataset):
    def __init__(self, img_dir: str, transforms: List = list(), balance_classes: bool = True, upsample_transforms_dict: Dict[str, torch.nn.Module] = dict()) -> None:
        self.img_dir = img_dir
        self.upsample_transforms_dict = upsample_transforms_dict
        
        # Get image directories
        img_paths = []
        for (dirpath, _, filenames) in os.walk(img_dir):
            for filename in filenames:
                extension = pathlib.Path(filename).suffix
                if extension == '.jpg':
                    path = os.path.join(dirpath, filename)
                    path = os.path.normcase(path)
                    img_paths.append(path)

        # Create full dataset
        self.dataset_df = self._create_full_dataset(img_paths)

        # Balance classes
        if balance_classes:
            for transform_key in self.upsample_transforms_dict:
                self.dataset_df['img1_' + transform_key] = False
                self.dataset_df['img2_' + transform_key] = False
            duplicate_pairs_df = self._upsample_duplicate_pairs()

            # Calculate image hashes
            image_hash_dict = dict()
            for path in img_paths:
                _image = Image.open(path)
                _image = _image.resize((256, 256), resample=Resampling.BILINEAR)
                image_hash_dict[path] = imagehash.whash(_image)
                
            non_duplicate_pairs_df = self._downsample_non_duplicate_pairs(image_hash_dict, duplicate_pairs_df.shape[0])
            del(image_hash_dict)  # Save memory

            self.dataset_df = pd.concat([duplicate_pairs_df, non_duplicate_pairs_df], ignore_index=True)

        # Load and transform images
        self.image_dict = dict()
        for path in img_paths:
            _image = read_image(path).type(torch.float)
            for transform_func in transforms:
                _image = transform_func(_image)
            _image = _image.div(255) # Range has to be between 0 and 1
            self.image_dict[path] = _image


    def __len__(self) -> int:
        return self.dataset_df.shape[0]
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.dataset_df.loc[idx]
        
        img_path_1 = sample['image1']
        img_path_2 = sample['image2']

        img_1 = self.image_dict[img_path_1]
        img_2 = self.image_dict[img_path_2]
        
        for transform_key in self.upsample_transforms_dict.keys():
            if sample['img1_' + transform_key]:
                img_1 = self.upsample_transforms_dict[transform_key](img_1)
            if sample['img2_' + transform_key]:
                img_2 = self.upsample_transforms_dict[transform_key](img_2)
        
        label = sample['label']
        return img_1, img_2, label
    

    def _is_duplicate(self, path_1: str, path_2: str) -> bool:
        filename_1_parts = os.path.basename(path_1).split('_')
        filename_2_parts = os.path.basename(path_2).split('_')
        # {city}_{roomid}_{match number}.jpg
        return filename_1_parts[0] == filename_2_parts[0] and \
            filename_1_parts[1] == filename_2_parts[1]


    def _get_label(self, path_1: str, path_2: str) -> int:
        return int(self._is_duplicate(path_1, path_2))


    def _create_full_dataset(self, paths):
        image1s = []
        image2s = []
        labels = []
        for path1 in paths:
            for path2 in paths:
                image1s.append(path1)
                image2s.append(path2)
                labels.append(self._get_label(path1, path2))
        return pd.DataFrame({
            'image1': image1s,
            'image2': image2s,
            'label': labels
        })


    def _upsample_duplicate_pairs(self):
        duplicate_pairs_df = self.dataset_df.loc[self.dataset_df['label'] == 1].copy()

        if len(self.upsample_transforms_dict) == 0:
            return duplicate_pairs_df
        
        upsampled_df = duplicate_pairs_df.copy()
        
        # Permutate through all upsample transforms
        for transform_key in self.upsample_transforms_dict.keys():
            for transform_img1 in [True, False]:
                for transform_img2 in [True, False]:
                    if transform_img1 or transform_img2:
                        temp_df = duplicate_pairs_df.copy()
                        temp_df['img1_' + transform_key] = transform_img1
                        temp_df['img2_' + transform_key] = transform_img2
                        upsampled_df = pd.concat([upsampled_df, temp_df], ignore_index=True)
            duplicate_pairs_df = upsampled_df.copy()
        
        return upsampled_df
    
    def _downsample_non_duplicate_pairs(self, image_hash_dict, num_duplicate_samples):
        
        def compute_similarity(img1_path, img2_path):
            return image_hash_dict[img1_path] - image_hash_dict[img2_path]
        
        non_duplicate_pairs_df = self.dataset_df.loc[self.dataset_df['label'] == 0].copy()

        if num_duplicate_samples >= non_duplicate_pairs_df.shape[0]:
            return non_duplicate_pairs_df

        # Compute similarity based on image hashes - hamming distance
        non_duplicate_pairs_df['similarity'] = non_duplicate_pairs_df.apply(
            lambda row: compute_similarity(row['image1'], row['image2']),
            axis=1)
        
        # Compute weights to be used for sampling
        # The lower the similarity (more similar), the higher the weight
        non_duplicate_pairs_df['sampling_weight'] = 1 / np.exp(non_duplicate_pairs_df['similarity'])

        downsampled_df = non_duplicate_pairs_df.sample(num_duplicate_samples,
                                                       weights=non_duplicate_pairs_df['sampling_weight'],
                                                       random_state=0)  # Remove randomness in samples

        return downsampled_df.drop(['similarity', 'sampling_weight'], axis=1)

