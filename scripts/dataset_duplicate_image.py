import os
import pathlib

from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

class DuplicateImageDataset(Dataset):
    def __init__(self, img_dir: str, transforms: list[torch.nn.Module] = None) -> None:
        self.img_dir = img_dir
        self.transforms = transforms
        # Get image directories
        self.img_paths = []
        for (dirpath, _, filenames) in os.walk(img_dir):
            for filename in filenames:
                extension = pathlib.Path(filename).suffix
                if extension == '.jpg':
                    path = os.path.join(dirpath, filename)
                    path = os.path.normcase(path)
                    image = read_image(path)
                    if self.transforms:
                        for transform in self.transforms:
                            image = transform(image)
                    self.img_paths.append((image, path))

    def __len__(self) -> int:
        return len(self.img_paths) ** 2
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_unique_imgs = len(self.img_paths)
        img_1, img_path_1 = self.img_paths[idx // num_unique_imgs]
        img_2, img_path_2 = self.img_paths[idx % num_unique_imgs]
        label = self._get_label(img_path_1, img_path_2)
        return img_1, img_2, label

    def _is_duplicate(self, path_1: str, path_2: str) -> bool:
        filename_1_parts = os.path.basename(path_1).split('_')
        filename_2_parts = os.path.basename(path_2).split('_')
        # {city}_{roomid}_{match number}.jpg
        return filename_1_parts[0] == filename_2_parts[0] and \
            filename_1_parts[1] == filename_2_parts[1]

    def _get_label(self, path_1: str, path_2: str) -> int:
        return int(self._is_duplicate(path_1, path_2))
