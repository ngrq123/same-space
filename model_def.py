from typing import Any, Dict, Union, Sequence, Tuple, cast

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import Normalize, RandomHorizontalFlip, RandomRotation, Resize
import torch

from scripts.dataset_duplicate_image import DuplicateImageDataset
from data import download_data

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class FlattenAdaptiveAvgPool2d(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(tensor, 1)


class SiameseNetwork(nn.Module):
    def __init__(self, net1: nn.Module, net2: nn.Module) -> None:
        super().__init__()
        self.net1 = net1
        self.net2 = net2


    def forward(self, images: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img1, img2 = images
        output_1, output_2 = self.net1(img1.float()), self.net2(img2.float())
        output = torch.cat((output_1, output_2), dim=1)
        return output


class SiameseDuplicateImageDetectionTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        download_dir = './data'
        download_data(download_dir)

        self.train_dataset = DuplicateImageDataset('./data/Airbnb Data/Training Data',
            transforms=[
                Resize((256, 256), antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ],
            upsample_transforms_dict={
                'hflip': RandomHorizontalFlip(p=1),
                'anticlockwise_rot': RandomRotation((5, 5)),
                'clockwise_rot': RandomRotation((-5, -5))
            })
        self.validate_dataset = DuplicateImageDataset('./data/Airbnb Data/Test Data',
            transforms=[
                Resize((256, 256), antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ],
            upsample_transforms_dict={
                'hflip': RandomHorizontalFlip(p=1),
                'anticlockwise_rot': RandomRotation((5, 5)),
                'clockwise_rot': RandomRotation((-5, -5))
            })

        # Instantiate MobileNetV3, and load pre-trained weights
        mobilenetv3_1 = mobilenet_v3_large(pretrained=True)
        mobilenetv3_2 = mobilenet_v3_large(pretrained=True)

        ### Build MobileNetV3 without last layer
        model_submodule_excl_last = list(mobilenetv3_1.children())[:-1]
        model_submodule_excl_last = nn.Sequential(*model_submodule_excl_last)
        flatten_layer = FlattenAdaptiveAvgPool2d()  # From MobileNetV3 PyTorch source code, there is a flatten in forward() before classifier
        flatten_submodule = nn.Sequential(flatten_layer)
        model_last_submodule = list(list(mobilenetv3_1.children())[-1].children())[:-1]  # Remove last layer from last submodule (classifier)
        model_last_submodule = nn.Sequential(*model_last_submodule)

        # Freeze layers except last submodule (classifier)
        for param in model_submodule_excl_last.parameters():
            param.requires_grad_(False)
        for param in flatten_submodule.parameters():
            param.requires_grad_(False)
        
        mobilenetv3_1 = nn.Sequential(*([model_submodule_excl_last] + [flatten_submodule] + [model_last_submodule]))

        ### Build MobileNetV3 without last layer
        model_submodule_excl_last = list(mobilenetv3_2.children())[:-1]
        model_submodule_excl_last = nn.Sequential(*model_submodule_excl_last)
        flatten_layer = FlattenAdaptiveAvgPool2d()  # From MobileNetV3 PyTorch source code, there is a flatten in forward() before classifier
        flatten_submodule = nn.Sequential(flatten_layer)
        model_last_submodule = list(list(mobilenetv3_2.children())[-1].children())[:-1]  # Remove last layer from last submodule (classifier)
        model_last_submodule = nn.Sequential(*model_last_submodule)

        # Freeze layers except last submodule (classifier)
        for param in model_submodule_excl_last.parameters():
            param.requires_grad_(False)
        for param in flatten_submodule.parameters():
            param.requires_grad_(False)
        
        mobilenetv3_2 = nn.Sequential(*([model_submodule_excl_last] + [flatten_submodule] + [model_last_submodule]))

        duplicate_image_classifier = nn.Sequential(
            nn.Linear(1280 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(self.context.get_hparam('dropout1')),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(self.context.get_hparam('dropout2')),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.context.get_hparam('dropout3')),
            nn.Linear(512, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )

        model = nn.Sequential(
            SiameseNetwork(mobilenetv3_1, mobilenetv3_2),
            duplicate_image_classifier
        )

        # Wrap the model.
        self.model = self.context.wrap_model(model)
        ##### Model built #####

        self.optimizer = self.context.wrap_optimizer(
            torch.optim.RMSprop(
                [params for params in self.model.parameters() if params.requires_grad],
                lr=self.context.get_hparam('learning_rate'),
                alpha=self.context.get_hparam('alpha')
            )
        )


    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True)
    

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(self.validate_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True)


    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)
        img_1s, img_2s, labels = batch

        # Pass through Siamese network
        output = self.model((img_1s, img_2s))

        pred = torch.flatten(output)
        labels = labels.type(torch.float)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, labels)

        # Backpropagation
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        m = nn.Sigmoid()
        pred = m(pred)
        pred = (pred > 0.5).type(torch.int)
        correct = (pred == labels).sum().item()

        return {'loss': loss, 'accuracy': (correct / labels.shape[0])}


    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)
        img_1s, img_2s, labels = batch

        # Pass through Siamese network
        output = self.model((img_1s, img_2s))

        pred = torch.flatten(output)
        labels = labels.type(torch.float)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, labels)

        m = nn.Sigmoid()
        pred = m(pred)
        pred = (pred > 0.5).type(torch.int)
        correct = (pred == labels).sum().item()

        return {'validation_loss': loss, 'validation_accuracy': (correct / labels.shape[0])}
