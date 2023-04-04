from collections.abc import Sequence
from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch


class SiameseNetwork(nn.Module):
    def __init__(self, net1: nn.Module, net2: nn.Module) -> None:
        super().__init__()
        self.net1 = net1
        self.net2 = net2


    def forward(self, images: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img1, img2 = images
        output_1, output_2 = self.net1(img1), self.net2(img2)
        output = torch.cat((output_1, output_2), dim=1)
        return output
        

class SiameseDuplicateImageNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.preprocess = weights.transforms()
        
        # Instantiate MobileNetV3, and load pre-trained weights
        mobilenetv3_1 = mobilenet_v3_large(weights=weights)
        mobilenetv3_2 = mobilenet_v3_large(weights=weights)
        
        # Freeze MobileNetV3 models
        for param in mobilenetv3_1.parameters():
            param.requires_grad_(False)
        for param in mobilenetv3_2.parameters():
            param.requires_grad_(False)

        duplicate_image_classifier = nn.Sequential(
            nn.Linear(1000 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )

        self.model = nn.Sequential(
            SiameseNetwork(mobilenetv3_1, mobilenetv3_2),
            duplicate_image_classifier
        )
            

    def forward(self, img_1: torch.Tensor, img_2: torch.Tensor) -> torch.Tensor:
        # Preprocess images
        img1 = self.preprocess(img_1)
        img2 = self.preprocess(img_2)

        # Pass through Siamese network
        output = self.model((img1, img2))
        return output





