from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch

class SiameseDuplicateImageNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.preprocess = weights.transforms()
        
        # Instantiate MobileNetV3, and load pre-trained weights
        self.network_1 = mobilenet_v3_large(weights=weights)
        self.network_2 = mobilenet_v3_large(weights=weights)
        
        # Freeze MobileNetV3 models
        for param in self.network_1.parameters():
            param.requires_grad_(False)
        for param in self.network_2.parameters():
            param.requires_grad_(False)

        self.duplicate_image_classifier = nn.Sequential(
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
            

    def forward(self, img_1, img_2):
        # Preprocess images
        img1 = self.preprocess(img_1)
        img2 = self.preprocess(img_2)

        # Pass through Siamese network
        model_1_output = self.network_1(img1)
        model_2_output = self.network_2(img2)

        output_concat = torch.cat((model_1_output, model_2_output), 1)
        output = self.duplicate_image_classifier(output_concat)
        return output





