from collections.abc import Sequence
from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch


class FlattenAdaptiveAvgPool2d(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(tensor, 1)


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

        # Remove last layer
        model_submodule_excl_last = list(mobilenetv3_1.children())[:-1]
        model_submodule_excl_last = nn.Sequential(*model_submodule_excl_last)
        flatten_layer = FlattenAdaptiveAvgPool2d()  # From MobileNetV3 PyTorch source code, there is a flatten in forward() before classifier
        flatten_module = nn.Sequential(flatten_layer)
        model_last_submodule = list(list(mobilenetv3_1.children())[-1].children())[:-1]  # Remove last layer from last submodule (classifier)
        model_last_submodule = nn.Sequential(*model_last_submodule)
        mobilenetv3_1 = model_submodule_excl_last.append(flatten_module).append(model_last_submodule)

        model_submodule_excl_last = list(mobilenetv3_2.children())[:-1]
        model_submodule_excl_last = nn.Sequential(*model_submodule_excl_last)
        flatten_layer = FlattenAdaptiveAvgPool2d()  # From MobileNetV3 PyTorch source code, there is a flatten in forward() before classifier
        flatten_module = nn.Sequential(flatten_layer)
        model_last_submodule = list(list(mobilenetv3_2.children())[-1].children())[:-1]  # Remove last layer from last submodule (classifier)
        model_last_submodule = nn.Sequential(*model_last_submodule)
        mobilenetv3_2 = model_submodule_excl_last.append(flatten_module).append(model_last_submodule)

        duplicate_image_classifier = nn.Sequential(
            nn.Linear(1280 * 2, 2048),
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
    

    def train_loop(self, dataloader: torch.utils.data.DataLoader, loss_fn, optimizer, device):
        size = len(dataloader.dataset)
        for batch, (img_1s, img_2s, ys) in enumerate(dataloader):
            img_1s = img_1s.to(device)
            img_2s = img_2s.to(device)
            ys = ys.to(device)

            # Preprocess images
            img1s = self.preprocess(img_1s)
            img2s = self.preprocess(img_2s)

            pred = self.model((img1s, img2s))
            m = nn.Sigmoid()
            loss = loss_fn(torch.flatten(m(pred)), ys.type(torch.float))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(img_1s)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(self, dataloader, loss_fn, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for img_1s, img_2s, ys in dataloader:
                img_1s = img_1s.to(device)
                img_2s = img_2s.to(device)
                ys = ys.to(device)

                # Preprocess images
                img1s = self.preprocess(img_1s)
                img2s = self.preprocess(img_2s)
                
                pred = self.model((img1s, img2s))
                m = nn.Sigmoid()
                pred = torch.flatten(m(pred))
                test_loss += loss_fn(pred, ys.type(torch.float)).item()
                pred = (pred > 0.5).type(torch.int)
                correct += (pred == ys).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")







