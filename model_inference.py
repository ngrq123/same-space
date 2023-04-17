from determined.experimental import client
from determined import pytorch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, RandomHorizontalFlip, RandomRotation, Resize
import torch

from scripts.dataset_duplicate_image import DuplicateImageDataset


trial = pytorch.load_trial_from_checkpoint_path('checkpoints\9e2b111e-e352-454b-b0ce-a51d5205f514')
model = trial.model
print('Model loaded')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

validation_dataset = DuplicateImageDataset('./data/Airbnb Data/Test Data',
                                            transforms=[
                                                Resize((256, 256), antialias=True),
                                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ],
                                            upsample_transforms_dict={
                                                'hflip': RandomHorizontalFlip(p=1),
                                                'anticlockwise_rot': RandomRotation((5, 5)),
                                                'clockwise_rot': RandomRotation((-5, -5))
                                            })
dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

size = len(dataloader.dataset)
num_batches = len(dataloader)
validation_loss, correct = 0, 0

batch_count = 0
with torch.no_grad():
    for img_1s, img_2s, ys in dataloader:
        img_1s.to(device)
        img_2s.to(device)
        ys.to(device)

        print(batch_count)
        batch_count += 1
        # Pass through Siamese network
        output = model((img_1s, img_2s))

        pred = torch.flatten(output)
        labels = ys.type(torch.float)

        validation_loss += torch.nn.functional.binary_cross_entropy_with_logits(pred, labels)

        pred = torch.nn.Sigmoid()(pred)
        pred = (pred > 0.5)
        correct += (pred == labels).sum().item()

validation_loss /= num_batches
correct /= size
print(f"Average validation loss: {validation_loss:>8f}, Accuracy: {(100*correct):>0.1f}% \n")