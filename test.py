import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import glob
from dev import ENet

# -------------------------
# DATASET DEFINITION
# -------------------------
class TusimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = []  # <-- Initialize the list to store all annotations

        # Load all JSON annotation files
        annotation_files = glob.glob(os.path.join(root_dir, "*.json"))
        for file in annotation_files:
            with open(file, 'r') as f:
                for line in f:
                    self.annotations.append(json.loads(line))  # Properly load each line
        
        print(annotation_files)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.root_dir, annotation['raw_file'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a blank mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Draw lane lines on the mask
        for lane in annotation['lanes']:
            for x, y in zip(lane, annotation['h_samples']):
                if x != -2:  # Ignore invalid points
                    cv2.circle(mask, (x, y), radius=5, color=255, thickness=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask



# -------------------------
# MODEL DEFINITION (UNET)
# -------------------------
class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
        )
        self.decoder = nn.Sequential(
            self.conv_block(256, 128),
            self.conv_block(128, 64),
            self.conv_block(64, num_classes, final=True),
        )

    def conv_block(self, in_channels, out_channels, final=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        if not final:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

# -------------------------
# TRAINING LOOP
# -------------------------
def train_model():
    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10
    root_dir = 'C:\\Users\\prote\\.cache\\kagglehub\\datasets\\manideep1108\\tusimple\\versions\\5\\TUSimple\\train_set'

    # Define Albumentations transformations
    train_transforms = A.Compose([
        A.Resize(256, 256),  # Resize both image and mask to the same size
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()  # Convert to PyTorch tensors
    ])

    # Load dataset
    train_dataset = TusimpleDataset(root_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = UNet(num_classes=1)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)  # Move data to the chosen device

            # Add the channel dimension to masks
            masks = masks.unsqueeze(1).float()  # Convert masks to float (from [8, 256, 256] -> [8, 1, 256, 256])

            # Forward pass
            outputs = model(images)

            # Upsample outputs to match the mask size
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            # Compute the loss
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")



    # Save the model
    torch.save(model.state_dict(), 'lane_detection_model.pth')
    print("Model saved!")

# -------------------------
# TESTING / INFERENCE
# -------------------------

def region(image):
    """ Applies a region mask to isolate lanes. """
    h, w = image.shape[:2]  # Get height and width

    # Define a polygonal region (Adjust coordinates as needed)
    triangle = np.array([ 
        [(0, h), (w//2 - 100, h//2), (w//2 + 100, h//2), (w, h)]
    ], dtype=np.int32)

    # Create a blank mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill the polygon (ROI) with white (255)
    cv2.fillPoly(mask, [triangle], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def test_model():
    """ Loads the model and runs lane detection on a test image. """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ENet(2, 4).to(device)
    model.load_state_dict(torch.load("adam_v0.1a.pth", map_location=device))
    model.eval()

    # Load and preprocess a test image
    test_image_path = 'C:/Users/prote/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/clips/0313-1/120/20.jpg'
    image = cv2.imread(test_image_path)
    original_image = image.copy()

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the region function
    image = cv2.resize(image, (512, 256))
    masked_image = region(image)

    # Debugging: Show the masked image before feeding it to the model
    plt.figure(figsize=(10, 5))
    plt.imshow(masked_image, cmap='gray')
    plt.title("Masked Image (Region of Interest)")
    plt.show()

    # Normalize and convert to PyTorch tensor
    input_tensor = torch.from_numpy(masked_image).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # Run inference
    with torch.no_grad():
        binary_output, instance_output = model(input_tensor)

    # Apply sigmoid activation to get probability values
    binary_mask = torch.sigmoid(binary_output)

    # Ensure binary_mask is still a tensor before applying .cpu()
    if not isinstance(binary_mask, torch.Tensor):
        binary_mask = torch.tensor(binary_mask)

    # Set a higher threshold for stricter lane detection
    dynamic_threshold = binary_mask[0, 0].mean().item()  # Increase margin
    dynamic_threshold = min(dynamic_threshold, 0.7)  # Cap at 0.7

    print(f"Dynamic Threshold (with margin): {dynamic_threshold}")

    # Apply threshold and convert to NumPy array
    binary_mask = (binary_mask > dynamic_threshold).float()  # Convert to binary mask
    binary_mask = binary_mask.cpu().numpy()  # Convert tensor to NumPy

    # Convert binary mask to 0-255 for visualization
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Display the binary mask
    plt.figure(figsize=(5, 5))
    plt.imshow(binary_mask[0, 0], cmap='gray')  # Display the binary mask
    plt.title("Binary Mask")
    plt.show()

    # Overlay detected lane lines on the original image
    lanes_overlay = cv2.applyColorMap(binary_mask[0, 0], cv2.COLORMAP_JET)  # Use the first (and only) channel

    # Resize the overlay to match the original image size
    lanes_overlay_resized = cv2.resize(lanes_overlay, (original_image.shape[1], original_image.shape[0]))

    # Blend the images
    result_image = cv2.addWeighted(original_image, 0.7, lanes_overlay_resized, 0.3, 0)

    # Display the final result
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Lane Detection Result")
    plt.show()

# -------------------------
# MAIN FUNCTION
# -------------------------
if __name__ == "__main__":
    mode = input("Enter 'train' to train or 'test' to test: ").strip().lower()

    if mode == 'train':
        train_model()
    elif mode == 'test':
        test_model()
    else:
        print("Invalid mode! Please enter 'train' or 'test'.")
