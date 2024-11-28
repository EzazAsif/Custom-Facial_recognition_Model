
import torch
import torch.nn as nn


from PIL import Image
import torchvision.transforms as transforms


import torch

import torch.nn.functional as F

# Instantiate model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Element-wise addition for residual connection
        return self.relu(out)







class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Additional convolution layers
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Adding two more convolutional layers
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # New conv layer 1
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # New conv layer 2
        self.bn4 = nn.BatchNorm2d(512)

        # Residual blocks with appropriate downsampling
        self.res_block1 = ResidualBlock(512, 512)
        self.res_block2 = ResidualBlock(512, 512)

        # Pooling layer after all convolutions (adjust pooling here for final spatial dimensions)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.dropout_fc1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.dropout_fc2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(1024, 256)  # Embedding size

    def forward_one(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)  # Pooling to reduce spatial dimensions gradually

        # Additional convolution layers
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)

        # New convolutional layers
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Pass through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Global average pooling to obtain fixed-size feature maps (4x4 output)
        x = self.global_avg_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout_fc2(x)

        x = self.fc3(x)  # Output embeddings
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


model = SiameseNetwork().to(device)
# Load the state dictionary (trained parameters) into the model
model.load_state_dict(torch.load("FR_model.pth"))

# Move the model to the appropriate device (CPU or GPU)
model.to(device)



def predict_similarity(siamese_network, image1, image2, threshold=0.5):

   # Define the transformations
    # Define the transformations
    transform = transforms.Compose([
                transforms.Resize((128, 128)),  # Resize images to a fixed size
                transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize grayscale images (adjust the values as needed)
])
    image1 = transform(image1).unsqueeze(0).to(device)  # Add batch dimension and move to device
    image2 = transform(image2).unsqueeze(0).to(device)

    # Ensure the model is in evaluation mode
    siamese_network.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Pass both images through the Siamese Network
        output1, output2 = siamese_network(image1, image2)
        
        # Calculate the Euclidean distance between the embeddings
        distance =torch.nn.functional.pairwise_distance(output1, output2)
        
        # Check if the distance is less than the threshold
        if distance.item() < threshold:
            return True  # Images are of the same person
        else:
            return False  # Images are of different people


# Load images individually
image1 = Image.open('rafid1.jpg').convert('RGB')
image2 = Image.open('rafid2.jpg').convert('RGB')
image3 = Image.open('rathri1.jpg').convert('RGB')
image4 = Image.open('rathri2.jpg').convert('RGB')


#should be true
print(predict_similarity(model, image1, image2, threshold=0.5))
print(predict_similarity(model, image3, image4, threshold=0.5))
print("\n")
#should be False
print(predict_similarity(model, image1, image4, threshold=0.5))
print(predict_similarity(model, image2, image3, threshold=0.5))
print(predict_similarity(model, image1, image3, threshold=0.5))
print(predict_similarity(model, image2, image4, threshold=0.5))
