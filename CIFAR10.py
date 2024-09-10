import torch
import torch.nn as nn
import torch.nn. functional as F
import torchvision
import torchvision.transforms as transforms
import time

# Device configuration - Use GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters 
num_epochs = 35  # Number of times the model will go through the entire dataset
learning_rate = 0.1  # Initial learning rate for the optimizer

# Image transformation for the training set
transform_train = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize the images
    transforms.RandomHorizontalFlip(),  # Augment data by randomly flipping images horizontally
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # Randomly crop the image with padding
])

# Image transformation for the testing set (no augmentation, only normalization)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2818)),
])

# Load the CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(
    root='cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)  # Batch size 128, shuffled

# Load the CIFAR-10 testing dataset
testset = torchvision.datasets.CIFAR10(
    root='cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


# Define a basic block of the ResNet
class BasicBlock(nn.Module):
    expansion = 1  # No expansion in this block (expansion factor = 1)
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # Batch normalization after convolution

        # Second convolutional layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # Batch normalization after convolution

        # Shortcut connection for residual learning
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:  # If input and output sizes don't match, use 1x1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # Apply the first convolutional layer and activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply the second convolutional layer
        out = self.bn2(self.conv2(out))
        # Add shortcut (skip connection)
        out += self.shortcut(x)
        # Apply ReLU to the sum
        out = F.relu(out)
        return out

# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):  # num_classes=10 for CIFAR-10
        super(ResNet, self).__init__()
        self.in_planes = 64  # Initial number of channels

        # Initial convolutional layer and batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Define the ResNet layers (4 layers in total)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # First layer (no downsampling)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # Second layer (downsampling)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # Third layer (downsampling)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # Fourth layer (downsampling)

        # Final fully connected layer
        self.linear = nn.Linear(512*block.expansion, num_classes)

    # Create a ResNet layer (with multiple blocks)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # First block may have stride > 1 for downsampling
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # Define the forward pass through the network
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Initial conv + ReLU
        out = self.layer1(out)  # Pass through first ResNet layer
        out = self.layer2(out)  # Pass through second ResNet layer
        out = self.layer3(out)  # Pass through third ResNet layer
        out = self.layer4(out)  # Pass through fourth ResNet layer
        out = F.avg_pool2d(out, 4)  # Average pooling
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.linear(out)  # Final classification layer
        return out
    
# Instantiate the ResNet model
def ResNetModel():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet architecture

model = ResNetModel().to(device)  # Send model to GPU/CPU

# If using CUDA, print the name of the GPU
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# Print the total number of model parameters
print("Model No. of Parameters: ", sum([param.nelement() for param in model.parameters()]))
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross entropy loss for classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # SGD with momentum

# Learning rate scheduler (Piecewise Linear)
total_step = len(train_loader)
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=learning_rate, step_size_up=15, 
                                                   step_size_down=15, mode="triangular", verbose=False)
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/learning_rate, end_factor=0.005/5, verbose=False)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])

# Train the model
model.train()  # Set the model to training mode
print("> Training")
start = time.time()  # Record start time

for epoch in range(num_epochs):  # Train for the set number of epochs
    for i, (images, labels) in enumerate(train_loader):  # Loop through batches of data
        images = images.to(device)  # Move images to device (GPU/CPU)
        labels = labels.to(device)  # Move labels to device

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)  # Compute loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        # Print loss every 100 batches
        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    scheduler.step()  # Step the learning rate scheduler

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Test the model 
print("> Testing")
start = time.time() #time generation
model.eval()
with torch.no_grad(): # Disable gradient calculation for testing
    correct=0
    total=0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Test Accuracy: {} %'.format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")