import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def train_defect_detector(data_dir='./dataset', epochs=5):
    print("Loading Pretrained ResNet18...")
    
    # 1. Setup Image Transformations (Resize and normalize for the neural network)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load  custom dataset
    dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Load a pretrained model and modify the final layer for 2 classes (Pass/Fail)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 output classes
    
    # 4. Setup training logic
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 5. Train the model
    print(f"Found {len(dataset)} images. Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")
        
    print("Training complete! Model saved as 'defect_model.pth'")
    torch.save(model.state_dict(), 'defect_model.pth')

if __name__ == '__main__':
    # You MUST have a folder named 'dataset' with 'pass' and 'fail' subfolders!
    train_defect_detector()