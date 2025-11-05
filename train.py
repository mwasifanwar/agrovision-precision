import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml

from agrovision_precision.core.disease_detector import PlantDiseaseCNN
from agrovision_precision.data.data_augmentation import DataAugmentation

def main():
    parser = argparse.ArgumentParser(description='Train AgroVision Precision Models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Initializing AgroVision Precision Training...")
    
    print("Training disease detection model...")
    train_disease_model(config, args.epochs, args.batch_size)
    
    print("Training completed successfully!")

def train_disease_model(config, epochs, batch_size):
    model = PlantDiseaseCNN(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Disease Detection Model Architecture:")
    print(model)
    
    augmentation = DataAugmentation()
    
    train_losses = []
    for epoch in range(epochs):
        model.train()
        
        epoch_loss = 0.0
        for batch_idx in range(100):
            synthetic_images = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
            synthetic_labels = torch.randint(0, 10, (batch_size,))
            
            augmented_images, _ = augmentation.augment_batch(synthetic_images)
            
            images_tensor = torch.tensor(augmented_images)
            
            optimizer.zero_grad()
            outputs = model(images_tensor)
            loss = criterion(outputs, synthetic_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 100
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'models/disease_detector.pth')
    print("Disease detection model saved to models/disease_detector.pth")

if __name__ == "__main__":
    main()