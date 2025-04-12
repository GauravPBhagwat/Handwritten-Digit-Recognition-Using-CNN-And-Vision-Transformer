import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from vit_mnist import ViT
import time
import matplotlib.pyplot as plt
import numpy as np

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64  # Reduced batch size
    learning_rate = 1e-3  # Increased learning rate
    num_epochs = 50
    patience = 5  # Early stopping patience

    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                 transform=train_transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                transform=test_transform)

    # Split training data into train and validation sets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)

    # Model
    print("Initializing model...")
    model = ViT(
        image_size=28,
        patch_size=4,  # Smaller patches
        num_classes=10,
        dim=256,       # Increased dimension
        depth=8,       # Increased depth
        heads=8,
        mlp_dim=512,   # Increased MLP dimension
        channels=1,
        dim_head=32,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': []
    }

    # Training loop
    print("Starting training...")
    best_accuracy = 0
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print progress
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')

            train_accuracy = 100 * correct / total
            train_loss = total_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader)

            # Test phase
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_accuracy = 100 * test_correct / test_total

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            history['test_acc'].append(test_accuracy)

            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print(f'Test Acc: {test_accuracy:.2f}%')

            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'history': history
                }, 'vit_mnist_best.pth')
                print(f'New best accuracy: {best_accuracy:.2f}%')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        total_time = time.time() - start_time
        print(f'\nTraining finished!')
        print(f'Total training time: {total_time/60:.2f} minutes')
        print(f'Best Test Accuracy: {best_accuracy:.2f}%')

        # Plot training history
        plt.figure(figsize=(12, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.plot(history['test_acc'], label='Test Acc')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

if __name__ == '__main__':
    train_model()