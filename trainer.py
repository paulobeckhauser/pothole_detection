import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


def trainer(network, optimizer ,train_loader, val_loader, num_epochs=10, lr=0.001, device='cpu'):
    
    out_dict = {'train_acc': [],
              'val_acc': [],
              'train_loss': [],
              'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        network.train()
        train_loss = 0
        train_correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            loss = network.loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs > 0.5).eq(labels > 0.5).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        out_dict['train_acc'].append(train_acc)
        out_dict['train_loss'].append(train_loss)
        
        # Validation
        network.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = network(images)
                loss = network.loss(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs > 0.5).eq(labels > 0.5).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        out_dict['val_acc'].append(val_acc)
        out_dict['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return out_dict


def plot_loss_accuracy(out_dict):
    # make some pretty plots

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(out_dict['train_loss'], label='Train Loss')
    ax[0].plot(out_dict['val_loss'], label='Val Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].title.set_text('Loss')
    ax[0].legend()
    
    ax[1].plot(out_dict['train_acc'], label='Train Acc')
    ax[1].plot(out_dict['val_acc'], label='Val Acc')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].title.set_text('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('results/loss_accuracy.png')
    plt.show()