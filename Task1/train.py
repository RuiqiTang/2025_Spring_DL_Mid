import os 
import time
import torch 
from tqdm import tqdm
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader,val_loader,device,
                project_base='mid-term-tensorboard/Task-1',epochs=20,
                lr=1e-3,fine_tune_lr=1e-4):
    timestamp=time.time()
    tensorboard_dir=os.path.join(project_base,'output','tensorboard',f'ResNet_{timestamp}')
    if not os.path.exists(tensorboard_dir): # create tensorboard archieve
      os.makedirs(tensorboard_dir)
    
    writer = SummaryWriter(log_dir=tensorboard_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.fc.parameters()},  # 新的分类层，较大学习率
        {'params': [p for name, p in model.named_parameters() if "fc" not in name], 'lr': fine_tune_lr}
    ], lr=lr, momentum=0.9)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # Loss perserved in tensorboard
        #print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

        # 写入 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.close()
    print(f"Tensorboard saved in:{tensorboard_dir}")
    
    # save model file
    model_savebase=os.path.join(project_base,'output','model')
    model_fp=os.path.join(model_savebase,f'ResNet_model_{timestamp}.pth')
    torch.save(model.state_dict(),model_fp)
    print(f"Model saved in:{model_fp}")
    
    return