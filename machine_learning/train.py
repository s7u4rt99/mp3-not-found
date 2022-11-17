import pandas as pd
import numpy as np
import torch
import albumentations
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import cv2
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from model import ASTCNN, ASTCNNGray, ASTResNet


lb = joblib.load('lb.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv("ast_dataset/data_final.csv")
X = df['image_path'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class ASLImageDataset(Dataset):
    def __init__(self, path, labels):
        self.X = path
        self.y = labels
        
        self.aug = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
        ])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.X[idx])
        image = self.aug(image=np.array(image))['image']
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float).permute(0, 3, 1, 2)
        labels = torch.tensor(self.y[idx], dtype=torch.long)
        return image, labels
    
class ASLGrayImageDataset(Dataset):
    def __init__(self, path, labels):
        self.X = path
        self.y = labels
        
        self.aug = albumentations.Compose([
            albumentations.Resize(256, 256, always_apply=True),
        ])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.X[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.aug(image=np.array(image))['image']
        image = torch.tensor(image, dtype=torch.float)
        image = torch.unsqueeze(image, 0)
        image /= 255
        labels = torch.tensor(self.y[idx], dtype=torch.long)
        return image, labels
    
train_dataset = ASLGrayImageDataset(X_train, y_train)
validation_dataset = ASLGrayImageDataset(X_test, y_test)
full_dataset = ASLGrayImageDataset(X, y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
full_dataset_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

model = ASTResNet().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criteria = nn.CrossEntropyLoss()

def train(model, data_loader, optimizer, criteria, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for idx, (X, y) in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Training", total=len(data_loader)):
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(X)
        
        loss = criteria(y_pred, y)
        epoch_loss += loss.item()
        
        # Append Statistics
        _, y_pred = torch.max(y_pred, 1)
        epoch_acc += (y_pred == y).sum().item()
        
        loss.backward()
        optimizer.step()
    
    train_loss = epoch_loss / len(data_loader.dataset)
    train_acc = epoch_acc / len(data_loader.dataset)

    print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")

    return train_loss, train_acc

def validate(model, data_loader, criteria, epoch):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for idx, (X, y) in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Validation", total=len(data_loader)):
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            loss = criteria(y_pred, y)
            epoch_loss += loss.item()

            # Append Statistics
            _, y_pred = torch.max(y_pred, 1)
            epoch_acc += (y_pred == y).sum().item()

    val_loss = epoch_loss / len(data_loader.dataset)
    val_acc = epoch_acc / len(data_loader.dataset)

    print(f"Epoch {epoch} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    return val_loss, val_acc

train_losses = []
train_accs = []

val_losses = []
val_accs = []

NUM_EPOCHS = 12

count = 0
prev_val_loss = 0

for epoch in range(1, NUM_EPOCHS+1):
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    
    train_loss, train_acc = train(model, train_loader, optimizer, criteria, epoch)
    val_loss, val_acc = validate(model, test_loader, criteria, epoch)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    if (prev_val_loss - val_loss) < 0.001:
        count += 1
    else:
        count = 0
    
    prev_val_loss = val_loss
        
    if count == 3:
        break
    
    
print('Saving model...')
torch.save(model.state_dict(), 'model_resnet.pth')

plt.figure(figsize=(10, 7))
plt.plot(train_accs, color='green', label='train accuracy')
plt.plot(val_accs, color='blue', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
 
# loss plots

plt.figure(figsize=(10, 7))
plt.plot(train_losses, color='orange', label='train loss')
plt.plot(val_losses, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

# Plot confusion matrix
model.eval()

# Get all predictions in an array and all targets in an array
y_preds = np.array([])
targets = np.array([])

for X, y in full_dataset_loader:
    X = X.to(device)
    y = y.to(device)
    
    y_pred = model(X)
    _, y_pred = torch.max(y_pred, 1)
    
    y_pred = y_pred.cpu().numpy()
    
    y_preds.append(y_pred)
    targets.append(y)

y_preds = np.concatenate(y_preds)
targets = np.concatenate(targets)

cf_matrix = confusion_matrix(targets, y_preds)

df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*len(lb.classes_), index = [i for i in lb.classes_],
                     columns = [i for i in lb.classes_])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')
    
    

