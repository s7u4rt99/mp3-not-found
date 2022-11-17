import torch
import joblib
import numpy as np
import cv2
from model import ASTCNN, ASTCNNGray, ASTResNet
import albumentations
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


 
# # load label binarizer
# GRAY = True
# RESNET = False

# lb = joblib.load('lb.pkl')
# model = ASTResNet().cuda()
# model.load_state_dict(torch.load('model_resnet.pth', map_location=torch.device('cpu')))
# print(model)
# print('Model loaded')


# def hand_area(img):
#     hand = img[100:324, 100:324]
#     hand = cv2.resize(hand, (256,256))
#     return hand

# cap = cv2.VideoCapture(0)

# if (cap.isOpened() == False):
#     print('Error while trying to open camera. Plese check again...')
# # get the frame width and height
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# while(cap.isOpened()):
#     # capture each frame of the video
#     ret, frame = cap.read()
#     # get the hand area on the video capture screen
#     cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
#     hand = hand_area(frame)
#     image = hand
    
#     if GRAY:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif RESNET:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         combined = np.concatenate((image, gray[..., None]), axis=-1)     
#         image = combined.transpose((2, 0, 1))   
#     else:
#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
#     image = torch.tensor(image, dtype=torch.float).cuda()
#     image = image/255
#     image = image.unsqueeze(0)
#     image = image.unsqueeze(0)
    
#     outputs = model(image)
#     _, preds = torch.max(outputs.data, 1)
#     print(preds)
    
#     cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#     cv2.imshow('image', frame)
#     # press `q` to exit
#     if cv2.waitKey(27) & 0xFF == ord('q'):
#         break
# # release VideoCapture()
# cap.release()
# # close all frames and video windows
# cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv("ast_dataset/data_final.csv")
    X = df['image_path'].values
    y = df['label'].values
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
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
    
    
    full_dataset = ASLGrayImageDataset(X, y)
    full_dataset_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)
    
    
    lb = joblib.load('lb.pkl')
    model = ASTResNet().cuda()
    model.load_state_dict(torch.load('model_resnet.pth', map_location=torch.device('cpu')))
    print(model)
    print('Model loaded')
    
        # Plot confusion matrix
    model.eval()

    # Get all predictions in an array and all targets in an array
    y_preds = []
    targets = []

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
    