import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

lb = joblib.load('lb.pkl')

class ASTCNN(nn.Module):
    def __init__(self):
        super(ASTCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, len(lb.classes_))
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ASTCNNGray(nn.Module):
    def __init__(self):
        super(ASTCNNGray, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, len(lb.classes_))
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ConvolutionalBlock(nn.Module):
    def __init__(self):
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, s):
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=64, planes=64, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(64, 3, kernel_size=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*32*32, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, len(lb.classes_))
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) 
        v = v.view(-1, 3*32*32)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.relu(self.fc2(v))
        v = F.relu(self.fc3(v))
        v = self.fc4(v)
        return v
        
class ASTResNet(nn.Module):
    def __init__(self):
        super(ASTResNet, self).__init__()
        self.conv = ConvolutionalBlock()
        self.res_block_1 = ResBlock()
        self.res_block_2 = ResBlock()
        self.res_block_3 = ResBlock()
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.pool_3 = nn.MaxPool2d(2, 2)
        self.outblock = OutBlock()
        
    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        x = self.res_block_1(x)
        x = self.pool_1(x)
        print(x.shape)
        
        x = self.res_block_2(x)
        x = self.pool_2(x)
        print(x.shape)
        
        x = self.res_block_3(x)
        x = self.pool_3(x)
        print(x.shape)
        
        x = self.outblock(x)
        return x
        
         
if __name__ == '__main__':
    model = ASTResNet()
    print(model)
    
    test = torch.randn(3, 1, 256, 256)
    result = model(test)
    
    print(result)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))