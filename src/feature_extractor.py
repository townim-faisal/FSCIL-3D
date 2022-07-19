from src.pointnet import PointNetfeat
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetfeatV2(PointNetfeat):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeatV2, self).__init__(global_feat, feature_transform)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        matrix1024x1024 = self.bn3(self.conv3(x))
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        # maxpooling for permutation invariance symmetric function
        x = nn.MaxPool1d(matrix1024x1024.size(-1))(matrix1024x1024)
        x = nn.Flatten(1)(x)
        if self.global_feat:
            return x, trans, trans_feat, matrix1024x1024
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, matrix1024x1024

class FeatureExtractor(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(FeatureExtractor, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeatV2(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat, matrix1024x1024  = self.feat(x)
        feat = x
        x = F.relu(self.bn1(self.fc1(x))) 
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # feat = x
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), feat, trans_feat, matrix1024x1024