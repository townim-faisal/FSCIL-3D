from src.with_w2v.pointnet import STN3d, STNkd, PointnetLoss
from src.knn import *
import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, centroids=None, sim=None):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        
        if centroids is not None and sim is not None:
            self.bn4 = nn.BatchNorm1d(centroids.shape[0])
            self.kmeans = KMeans(n_clusters=centroids.shape[0], mode=sim, verbose=1, max_iter=1000000, centroids = centroids)

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
        x = self.bn3(self.conv3(x))
        # print(self.kmeans.predict(x.transpose(2,1)), self.kmeans.predict(x.transpose(2,1))[0].shape)
        # sys.exit()
        x = self.kmeans.cluster_sim(x.transpose(2,1)) # batch_size x n_samples x n_clusters
        x = x.transpose(2,1) # batch_size x n_clusters x n_samples
        # x = self.bn4(x)
        # maxpooling for permutation invariance symmetric function
        # x = torch.max(x, 2, keepdim=True)[0] # nn.MaxPool1d(x.size(-1))(x) # torch.max(x, 2, keepdim=True)[0]
        x = nn.AvgPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        # x = self.bn4(x)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias != None:
            m.bias.data.fill_(0)

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')


class PointNet300(nn.Module):
    def __init__(self, k=2, feature_transform=True, centroids=None, sim=None):
        super(PointNet300, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, centroids=centroids, sim=sim)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 300)
        self.fc3 = nn.Linear(300, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(300)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x))) 
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        feat = x
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), feat, trans_feat



class PointNetCls300(nn.Module):
    def __init__(self, k=2, feature_transform=True, centroids=None, sim=None):
        super(PointNetCls300, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, centroids=centroids, sim=sim)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 300)
        # self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(300)
        self.relu = nn.ReLU() # LeakyReLU, ReLU
        # self.rn = RelationNetwork(in_dim=self.fc2.out_features*2)
        self.rn = nn.Sequential(
            nn.Linear(self.fc2.out_features*2, 300),
            nn.LeakyReLU(), 
            nn.Linear(300, self.fc2.out_features*2),
            nn.LeakyReLU(), 
            nn.Linear(self.fc2.out_features*2, 1),
            nn.Sigmoid()
        )
        self.rn.apply(init_weights)

    def forward(self, input, attribute):
        outputs = {} 
        x, trans, trans_feat = self.feat(input)
        x = self.relu(self.bn1(self.fc1(x))) 
        x = self.relu(self.bn2(self.fc2(x)))
        outputs['feats'] = x
        outputs['trans_feat'] = trans_feat
        
        b, n = x.shape[0], attribute.shape[0]
        feat_dim = x.shape[1] + attribute.shape[1]

        sample_features_ext = attribute.unsqueeze(0).repeat(b,1,1)
        # print(sample_features_ext.shape) # torch.Size([1, 3, 2048])

        batch_features_ext = x.unsqueeze(0).repeat(n,1,1)
        # print(batch_features_ext.shape) # torch.Size([3, 1, 2048])
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        # print(batch_features_ext.shape) # torch.Size([1, 3, 2048])
        
        # concat att->features + actual_feature
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext),2)
        # print(relation_pairs.shape) # torch.Size([1, 3, 4096])
        relation_pairs = relation_pairs.view(-1, feat_dim)
        # print(relation_pairs.shape) # torch.Size([3, 4096])
        
        # get relation score
        relations = self.rn(relation_pairs)
        # print(relations.shape) # torch.Size([3, 1])
        relations = relations.view(-1,n)

        outputs['pred'] = relations
        return outputs