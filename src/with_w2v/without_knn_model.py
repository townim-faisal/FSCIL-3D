
import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
from src.with_w2v.pointnet import PointNetfeat

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


class BaselineModel(nn.Module):
    def __init__(self, feature_transform=True):
        super(BaselineModel, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
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

    def forward(self, input, protype_features):
        outputs = {} 
        x, trans, trans_feat = self.feat(input)
        x = self.relu(self.bn1(self.fc1(x))) 
        x = self.relu(self.bn2(self.fc2(x)))
        outputs['feats'] = x
        outputs['trans_feat'] = trans_feat
        
        b, n = x.shape[0], protype_features.shape[0]
        feat_dim = x.shape[1] + protype_features.shape[1]

        sample_features_ext = protype_features.unsqueeze(0).repeat(b,1,1)
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


