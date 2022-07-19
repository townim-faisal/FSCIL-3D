import numpy as np
import os, sys, time, copy, gc, argparse
import torch
from path import Path
import torchmetrics
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from configs.shapenet_co3d_info import nb_cl_fg, len_cls, model_heads
from src.data_utils.datautil_3D_memory_incremental_shapenet_co3d import *
from src.provider import *
from src.util import *
from src.without_w2v.pointnet import PointNetfeat
from src.knn import KMeans

"""# Matrix 1024x1024 collection for task 0"""

#################################################### Train and Test ############################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # "cuda:0" if torch.cuda.is_available() else "cpu"

"""
Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='./configs/ShapeNet_CO3D_incremental_config.yaml', help='path of config file')

input_arguments = parser.parse_args()

args = Argument(config_file = input_arguments.config_file)
args.num_samples = 0
args.batch_size = 1
print('Configurations:', args.__dict__)

path = Path(args.dataset_path)
# os.mkdir('./saved_models')
saved_model_path = Path('./saved_models/pointnet/').mkdir_p()

start_iter = 0

# seeding
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataloader = DatasetGen(args, root=path)
print(50*'*')
dataset = dataloader.get(0,'training')
trainloader = dataset[0]['train']
testloader = dataset[0]['test']
sem_train = dataset[0]['sem_train'].to(device).float()


class PointNetfeatV2(PointNetfeat):
    def __init__(self, global_feat = True, feature_transform = True):
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
        matrix1024x1024 = self.bn3(self.conv3(x)) # batch number x feature vector size x number of points
        x = torch.max(matrix1024x1024, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        # maxpooling for permutation invariance symmetric function
        # x = nn.MaxPool1d(matrix1024x1024.size(-1))(matrix1024x1024)
        x = nn.Flatten(1)(x)
        if self.global_feat:
            return x, trans, trans_feat, matrix1024x1024.transpose(2,1)
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, matrix1024x1024.transpose(2,1)

class PointNetWithoutw2v(nn.Module):
    def __init__(self, k=2, feature_transform=True):
        super(PointNetWithoutw2v, self).__init__()
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



class PointNetWithw2v(nn.Module):
    def __init__(self, feature_transform=True, att_size=300):
        super(PointNetWithw2v, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeatV2(global_feat=True, feature_transform=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, att_size)
        # self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(att_size)
        self.relu = nn.ReLU()

    def forward(self, x, old_att=None):
        x, trans, trans_feat, matrix1024x1024 = self.feat(x)
        # feat = x
        x = self.relu(self.bn1(self.fc1(x))) 
        x = self.relu(self.bn2(self.fc2(x))) # v3
        feat = x
        sem = self.relu(old_att)
        x = F.linear(x, sem)
        return F.log_softmax(x, dim=-1), feat, trans_feat, matrix1024x1024



"""
## without w2v
"""
model = PointNetWithoutw2v(len_cls[0], feature_transform=True)
model.load_state_dict(torch.load(os.path.join(saved_model_path, 'task_0_without_knn_without_w2v.pth')), strict=True)
model = model.to(device)

Path('./saved_models/pointnet/matrix_1024x1024/without_w2v').mkdir_p()
matrix_path = './saved_models/pointnet/matrix_1024x1024/without_w2v'

a = []
i = 0
k = 0

with torch.no_grad():
    model.eval()
    t = tqdm(enumerate(trainloader, 0), total=len(trainloader), ncols=100, smoothing=0.9, position=0, leave=True)
    for batch_id, data in t:
        t.set_postfix_str(f'Batch no = {batch_id+1}')
        inputs, labels = data['pointclouds'].to(device).float(), data['labels'].to(device)
        inputs = inputs.transpose(1,2)  #[bs,3,1024] 
        pred, _, _, matrix1024x1024 = model(inputs)
        # print(matrix1024x1024.size())
        # break
        if (batch_id+1) % 10 == 0:
            i+=1
            k += len(a)
            a = torch.cat(a, dim=1)
            a = a.reshape(a.shape[1],a.shape[2])
            print('mattrix1024x1024 size = '+str(a.shape))
            torch.save(a, matrix_path+'/mat_'+str(i)+'.pth')
            del a
            gc.collect()
            a = []
            a.append(matrix1024x1024)
            print('mattrix1024x1024 size = '+str(len(a)))
        else:
            a.append(matrix1024x1024)


if len(a) > 0:
    i+=1
    k += len(a)
    a = torch.cat(a, dim=1)
    a = a.reshape(a.shape[1],a.shape[2])
    print(a.size())
    torch.save(a, matrix_path+'/mat_'+str(i)+'.pth')
    del a
    gc.collect()
print(i, k)

# result = torch.cat(a, dim=1) 
# print(result.size())

k = 0 
min_a, max_a = [], []
for i in os.listdir(matrix_path):
  print(i)
  a = torch.load(matrix_path+f'/{i}')
  k+=a.shape[0]
  min_a.append(torch.min(a).item())
  max_a.append(torch.max(a).item())
  print(a.size(), torch.min(a).item(), torch.max(a).item())
  del a
  gc.collect()
print(k, min(min_a), max(max_a))



"""
## with w2v 
"""
model = PointNetWithw2v(att_size=300, feature_transform=True)
model.load_state_dict(torch.load(os.path.join(saved_model_path, 'task_0_without_knn_with_w2v.pth')), strict=True)
model = model.to(device)

Path('./saved_models/pointnet/matrix_1024x1024/with_w2v').mkdir_p()
matrix_path = './saved_models/pointnet/matrix_1024x1024/with_w2v'

a = []
i = 0
k = 0

with torch.no_grad():
    model.eval()
    t = tqdm(enumerate(trainloader, 0), total=len(trainloader), ncols=100, smoothing=0.9, position=0, leave=True)
    for batch_id, data in t:
        t.set_postfix_str(f'Batch no = {batch_id+1}')
        inputs, labels = data['pointclouds'].to(device).float(), data['labels'].to(device)
        inputs = inputs.transpose(1,2)  #[bs,3,1024] 
        pred, _, _, matrix1024x1024 = model(inputs, sem_train)
        # print(matrix1024x1024.size())
        # break
        if (batch_id+1) % 10 == 0:
            i+=1
            k += len(a)
            a = torch.cat(a, dim=1)
            a = a.reshape(a.shape[1],a.shape[2])
            print('mattrix1024x1024 size = '+str(a.shape))
            torch.save(a, matrix_path+'/mat_'+str(i)+'.pth')
            del a
            gc.collect()
            a = []
            a.append(matrix1024x1024)
            print('mattrix1024x1024 size = '+str(len(a)))
        else:
            a.append(matrix1024x1024)


if len(a) > 0:
    i+=1
    k += len(a)
    a = torch.cat(a, dim=1)
    a = a.reshape(a.shape[1],a.shape[2])
    print(a.size())
    torch.save(a, matrix_path+'/mat_'+str(i)+'.pth')
    del a
    gc.collect()
print(i, k)

# result = torch.cat(a, dim=1) 
# print(result.size())

k = 0 
min_a, max_a = [], []
for i in os.listdir(matrix_path):
  print(i)
  a = torch.load(matrix_path+f'/{i}')
  k+=a.shape[0]
  min_a.append(torch.min(a).item())
  max_a.append(torch.max(a).item())
  print(a.size(), torch.min(a).item(), torch.max(a).item())
  del a
  gc.collect()
print(k, min(min_a), max(max_a))


"""
# Centroids collection
"""

"""
Run these to install torchpq:

conda install -c conda-forge cupy
pip install torchpq
"""

from torchpq.clustering import MinibatchKMeans
import numpy as np
import os, sys, time, copy, gc, argparse
import torch
from datetime import datetime
from path import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # "cuda:0" if torch.cuda.is_available() else "cpu"

tol = 1e-3

for n_clusters in [3, 4, 5, 6, 256, 512, 1024, 2048, 4096]:
    """
    ## without w2v
    """
    Path('./saved_models/pointnet/centroids/without_w2v').mkdir_p()
    centroids_path = './saved_models/pointnet/centroids/without_w2v'
    matrix_path = './saved_models/pointnet/matrix_1024x1024/without_w2v'

    kmeans_euclid = MinibatchKMeans(n_clusters=n_clusters, distance="euclidean", init_mode="kmeans++")
    print('without w2v - euclid', n_clusters, ':', datetime.now().strftime("%H:%M:%S"))

    for i in os.listdir(matrix_path):
        a = torch.load(matrix_path+f'/{i}').to(device).float().transpose(0, 1) # n_features x n_data
        kmeans_euclid.fit_minibatch(a)
        del a
        gc.collect()
        # print('completed:', matrix_path+f'/{i}')
    
    print("Error:", kmeans_euclid.error)
    centroids = kmeans_euclid.centroids
    centroids = centroids.float().transpose(0, 1) # n_clusters x n_features
    print('without w2v - euclid:', centroids.size(), ':', datetime.now().strftime("%H:%M:%S"))
    torch.save(centroids, centroids_path+f'/{n_clusters}x1024_euclidean.pth')


    kmeans_cosine = MinibatchKMeans(n_clusters=n_clusters, distance="cosine", init_mode="kmeans++")
    print('without w2v - cosine',n_clusters, ':', datetime.now().strftime("%H:%M:%S"))
    
    for i in os.listdir(matrix_path):
        a = torch.load(matrix_path+f'/{i}').to(device).float().transpose(0, 1) # n_features x n_data
        kmeans_cosine.fit_minibatch(a)
        del a
        gc.collect()
        # print('completed:', matrix_path+f'/{i}')

    print("Error:", kmeans_cosine.error)
    centroids = kmeans_cosine.centroids
    centroids = centroids.float().transpose(0, 1) # n_clusters x n_features
    print('without w2v - cosine:', centroids.size(), ':', datetime.now().strftime("%H:%M:%S"))
    torch.save(centroids, centroids_path+f'/{n_clusters}x1024_cosine.pth')


    """
    ## with w2v
    """
    Path('./saved_models/pointnet/centroids/with_w2v').mkdir_p()
    centroids_path = './saved_models/pointnet/centroids/with_w2v'
    matrix_path = './saved_models/pointnet/matrix_1024x1024/with_w2v'

    kmeans_euclid = MinibatchKMeans(n_clusters=n_clusters, distance="euclidean", init_mode="kmeans++")
    print('with w2v - euclid', n_clusters, ':', datetime.now().strftime("%H:%M:%S"))
    
    for i in os.listdir(matrix_path):
        a = torch.load(matrix_path+f'/{i}').to(device).float().transpose(0, 1) # n_features x n_data
        kmeans_euclid.fit_minibatch(a)
        del a
        gc.collect()
        # print('completed:', matrix_path+f'/{i}')
        # print('completed:', matrix_path+f'/{i}')
    
    print("Error:", kmeans_euclid.error)
    centroids = kmeans_euclid.centroids
    centroids = centroids.float().transpose(0, 1) # n_clusters x n_features
    print('with w2v - euclid:', centroids.size(), ':', datetime.now().strftime("%H:%M:%S"))
    torch.save(centroids, centroids_path+f'/{n_clusters}x1024_euclidean.pth')

    kmeans_cosine = MinibatchKMeans(n_clusters=n_clusters, distance="cosine", init_mode="kmeans++")
    print('without w2v - cosine',n_clusters, ':', datetime.now().strftime("%H:%M:%S"))

    for i in os.listdir(matrix_path):
        a = torch.load(matrix_path+f'/{i}').to(device).float().transpose(0, 1) # n_features x n_data
        kmeans_cosine.fit_minibatch(a)
        del a
        gc.collect()
        # print('completed:', matrix_path+f'/{i}')

    print("Error:", kmeans_cosine.error)
    centroids = kmeans_cosine.centroids
    centroids = centroids.float().transpose(0, 1) # n_clusters x n_features
    print('with w2v - cosine:', centroids.size(), ':', datetime.now().strftime("%H:%M:%S"))
    torch.save(centroids, centroids_path+f'/{n_clusters}x1024_cosine.pth')