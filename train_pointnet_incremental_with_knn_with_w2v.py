import numpy as np
import os, sys, time, copy, gc, argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from path import Path
import torchmetrics
import torch.nn as nn
from tqdm import tqdm

from src.data_utils.datautil_3D_memory_incremental_shapenet_co3d import *
from configs.shapenet_co3d_info import nb_cl_fg, len_cls, model_heads
from src.provider import *
from src.util import *


#################################################### TConfigurations ############################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='./configs/ShapeNet_CO3D_incremental_config.yaml', help='path of config file')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')

input_arguments = parser.parse_args()

args = Argument(config_file = input_arguments.config_file)
print('Configurations:', args.__dict__)

path=Path(args.dataset_path)

start_iter = 0
amsgrad = True
eps = 1e-8


np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataloader = DatasetGen(args, root=path, fewshot=args.fewshot)

acc_tasks = []
feat_class = {} # feature for each class



print("cluster 300".upper())
from with_w2v.pointnet_knn import PointNetCls300 as PointNetCls, PointNet300 as FeatureExtractor
load_model_path = './saved_models/task_0_cluster_with_w2v.pth' # 'task_0_cluster.pth', 'task_0_without_knn.pth'


print('Load feature extractor model:', load_model_path)


centroids = torch.load('./saved_models/centroids/with_w2v/1024x1024_'+args.sim+'.pth')
centroids = centroids.to(device).float()
centroids = svd_centroid_conversion(centroids)

# feature extractor model for prototype building
feature_extractor_model = FeatureExtractor(len_cls[0], feature_transform=True, centroids=centroids, sim=args.sim).to(device)
feature_extractor_model.load_state_dict(torch.load(load_model_path), strict=True)
feature_extractor_model = feature_extractor_model.to(device)
for name, param in feature_extractor_model.named_parameters():
    param.requires_grad_(False)
# del feature_extractor_model.sem
# print(feature_extractor_model)
# feature_extractor_model = feature_extractor_model.feat
feature_extractor_model.eval()
print("protype building model is loaded..........")


#################################################### Functions ############################################
def count_parameters(model):
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return print('num_param',params/1000000)


class RelationLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, num_class=20, loss_type='mse'):
        super(RelationLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.n = num_class
        self.loss_type= loss_type
        self.refresh_meter()

    def refresh_meter(self):
        self.ce_loss_meter = AverageMeter()

    def feature_transform_regularizer(self, trans):
        d = trans.size()[1]
        # batchsize = trans.size()[0]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
        return loss


    def forward(self, pred, target, trans_feat):
        one_hot_labels = F.one_hot(target, num_classes=self.n)
        ce_loss = F.binary_cross_entropy(pred, one_hot_labels.float())
        mat_diff_loss = self.feature_transform_regularizer(trans_feat)
        total_loss = mat_diff_loss*self.mat_diff_loss_scale #+ v2_loss#+ args.lamda1*(dist_ps+dist_fs)
        total_loss+=ce_loss
        self.ce_loss_meter.update(ce_loss)
        return total_loss


def train(epochs, model_student, model_teacher, student_optimizer, train_loader, test_loader, task_num, args, sem_train, sem_test) : 
    #######################   Train  
    loss_total = AverageMeter()
    loss_cls= AverageMeter()
    accuracy = torchmetrics.Accuracy().cuda()
    best_model = None
    
    if task_num>=1:
        model_teacher.eval()
        num_old_classes = len_cls[task_num-1]
        

    # build prototype vector for each step
    with torch.no_grad():
        f = 0
        sem = None
        for batch_id, data in enumerate(train_loader):
            inputs, labels = data['pointclouds'].to(device).float(), data['labels'].to(device)
            inputs = inputs.transpose(1,2)  #[bs,3,1024] 
            _, feat, _ = feature_extractor_model(inputs)
            f = feat.shape[1]
            for i, x in enumerate(labels):
                if x.item() in feat_class:
                    if task_num>=1 and args.num_samples>0:
                        if x.item()<num_old_classes:
                            continue
                    else:
                        feat_class[x.item()].append(feat[i])
                else:
                    feat_class[x.item()] = []
                    feat_class[x.item()].append(feat[i])
        # print(feat_class.keys())

        protype_vector = []
        for i in range(len_cls[task_num]):
            if task_num == 0:
                n = len(feat_class[i])
                feat_class[i] = torch.cat(feat_class[i]).reshape(n,f).mean(0, True)
                protype_vector.append(feat_class[i])
            else:
                if i>=num_old_classes:
                    n = len(feat_class[i])
                    feat_class[i] = torch.cat(feat_class[i]).reshape(n,f).mean(0, True)
                protype_vector.append(feat_class[i])
        protype_vector = torch.cat(protype_vector, 0)
        print('Feature vector shape:',protype_vector.shape)
    
    # sys.exit()
    # TRAINING
    best_acc = 0
    for epoch in range(epochs):
        model_student.train()
        t = tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9, position=0, leave=True, desc="Train: Epoch: "+str(epoch+1))
        for batch_id, data in t:
            inputs, labels = data['pointclouds'].to(device).float(), data['labels'].to(device)
            inputs = inputs.cpu().data.numpy()
            inputs = random_point_dropout(inputs)
            inputs[:,:, 0:3] = random_scale_point_cloud(inputs[:,:, 0:3])
            inputs[:,:, 0:3] = shift_point_cloud(inputs[:,:, 0:3])
            inputs = torch.Tensor(inputs).to(device).float()
            inputs = inputs.transpose(1,2)
            student_optimizer.zero_grad()    
            outputs = model_student(inputs, sem_train)
            outputs_student, trans_feat = outputs['pred'], outputs['trans_feat']
            output_feats = outputs['feats']
            batch_protype_vector = []
            sem = F.relu(sem_test)
            batch_sem = []
            for i in labels:
                batch_protype_vector.append(protype_vector[i].reshape(1,protype_vector.shape[1]))
                batch_sem.append(sem[i].reshape(1,sem.shape[1]))
            batch_protype_vector = torch.cat(batch_protype_vector, 0)  
            batch_sem = torch.cat(batch_sem, 0)
            loss_classification = classification_loss(outputs_student, labels, trans_feat)
            loss = loss_classification
            loss.backward()
            student_optimizer.step()
            loss_total.update(loss)
            loss_cls.update(loss_classification)
            accuracy(outputs_student.softmax(dim=-1), labels)  
        
            acc = 100*accuracy.compute()
            
            t.set_postfix_str(f'Tot_Loss: {loss_total.avg:.4} ce: {classification_loss.ce_loss_meter.avg:.4} Acc: {acc:.4}') 
            classification_loss.refresh_meter()
            
        
        ########################## Test
        model_student.eval() 
        accuracy = torchmetrics.Accuracy().cuda()
        correct_task = [0 for i in range(args.ntasks)]
        num = [0 for i in range(args.ntasks)]

        with torch.no_grad():
            total_correct = 0
            total_testset = 0
            for batch_id, data in enumerate(test_loader):
                inputs, labels, task_label = data['pointclouds'].to(device).float(), data['labels'].to(device), data['task_label']
                inputs = inputs.transpose(1,2)  #[bs,3,1024] 
                outputs = model_student(inputs, sem_test)
                output = outputs['pred']
                _, pred = output.max(1)
                correct = pred.eq(labels).cpu().sum()
                total_correct += correct.item()
                total_testset += inputs.size()[0]
                accuracy(output.softmax(dim=-1), labels)
                for j in range(len(labels)):   
                    num[task_label[j]] +=1
                    if pred[j]==labels[j]:
                        correct_task[task_label[j]] +=1
                                           
            acc_other = round(100*total_correct/float(total_testset), 1)
            print(f'Test: Epoch:{epoch+1} Acc: {acc_other}')
            
            
            if acc_other>=best_acc and epoch+1>=30:
                best_acc=acc_other
                print('save weights_task{}'.format(task_num))
                best_model=model_student
                # torch.save(model_student.state_dict(), os.path.join(saved_model_path, 'weight_task{}.pth'.format(task_num))) 
                
            print(5*'*') 

            # evaluation by euclid distance
            if epoch+1==epochs:
                if args.num_samples>0:
                    acc_tasks.append(round(best_acc, 1))
                else:
                    acc_tasks.append(round(best_acc, 1))
                return best_model

    # if args.num_samples>0:
    #     model_student.load_state_dict(torch.load(os.path.join(saved_model_path, 'weight_task{}.pth'.format(task_num))), strict=True)
    return model_student


#################################################### Train and Test ############################################

for t in range(0, args.ntasks):
    print(50*'*')
    dataset = dataloader.get(t,'training')
    trainloader=dataset[t]['train']
    testloader=dataset[t]['test']
    # semantics
    sem_train = dataset[t]['sem_train'].to(device).float()
    sem_test = dataset[t]['sem_test'].to(device).float()
    
    if t==0:
        student_model=PointNetCls(len_cls[t], feature_transform=True, centroids=centroids, sim=args.sim).to(device)
        student_model.load_state_dict(torch.load(load_model_path), strict=False)
        teacher_model=None
        epochs=50 # 20
        base_lr=args.lr
        wd=args.wd
    else:
        teacher_model=copy.deepcopy(student_model)
        epochs=40 #10 #60
        base_lr=(1e-4+1e-5)/2
        wd=(1e-6+1e-4)/2


    student_model = student_model.to(device)
    if t>0:
        print("Layers of backbone are freezed..............")
        teacher_model = teacher_model.to(device)
        ##############################################if you want freez feature extractor after task 0

        for name, param in student_model.named_parameters():
            if "feat" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
            # if t==1:
            #     print(name, param.requires_grad) 
        count_parameters(student_model)
        print("Freezed backbone after task 0")    
        #################################
    
    student_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=float(base_lr), betas=(0.9, 0.999), 
                                        eps=eps, amsgrad=amsgrad, weight_decay=float(wd))
    
    scheduler = torch.optim.lr_scheduler.StepLR(student_optimizer, step_size=20, gamma=0.5)
    
    classification_loss = RelationLoss(mat_diff_loss_scale=0.001, num_class=len_cls[t], loss_type=args.loss).to(device)
    
    student_model = train(epochs, student_model, teacher_model, student_optimizer, trainloader, testloader, t, args, sem_train, sem_test)
    
    print("All task's accuracy:",acc_tasks, "Average:", round(np.mean(acc_tasks),1))

print( "RPD:", round(100*(acc_tasks[0]-acc_tasks[-1])/acc_tasks[0], 1) )
torch.save(student_model.state_dict(), './with_knn_with_w2v_last_task.pt') 