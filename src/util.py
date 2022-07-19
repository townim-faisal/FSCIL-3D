import torch, os, yaml

"""#Argument"""
class Argument(object):
    def __init__(self, config_file):
        super(Argument, self).__init__()    
        config_file = open(config_file, 'r') # args.config_path
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        for key in config:
            setattr(self, key, config[key])
        # return print('done')

"""#AverageMeter"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def svd_centroid_conversion(centroids):
    U, S, Vh = torch.linalg.svd(centroids.transpose(0,1), full_matrices=False)
    # print(U.shape, S.shape, Vh.shape)
    # print(S, float(0.95*torch.sum(S)))

    # print(U.transpose(0,1),centroids)
    ind = 0
    for i in range(centroids.shape[0]):
        if i==0:
            ind = i
            if float(torch.sum(S[0]))>float(0.95*torch.sum(S)):
                break
        else:
            if float(torch.sum(S[:i+1]))>float(0.95*torch.sum(S)):
                break
            else:
                ind = i
    # print(ind, U[:, :ind].shape)#, Vh)

    cent = U[:, :ind] @ torch.diag(S[:ind]) @ Vh[:ind, :]
    return cent.transpose(0,1)