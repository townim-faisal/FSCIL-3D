# for training
nb_cl_fg = 39 # len(shapenet_label_selected) 
len_cls = [39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89] 
model_heads = [39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

# applicable for the dataset I made
task_ids_total=[] 
k = 0
for i in len_cls:
    task_ids_total.append([j for j in range(k, i)])
    k = i

# print(task_ids_total)
shapenet_label_not_selected = [2,6,7,10,11,17,18,20,23,31,35,36,43,46,47,50]
shapenet_label_selected =  list(set([i for i in range(55)])-set(shapenet_label_not_selected))

