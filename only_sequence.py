import numpy as np
from function import *
from tqdm import trange
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics
import copy
rs = 0.8
n_fold=10
dr_dis=np.loadtxt("./source_data/mat_drug_disease.txt")
dr_pre=np.loadtxt("./source_data/mat_drug_protein.txt")
pre_dis=np.loadtxt("./source_data/mat_protein_disease.txt")
simdr=np.loadtxt("./source_data/Similarity_Matrix_Drugs.txt")
simpre=np.loadtxt("./source_data/Similarity_Matrix_Proteins.txt")

drnum=len(dr_dis)
disnum=len(dr_dis[0])
prenum=len(pre_dis)

def drug_lapsimm(dd):
    mm = np.zeros(dd.shape)
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            if np.sum(dd[i, :])!=0:
                mm[i, j] = dd[i, j] / np.sqrt(np.sum(dd[i, :]) * np.sum(dd[i, :]))
    return mm


def dis_lapsimm(dd):
    mm = np.zeros(dd.shape)
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            if np.sum(dd[:, j]) != 0:
                mm[i, j] = dd[i, j] / np.sqrt(np.sum(dd[:, j]) * np.sum(dd[:, j]))
    return mm

index_1 = np.loadtxt("./dataset/index_1.txt")
index_0 = np.loadtxt("./dataset/index_0.txt")
index = np.hstack((index_1, index_0))
reala=dr_pre
sr = simdr
sd = simpre/100

RR=np.zeros(dr_pre.shape)

for f in trange(n_fold):
    a = np.loadtxt("./dataset/DTI" + str(f) + ".txt")
    idx=index[f,:]
    R1 = copy.deepcopy(a)
    R2 = copy.deepcopy(a)
    mr=drug_lapsimm(sr)
    md=dis_lapsimm(sd)
    R1 = rs * mr.dot(R1) + (1-rs) * a
    R2 = rs * R2.dot(md) + (1-rs) * a
    R = copy.deepcopy(a)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i][j] = max(R[i][j],(R1[i][j] + R2[i][j]) / 2)

    realvalue=np.zeros(R.shape)
    for i in range(len(idx)):
        d=int(idx[i]/prenum)
        p=int(idx[i]%prenum)
        realvalue[d,p]=reala[d,p]
    for i in range(len(idx)):
        d=int(idx[i]/prenum)
        p=int(idx[i]%prenum)
        RR[d,p]=R[d,p]

y_true_m = dr_pre.tolist()
y_pre_m = RR.tolist()

tpr_cov=[[] for i in range(n_fold)]
fpr_cov=[[] for i in range(n_fold)]
recall_cov=[[] for i in range(n_fold)]
precision_cov=[[] for i in range(n_fold)]

for f in range(n_fold):
    idx=index[f,:]
    singal=[[] for i in range(len(y_true_m))]
    singal_test=[[] for i in range(len(y_true_m))]
    for i in trange(len(y_true_m)):
        for j in range(len(y_true_m[0])):
            if i*prenum+j in idx:
                singal[i].append(y_true_m[i][j])
                singal_test[i].append(y_pre_m[i][j])
    y_true=singal
    y_pre=singal_test

    idx = []
    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    c = 0
    for i in trange(len(y_true)):
        if np.sum(np.array(y_true[i])) == 0:
            c += 1
            continue
        else:
            tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(y_true[i]), np.array(y_pre[i]))
            fpr_list.append(fpr1)
            tpr_list.append(tpr1)
            precision_list.append(precision1)
            recall_list.append(recall1)
    coverage = []

    for i in tpr_list:
        try:
            coverage.append(i.index(1.0)+1)
        except:
            print('1')
    print(np.mean(np.array(coverage)))

    tpr = equal_len_list(tpr_list)
    print(len(tpr ),type(tpr))
    fpr = equal_len_list(fpr_list)
    precision = equal_len_list(precision_list)
    recall = equal_len_list(recall_list)
    tpr_mean = np.mean(tpr, axis=0)
    tpr_cov[f]=tpr_mean
    print(type(tpr_mean),len(tpr_mean))
    fpr_mean = np.mean(fpr, axis=0)
    fpr_cov[f]=fpr_mean
    recall_mean = np.mean(recall, axis=0)
    recall_cov[f]=recall_mean
    precision_mean = np.mean(precision, axis=0)
    precision_cov[f]=precision_mean
    print('The auc of prediction is:', sklearn.metrics.auc(fpr_mean, tpr_mean))
    print('The aupr of prediction is:', sklearn.metrics.auc(recall_mean, precision_mean)+recall_mean[0]*precision_mean[0])

tpr = equal_len_list(tpr_cov)
fpr = equal_len_list(fpr_cov)
precision = equal_len_list(precision_cov)
recall = equal_len_list(recall_cov)

tpr_mean = np.mean(tpr, axis=0)
fpr_mean = np.mean(fpr, axis=0)
recall_mean = np.mean(recall, axis=0)
precision_mean = np.mean(precision, axis=0)

print('The auc of prediction is:', sklearn.metrics.auc(fpr_mean, tpr_mean))
print('The aupr of prediction is:', sklearn.metrics.auc(recall_mean, precision_mean)+recall_mean[0]*precision_mean[0])

np.savetxt('result/fpr_list_os.txt', fpr_mean)
np.savetxt('result/tpr_list_os.txt', tpr_mean)
np.savetxt('result/recall_list_os.txt', recall_mean)
np.savetxt('result/precision_list_os.txt', precision_mean)




