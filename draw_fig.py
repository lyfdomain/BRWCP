import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from tqdm import tqdm
import pylab as Plot

CPC_FPR = np.loadtxt('./result/fpr_list.txt')
CPC_TPR = np.loadtxt('./result/tpr_list.txt')
CPC_RECALL = np.loadtxt('./result/recall_list.txt')
CPC_P = np.loadtxt('./result/precision_list.txt')



plt.figure(1, figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.rcParams['figure.figsize'] = (4, 4)
plt.title('ROC curve', fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)


plt.plot(CPC_FPR, CPC_TPR, 'aqua', label='BRWCP:{:.4f}'.format(auc(CPC_FPR, CPC_TPR)))


plt.legend(loc='lower right', fontsize=12)
# plt.plot(fpr_mean, tpr_mean)


plt.subplot(1, 2, 2)
plt.title('PR curve', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)

plt.plot(CPC_RECALL, CPC_P, 'aqua', label='BRWCP:{:.4f}'.format(auc(CPC_RECALL, CPC_P)+CPC_RECALL[0]*CPC_P[0]))


plt.legend(loc='upper right', fontsize=12)
plt.show()
