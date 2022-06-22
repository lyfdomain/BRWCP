import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from tqdm import tqdm
import pylab as Plot

CPC_FPR = np.loadtxt('./result/fpr_list.txt')
CPC_TPR = np.loadtxt('./result/tpr_list.txt')
CPC_RECALL = np.loadtxt('./result/recall_list.txt')
CPC_P = np.loadtxt('./result/precision_list.txt')

of_FPR = np.loadtxt('./result/fpr_list_of.txt')
of_TPR = np.loadtxt('./result/tpr_list_of.txt')
of_RECALL = np.loadtxt('./result/recall_list_of.txt')
of_P = np.loadtxt('./result/precision_list_of.txt')

os_FPR = np.loadtxt('./result/fpr_list_os.txt')
os_TPR = np.loadtxt('./result/tpr_list_os.txt')
os_RECALL = np.loadtxt('./result/recall_list_os.txt')
os_P = np.loadtxt('./result/precision_list_os.txt')


wc_FPR = np.loadtxt('./result/fpr_list_wc.txt')
wc_TPR = np.loadtxt('./result/tpr_list_wc.txt')
wc_RECALL = np.loadtxt('./result/recall_list_wc.txt')
wc_P = np.loadtxt('./result/precision_list_wc.txt')

plt.figure(1, figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.rcParams['figure.figsize'] = (4, 4)
plt.title('ROC curve', fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)


plt.plot(CPC_FPR, CPC_TPR, 'aqua', label='BRWCP:{:.4f}'.format(auc(CPC_FPR, CPC_TPR)))
plt.plot(wc_FPR, wc_TPR, 'b', label='without correct:{:.4f}'.format(auc(wc_FPR, wc_TPR)))
plt.plot(of_FPR, of_TPR, 'r', label='only feature:{:.4f}'.format(auc(of_FPR, of_TPR)))
plt.plot(os_FPR, os_TPR, 'gold', label='only sequence:{:.4f}'.format(auc(os_FPR, os_TPR)))
plt.legend(loc='lower right', fontsize=12)


plt.subplot(1, 2, 2)
plt.title('PR curve', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)

plt.plot(CPC_RECALL, CPC_P, 'aqua', label='BRWCP:{:.4f}'.format(auc(CPC_RECALL, CPC_P)+CPC_RECALL[0]*CPC_P[0]))
plt.plot(wc_RECALL, wc_P, 'b', label='without correct:{:.4f}'.format(auc(wc_RECALL, wc_P)+wc_RECALL[0]*wc_P[0]))
plt.plot(of_RECALL, of_P, 'r', label='only feature:{:.4f}'.format(auc(of_RECALL, of_P)+of_RECALL[0]*of_P[0]))
plt.plot(os_RECALL, os_P, 'gold', label='only sequence:{:.4f}'.format(auc(os_RECALL, os_P)+os_RECALL[0]*os_P[0]))
plt.legend(loc='upper right', fontsize=12)
plt.show()
