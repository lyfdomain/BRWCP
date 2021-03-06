# BRWCP
BRCWP: Drug-Protein Interaction Prediction by Correcting the Effect of Incomplete Information in Heterogeneous Information
# Quick Start
to reproduce our results:
1. Download and unzip our code files
2. Run dti_devide.py to reproduce DTI data and their indexes for each fold and save them to ./dataset
3. Run run_main.py to reproduce the ten fold mean true positive rate list, false positive rate list, precision list, and reacll list about drug of the prediction results, and save them to ./result  
4. Run draw_fig.py to reproduce the AUC and AUPR of the prediction result and get the ROC/PRC firgure as follow
![Figure_1](https://user-images.githubusercontent.com/103353319/177000863-b8ba7df6-6696-4a62-a9d8-3c7b5d8343df.png)
## parameters in run_main.py are:
\# rs: restart probability for random walk, defult=0.8 \
\# fr: dimensionals of low-dimensional feature of drugs, defult=500 \
\# fp: dimensionals of low-dimensional feature of proteins, defult=300 \
\# K: number of neighbors retained after pruning, defult=10 \
\# $\eta$: decay term of neighbors' weight, defult=0.7 \
\# $l_1$: number of random walks in the incomplete information network, defult=5\
\# $l_2$: number of random walks in the complete information network, defult=5
# Reproduce ablation experiments
5. After step4, run without_correct.py to get tpr_list_wc, fpr_list_wc, precision list_wc, and reacll_list_wc, and save them to ./result 
6. Run only_feature.py to get tpr_list_of, fpr_list_of, precision list_of, and reacll_list_of, and save them to ./result 
7. Run only_sequence.py to get tpr_list_os, fpr_list_os, precision list_os, and reacll_list_os, and save them to ./result
8. Run draw_fig2.py to reproduce the AUCs and AUPRs of the prediction results and get the ROC/PRC firgure (figure 5 in our paper) as follow 
![Figure_3](https://user-images.githubusercontent.com/103353319/177002170-e4a6ba41-8c54-450f-92ae-e0c4e4b1efab.png)
# Contacts
If you have any questions, please email Li Yanfei (lyfinf@163.com)
