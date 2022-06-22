# BRWCP
BRCWP: Drug-Protein Interaction Prediction by Correcting the Effect of Incomplete Information in Heterogeneous Information
# Quick Start
to reproduce our results:
1. Save data in source_data to ./source_data
2. Run dti_devide.py to reproduce DTI data and their indexes for each fold and save them to ./dataset
3. Run run_main.py to reproduce the ten fold mean true positive rate list, false positive rate list, precision list, and reacll list about drug of the prediction results, and save them to ./result  
4. Run draw_fig.py to reproduce the AUC and AUPR of the prediction result and get the ROC/PRC firgure as follow
![Figure_111](https://user-images.githubusercontent.com/103353319/174630493-65b9feee-38f5-4a42-a511-5b715a1cc6a8.png)
## parameters in run_main.py are:
rs: restart probability for random walk, defult=0.8 \n
rs: restart probability for random walk, defult=0.8 
### rs: restart probability for random walk, defult=0.8 
### rs: restart probability for random walk, defult=0.8 
### rs: restart probability for random walk, defult=0.8 
# Reproduce ablation experiments
5. After step4, run without_correct.py to get tpr_list_wc, fpr_list_wc, precision list_wc, and reacll_list_wc, and save them to ./result 
6. Run only_feature.py to get tpr_list_of, fpr_list_of, precision list_of, and reacll_list_of, and save them to ./result 
7. Run only_sequence.py to get tpr_list_os, fpr_list_os, precision list_os, and reacll_list_os, and save them to ./result
8. Run draw_fig2.py to reproduce the AUCs and AUPRs of the prediction results and get the ROC/PRC firgure (figure 5 in our paper) as follow 
![Figure_xin1](https://user-images.githubusercontent.com/103353319/174951264-512f2955-d24d-446b-b0d5-742866bb5635.png)
