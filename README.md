# BRWCP
BRCWP: Drug-Protein Interaction Prediction by Correcting the Effect of Incomplete Information in Heterogeneous Information
Quick Start
to reproduce our results:
1. Save data in source_data to ./source_data
2. Run dti_devide to reproduce DTI data and their indexes for each fold and save them to ./dataset
3. Run dti_devide to reproduce the ten fold mean true positive rate list, false positive rate list, precision list, and reacll list about drug of the prediction results, and save them to ./result  
4. run draw_fig to reproduce the AUC and AUPR of the prediction result and get the ROC/PRC firgure
