import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pcntoolkit as pcn

os.getcwd()
os.chdir('./anat_HBR/')
processing_dir = os.getcwd()

anat = pd.read_csv('anat_norm.csv')
adhd200 = anat.loc[anat['site'] == 4]
anat = anat.loc[anat['site'] != 4]
sites = anat['site'].unique() # 1：CBD_H,2: CBD_P; 3:PKU

# Step 1: Prepare training and testing sets
te = np.random.uniform(size=anat.shape[0]) > 0.7
tr = ~te
anat_tr = anat.loc[tr]
anat_te = anat.loc[te]

te = np.random.uniform(size=adhd200.shape[0]) > 0.7
tr = ~te
adhd200_tr = adhd200.loc[tr]
adhd200_te = adhd200.loc[te]

# sample size check
for i,s in enumerate(sites):
    idx = anat_tr['site'] == s
    idxte = anat_te['site'] == s
    print(i,s, sum(idx), sum(idxte))

# Step 2: Configure HBR inputs: X=covariates, Y=measures and batch effects
measures = list(anat)[4:239]
X_train = (anat_tr['age']).to_numpy(dtype=float)
Y_train = anat_tr[measures].to_numpy(dtype=float)
batch_effects_train = anat_tr[['site','sex']].to_numpy(dtype=int)

with open('X_train.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_train), file)
with open('Y_train.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_train), file)
with open('trbefile.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_train), file)

X_test = (anat_te['age']).to_numpy(dtype=float)
Y_test = anat_te[measures].to_numpy(dtype=float)
batch_effects_test = anat_te[['site','sex']].to_numpy(dtype=int)

with open('X_test.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_test), file)
with open('Y_test.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_test), file)
with open('tsbefile.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_test), file)

#anat_tr[['age']].to_csv('x_train.txt',sep=' ', header=False,index=False)
#anat_tr[measures].to_csv('y_train.txt',sep=' ', header=False,index=False)
#anat_tr[['sitenum','sex']].to_csv('trbefile.txt',sep=' ', header=False,index=False)
#anat_te[['age']].to_csv('x_test.txt',sep=' ', header=False,index=False)
#anat_te[measures].to_csv('y_test.txt',sep=' ', header=False,index=False)
#anat_te[['sitenum','sex']].to_csv('tsbefile.txt',sep=' ', header=False,index=False)

#Step 3: Files and Folders grooming
respfile = os.path.join(processing_dir, 'Y_train.pkl')       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
covfile = os.path.join(processing_dir, 'X_train.pkl')        # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)

testrespfile_path = os.path.join(processing_dir, 'Y_test.pkl')    # measurements  for the testing samples
testcovfile_path = os.path.join(processing_dir, 'X_test.pkl')     # covariate file for the testing samples

trbefile = os.path.join(processing_dir, 'trbefile.pkl')      # training batch effects file (eg scanner_id, gender)  (columns: the various batch effects, rows: observations or subjects)
tsbefile = os.path.join(processing_dir, 'tsbefile.pkl')      # testing batch effects file

output_path = os.path.join(processing_dir, 'Models/')    #  output path, where the models will be written
log_dir = os.path.join(processing_dir, 'log/')
if not os.path.isdir(output_path):
    os.mkdir(output_path)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

outputsuffix = '_estimate'      # a string to name the output files, of use only to you, so adapt it for your needs.

# Step 4: Estimating the models
pcn.normative.estimate(covfile=covfile,
                       respfile=respfile,
                       tsbefile=tsbefile,
                       trbefile=trbefile,
                       alg='hbr',
                       log_path=log_dir,
                       output_path=output_path,
                       testcov=testcovfile_path,
                       testresp=testrespfile_path,
                       outputsuffix=outputsuffix,
                       savemodel=True,
                       binary=True)

#Step 5: Transfering the models to unseen sites
X_adapt = (adhd200_tr['age']).to_numpy(dtype=float)
Y_adapt = adhd200_tr[measures].to_numpy(dtype=float)
batch_effects_adapt = adhd200_tr[['site', 'sex']].to_numpy(dtype=int)

with open('X_adaptation.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_adapt), file)
with open('Y_adaptation.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_adapt), file)
with open('adbefile.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_adapt), file)

# Test data
X_test_txfr = (adhd200_te['age']).to_numpy(dtype=float)
Y_test_txfr = adhd200_te[measures].to_numpy(dtype=float)
batch_effects_test_txfr = adhd200_te[['site', 'sex']].to_numpy(dtype=int)

with open('X_test_txfr.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_test_txfr), file)
with open('Y_test_txfr.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(Y_test_txfr), file)
with open('txbefile.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_test_txfr), file)

respfile = os.path.join(processing_dir, 'Y_adaptation.pkl')
covfile = os.path.join(processing_dir, 'X_adaptation.pkl')
testrespfile_path = os.path.join(processing_dir, 'Y_test_txfr.pkl')
testcovfile_path = os.path.join(processing_dir, 'X_test_txfr.pkl')
trbefile = os.path.join(processing_dir, 'adbefile.pkl')
tsbefile = os.path.join(processing_dir, 'txbefile.pkl')

output_path = os.path.join(processing_dir, 'Transfer/')
if not os.path.isdir(output_path):
    os.mkdir(output_path)
model_path = os.path.join(processing_dir, 'Models/')  # path to the previously trained models
outputsuffix = '_transfer'  # suffix added to the output files from the transfer function

# point the models just trained, and new site data (training and testing).
yhat, s2, z_scores = pcn.normative.transfer(covfile=covfile,
                                            respfile=respfile,
                                            tsbefile=tsbefile,
                                            trbefile=trbefile,
                                            model_path=model_path,
                                            alg='hbr',
                                            log_path=log_dir,
                                            binary=True,
                                            output_path=output_path,
                                            testcov=testcovfile_path,
                                            testresp=testrespfile_path,
                                            outputsuffix=outputsuffix,
                                            savemodel=True)
