#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=10
step=5e-2
wvecDim=30
memDim=30
rho=1e-4

model="TreeTLSTM"
label="rating"

######################################################## 
# Probably a good idea to let items below here be
########################################################

outfile="models/${model}_${label}_wvecDim_${wvecDim}_memDim_${memDim}_step_${step}_epochs_${epochs}_rho_${rho}_nodrop.bin"

echo $outfile

python runNNet.py --step $step --epochs $epochs --outFile $outfile --wvecDim $wvecDim --memDim $memDim --model $model --rho $rho --label $label
read