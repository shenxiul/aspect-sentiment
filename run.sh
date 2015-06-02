#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=20
step=1e-1
wvecDim=10
memDim=10
rho=0e-6

model="TreeLSTM"
label="aspect"

######################################################## 
# Probably a good idea to let items below here be
########################################################

outfile="models/${model}_${label}_wvecDim_${wvecDim}_memDim_${memDim}_step_${step}_epochs_${epochs}_rho_${rho}.bin"

echo $outfile

python runNNet.py --step $step --epochs $epochs --outFile $outfile --wvecDim $wvecDim --memDim $memDim --model $model --rho $rho --label $label
