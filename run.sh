#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=20
step=5e-2
wvecDim=100
memDim=100
rho=1e-6

model="RNN" #either RNN, RNN2, RNN3, RNTN, or DCNN
label="rating"

######################################################## 
# Probably a good idea to let items below here be
########################################################

outfile="models/${model}_${label}_wvecDim_${wvecDim}_memDim_${memDim}_step_${step}_epochs_${epochs}_rho_${rho}.bin"

echo $outfile

python runNNet.py --step $step --epochs $epochs --outFile $outfile --wvecDim $wvecDim --memDim $memDim --model $model --rho $rho --label $label
