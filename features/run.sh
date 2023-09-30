#!/bin/bash

# https://stackoverflow.com/questions/12316167/does-linux-shell-support-list-data-structure
prefix="/home/igor/4o-SEMESTRE/IC0009" 
datasets=("Glomerulus" "Glomerulus_gamma1" "Glomerulus_gamma2" "Glomerulus_gamma3" "Glomerulus_gaussian" "Glomerulus_laplace" "Glomerulus_hist_equalize2") 
for ds in "${datasets[@]}"; 
do 
	python3 generate_features.py "${prefix}/${ds}" 
done

