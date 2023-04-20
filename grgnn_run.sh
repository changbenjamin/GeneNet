#!/bin/bash

#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time 1-0:30:00
#SBATCH --mem=10G
#SBATCH --output=genenet_all_settings3_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benjaminchang@college.harvard.edu

###

source /home/bc198/miniconda3/bin/activate

conda activate myenv

cd preprocessing

python Preprocessing_DREAM5.py --dream-num 3

#python Main_inductive_ensemble.py --traindata-name data3 --testdata-name data3

#python Main_inductive_semi.py --traindata-name data3 --testdata-name data3

#python Main_inductive_SVM.py --traindata-name data3 --testdata-name data3

#python Main_inductive_genenet.py --traindata-name data3 --testdata-name data3

cd ..

# OPTIMIZING NOISE PARAMETERS

python genenet.py --data-name data3 --bio-mean 0.4 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.4 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.35 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.35 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.3 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.3 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.25 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.25 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.2 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.2 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.15 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.15 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.1 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.1 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.05 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.05 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean 0.0 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean 0.0 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.05 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.05 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.1 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.1 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.15 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.15 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.2 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.2 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.25 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.25 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.3 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.3 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.35 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.35 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.4 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.4 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.45 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.45 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.5 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.5 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.55 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.55 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.6 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.6 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.65 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.65 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.7 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.7 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.75 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.75 --bio-std-dev 0.5
python genenet.py --data-name data3 --bio-mean -0.8 --bio-std-dev 0.0
python genenet.py --data-name data3 --bio-mean -0.8 --bio-std-dev 0.5