#!/bin/bash
#SBATCH --job-name=
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=test_output.out
#SBATCH --mail-user=urajnis@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=64G

srun python -u test.py