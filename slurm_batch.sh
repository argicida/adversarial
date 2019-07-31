#!/bin/bash -l
# NOTE the -l flag!
#
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J blackbox_test

# Standard out and Standard Error output files
#SBATCH -o blackbox.output
#SBATCH -e blackbox.output

# To send emails, set the adcdress below and remove one of the "#" signs.
# IMPORTANT : GIVES GPU FAILURE NOTIFICATIONS
#SBATCH --mail-user xxd9704@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# run time request
# day-hour:minute:second
#SBATCH -t 5:0:0

# use the "blackbox" account, which corresponds to our project name
# put the job in the "onboard" partition (cuz we are noobs) 
# -n 4 requests 4 CPUs
# --gres=gpu:v100:4 requests 4 Nvidia v100 GPUs
#SBATCH -A blackbox -p onboard -n 4 --gres=gpu:v100:4

# Job memory requirements in MB
# be generous, they have tons of RAM
# 65536 gives 64 Gig
#SBATCH --mem=65536

#
# Your job script goes below this line.  
# 

# spack configures the software stack for the session
# spack list for all available packages
# spack info for more information on a particular package
spack load python@3.6.8
spack load py-pytorch ^python@3.6.8 # this gives us torchvision and numpy
spack load py-pillow ^python@3.6.8 # this gives us PIL and pickle
python3 train_patch.py paper_obj

