#!/bin/bash -l
# NOTE the -l flag!
#
# If you need any help, please email rc-help@rit.edu
#


# Name of the job 
#SBATCH -J blackbox_test


# Standard out and Standard Error output files
#SBATCH -o blackbox.out
#SBATCH -e blackbox.err


# To send emails, set the adcdress below and remove one of the "#" signs.
# IMPORTANT : GIVES GPU FAILURE NOTIFICATIONS
#SBATCH --mail-user nxg8159@rit.edu
# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL


# run time request
# day-hour:minute:second
#SBATCH -t 24:0:0


# use the "blackbox" account, which corresponds to our project name
# put the job in the "interactive" partition, for the RTX6000 GPU on theocho.rc.rit.edu
# alternatively, we have access to the "onboard" partition
# -n 4 requests 4 CPUs
# --gres=gpu:v100:1 requests 1 Nvidia rtx6000 GPU (which gives us 24 gig of VRAM)
#SBATCH -A blackbox -p onboard -n 4 --gres=gpu:v100:1


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
source ./venv/bin/activate

for i in {1..200}
do
   python test_patch.py
done

deactivate
