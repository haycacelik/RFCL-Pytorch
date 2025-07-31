#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition dllabdlc_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name RFCLTrain             # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.
#SBATCH --output logs/%x-%A-HelloCluster.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-HelloCluster.err    # STDERR  short: -e logs/%x-%A-job_name.out



# Define the amount of memory required per node
#SBATCH --mem 6GB

lscpu
echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source /work/dlclarge2/sheikhr-rfcl-raaez/miniconda/bin/activate # Adjust to your path of Miniconda installation
conda activate torchRFCL 
cd /work/dlclarge2/sheikhr-rfcl-raaez/RFCL-Pytorch


export DISPLAY=:99
export MUJOCO_GL=egl  # or osmesa
export PYOPENGL_PLATFORM=egl

export MINARI_DATASETS_PATH=/work/dlclarge2/sheikhr-rfcl-raaez/rfcl-pytorch/datasets
# Running the job
start=`date +%s`

/work/dlclarge2/sheikhr-rfcl-raaez/RFCL-Pytorch/scripts/adroit/adroit_fast.sh

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
