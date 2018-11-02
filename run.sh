#!/bin/bash

#PBS -l nodes=1:ppn=2
#PBS -l walltime=24:00:00
#PBS -q normal_q
#PBS -A BDTScascades
#PBS -W group_list=cascades
#PBS -M ashishb@vt.edu
#PBS -m bea

# Add any modules you might require. This example removes all modules and then adds
# the Intel compiler and mvapich2 MPI modules module. Use the module avail command
# to see a list of available modules.
module purge
module load Anaconda/5.1.0
module load jdk/1.8.0
source activate pytorch

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Start training process
rm -rf run.sh.*
echo "Starting the conversion process for the dataset"
source ./setup_stanford_corenlp.sh
python3 make_datafiles.py -f ../TextSummarization/dataset/fulltext -s ../TextSummarization/summary -o output/

exit
