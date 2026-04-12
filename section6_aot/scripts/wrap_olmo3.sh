#!/bin/bash -l
#SBATCH --job-name=olmo3_aot
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

rm -r /data/jade-beta/jade3396/cache/models--allenai--OLMo-2-1124-7B/

apptainer exec --fakeroot --bind /data:/data --rocm $DATA/containers/pytorch_rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0.sif \
    bash /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/aot/scripts/olmo3.sh $1 $2
