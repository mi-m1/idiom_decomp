#!/bin/bash -l
#SBATCH --job-name=olmo2_infini
#SBATCH --partition=medium
#SBATCH --gres=gpu:0
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

apptainer exec \
    --fakeroot \
    --bind /data:/data \
    --bind /etc/ssl/certs:/etc/ssl/certs \
    --bind /etc/pki/ca-trust:/etc/pki/ca-trust \
    --rocm $DATA/containers/pytorch_rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0.sif \
    bash /data/jade-beta/jade3396/src/nov2025/idioms_decomposability/infini_freq/scripts/olmo2.sh
