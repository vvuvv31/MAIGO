#!/bin/bash
#SBATCH --job-name=recoil_stopping
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=/mnt/sda/wuwei/all_ion_elastic_20260911/recoil_%j.log
#SBATCH --error=/mnt/sda/wuwei/all_ion_elastic_20260911/recoil_%j.err
set -euo pipefail
cd /mnt/sda/wuwei/all_ion_elastic_20260911
source /software/geant4-11.3.2/bin/geant4.sh
/home/wuwei/topas/all-ion-elastic-build-20260911/topas-recoil /mnt/sda/wuwei/all_ion_elastic_20260911/recoil_stopping.txt
