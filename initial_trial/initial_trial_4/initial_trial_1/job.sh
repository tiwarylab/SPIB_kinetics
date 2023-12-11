#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -A tiwary-prj-chem
#SBATCH --ntasks-per-node=80
#SBATCH --job-name=FKBP_trial1_DMSO
#SBATCH --mail-user=sueminl@umd.edu
#SBATCH --mail-type=END,FAIL

module load gromacs/2019.4
gmx_mpi grompp -f step5_production.mdp -o md.tpr -c step5_0.gro -p topol.top -n index.ndx -maxwarn 4
mpirun -np $SLURM_NTASKS -c 80 gmx_mpi mdrun -v -deffnm md --plumed plumed_initial.dat -cpi md.cpt -ntomp 1
#awk 'NR%100==0' COLVAR>COLVAR_reduced
