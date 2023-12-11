module load gromacs/2019.4
gmx_mpi grompp -f step5_production.mdp -o md.tpr -c step5_0.gro -p topol.top -n index.ndx -maxwarn 4
