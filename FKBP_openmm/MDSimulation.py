#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import glob
import time
from sys import stdout
import re
import time

import warnings
warnings.filterwarnings("ignore")

from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmplumed import PlumedForce
import mdtraj as md
from mdtraj.reporters import XTCReporter
import json
from contextlib import contextmanager


class MDSimulation:

    def __init__(self, work_dir='.'):
        self.work_dir = os.path.abspath(work_dir)  # Store absolute path to avoid confusion

    @contextmanager
    def working_directory(self):
        """A context manager to change to the specified working directory."""
        original_dir = os.getcwd()
        try:
            os.chdir(original_dir+self.work_dir)
            yield
        finally:
            os.chdir(original_dir)

    def remove_previous_old_files(self):
        for f in glob.glob("bck*"):
            os.remove(f)
        for f in glob.glob("simulation_prod_run_*"):
            os.remove(f)

    def get_psf(self, path='../../../FKBP_openmm/openmm/'):
        file_path = path + 'sysinfo.dat'
        with open(file_path, 'r') as file:
            data = json.load(file)
        dim = data['dimensions']

        psf=CharmmPsfFile(path+'step3_input.psf')
        psf.setBox(dim[0]*angstroms,dim[0]*angstroms,dim[0]*angstroms)
        return psf

    def get_pdb(self, path='../../../FKBP_openmm/openmm/'):
        return PDBFile(path+'step3_input.pdb')


    def get_params(self, path='../../../FKBP_openmm/toppar/'):

        ligand_rtf=os.path.join(path,'top_all36_cgenff.rtf')
        ligand_prm=os.path.join(path,'par_all36_cgenff.prm')
        solvent_str=os.path.join(path,'toppar_water_ions.str')
        prot_rtf=os.path.join(path,'top_all36_prot.rtf')
        prot_prm=os.path.join(path,'par_all36_prot.prm')
        params = CharmmParameterSet(prot_rtf,ligand_rtf,prot_prm,ligand_prm,solvent_str)
        return params

    def get_LangevinM_system(self, psf, params, temp=300, dt=0.002):
        '''
        Creates a simulation instance under Langevin Middle Integrator designed in openmm
        '''
        # CHARMM lipid forcefield was parameterized for 8-12 cutoff
        system = psf.createSystem(params,nonbondedMethod=PME,nonbondedCutoff=1.2*nanometer,\
                                switchDistance=1.0*nanometer,constraints=HBonds);
        integrator = LangevinMiddleIntegrator(temp*kelvin, 1/picoseconds,dt*picoseconds);
        platform = Platform.getPlatformByName('CUDA');
        simulation = Simulation(psf.topology, system, integrator, platform)
        return simulation

    def add_pos_res(self, positions, top, simulation, k=10, molecule='protein'):
        '''
        Adds an harmonic potential to the heavy atoms of the system(proteins) with an user defined force constant 'k'
        '''
        
        AA=['ALA','ASP','CYS','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ARG','PRO','GLN','ASN','SER','THR','VAL','TRP','TYR']
        force = CustomExternalForce("kprot*periodicdistance(x, y, z, x0, y0, z0)^2") # Harmonic potential for position restrain
        force.addGlobalParameter("kprot",k*kilojoules_per_mole/angstroms**2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        index=0;
        for i, res in enumerate(top.residues()):
            if res.name in AA:                              # Required to select only the protein atoms
                for at in res.atoms():
                    if not re.search(r'H',at.name):         # All heavy Atoms -(exculdes Hydrogens)
                        force.addParticle(index,positions[index].value_in_unit(nanometers))
                    index+=1;

        posres_sys=simulation.context.getSystem()                  # A gets System for a simulation instance
        posres_sys.addForce(force)                          # Modifies system with custom Force

        if molecule.split('+')[-1]== 'ligand':
            memb_force = CustomExternalForce('klig*periodicdistance(x, y, z, x0, y0, z0)^2;')
            memb_force.addGlobalParameter('klig',k*kilojoules_per_mole/angstroms**2)
            memb_force.addPerParticleParameter('x0')
            memb_force.addPerParticleParameter('y0')
            memb_force.addPerParticleParameter('z0')
            topology=md.Topology.from_openmm(top)
            expression='resname DMS and name S or resname DMS and name O'
            python_exp=topology.select_expression(expression)
            req_indices=np.array(eval(python_exp))
            for ind in req_indices:
                memb_force.addParticle(ind,positions[ind].value_in_unit(nanometers))
            posres_sys.addForce(memb_force)

        simulation.context.reinitialize()                          # initializes the simulation instance with the modified system
        return simulation

    def simulation_preperation(self, equilibration=False):

        pdb = self.get_pdb()
        psf = self.get_psf()
        params = self.get_params()

        simulation = self.get_LangevinM_system(psf,params,temp=303,dt=0.002)
        simulation = self.add_pos_res(pdb.positions,psf.topology,simulation,10,'protein+ligand')    # 10 Kj/(mol.A^2) `
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(tolerance=500*kilojoule/(mole*nanometer), maxIterations=5000)

        if equilibration:
            equilibration_parameter = [[2.0,2.0],[1.0,1.0],[0.2,0.5],[0.0,0.2],[0.0,0.0]]
            
            for i,param in enumerate(equilibration_parameter):
                print('===== Equilibration step %i ====='%(i+1))
                pro_par,lig_par = param[0],param[1]
                equilibrium_steps(pro_par,lig_par)
            
            simulation.reporters=[]
            
            positions= simulation.context.getState(getPositions=True).getPositions()
            run_MD(simulation,positions,nsteps=50000,use_plumed=False,plumed_file=None)
            
            pdb_positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
            PDBFile.writeFile(simulation.topology, pdb_positions, open('step5_0.pdb', 'w'))
            simulation.saveState('equil_finate.state')
            simulation.saveCheckpoint('step5_0.chk')
        
            topology=md.Topology.from_openmm(simulation.topology)
            python_expression=topology.select_expression('not water and not name CLA and not name NA')
            req_indices=np.array(eval(python_expression))
            md.load('equil_final.pdb',atom_indices=req_indices).save_pdb('step5_0.pdb')
        
        
        else:
            print('===== Load pre-Equilibrated system Checkpoint file =====')
            chkptfile = 'step5_0.chk'
            simulation.loadCheckpoint(chkptfile)

        return simulation


    def run_MD(self, simulation, positions, nsteps=50000, use_plumed=True, plumed_file=None, report_steps=True, save_xtc=False, committor=False, stride=2000, threshold=3.0):


        pdb_positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
        PDBFile.writeFile(simulation.topology, pdb_positions, open('test_prod_starting.pdb', 'w'))
        
        if use_plumed:
            if plumed_file is None:
                raise ValueError("PLUMED file must be provided when use_plumed is True.")
            fid=open(plumed_file,'r')
            ff=fid.read()
            force=PlumedForce(ff)
            pl_system=simulation.context.getSystem()
            pl_system.addForce(force)
            simulation.context.reinitialize(True)
        
        simulation.reporters=[]
        
        simulation.context.setPositions(positions)
        outfname=f'simulation_prod_run_1.xtc'
        topology=md.Topology.from_openmm(simulation.topology)
        python_expression=topology.select_expression('not water and not name CLA and not name NA')
        req_indices=np.array(eval(python_expression))
        simulation.reporters.append(XTCReporter(outfname, 500, atomSubset=req_indices))
        
        if report_steps: 
            simulation.reporters.append(StateDataReporter(stdout, 20, step=True, progress=True,speed=True,totalSteps=nsteps,
                                                        potentialEnergy=True, kineticEnergy=True, temperature=True))
        
        if not committor:
            simulation.step(nsteps)
        elif committor:
            print(' ')
            print('========== Committor Activated ==========')
            print(' ')
            print('====== The Ligand has not dissociated yet ======')
            print(' ')
            print(' ')
            nsteps_new = nsteps-(stride*4)
            simulation.step(stride*4)
            
            xtc_files=[]
            for iter in range(0,nsteps_new,stride):
                if iter==0:
                    simulation.step(stride)
                    
                elif iter>0:
                    time_steps = 0.002
                    if iter%5000==0: print(' ****  Current steps  =   %i ps / %i ps'%(iter*time_steps, nsteps*time_steps))
                    if self.check_condition(threshold,iter):
                        if save_xtc:
                            simulation.reporters=[]
                            outfname=f'simulation_prod_run_%i.xtc'%iter
                            xtc_files.append(outfname)
                            topology=md.Topology.from_openmm(simulation.topology)
                            python_expression=topology.select_expression('not water and not name CLA and not name NA')
                            req_indices=np.array(eval(python_expression))
                            simulation.reporters.append(XTCReporter(outfname, 10, atomSubset=req_indices))
                        
                        simulation.step(stride)

                    else: 
                        print('========= Interrupt the simulation ========= ')
                        # Save iter value to a file when interrupting the simulation
                        with open('last_iter_value.txt', 'w') as f:
                            f.write(str(iter))
                        # Assuming the interruption is akin to a final stage, save all xtc_files
                        with open('xtc_files.txt', 'w') as f:
                            for fname in xtc_files:
                                f.write(fname + '\n')                        
                        break
        
        pdb_positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
        PDBFile.writeFile(simulation.topology, pdb_positions, open('final_run_end.pdb', 'w'))
        
        # Initialize PLUMED if specified
        if use_plumed:
            simulation.context.getSystem().removeForce(simulation.context.getSystem().getNumForces()-1)  #from plumedforc



    def check_condition(self, threshold=3.0,iter=0):
        val=float(open('COLVAR','r').readlines()[-2].strip().split(' ')[29])
        if val>threshold:
            print(f'========= I have reached values:{val:2.2f} ========= ')
            return False
        else: 
            if iter % 5000 == 0:  # Assuming a stride of 5000; adjust the multiplier as needed
                print(f' Distance between the ligand (CoM) and protein (CoM) = {val:3.2f}   nm   ')
            return True



    def equilibrium_steps(self, pro_par, lig_par):
        #simulation_constraints
        simulation.context.setParameter('kprot',pro_par*kilojoules_per_mole/angstroms**2)
        simulation.context.setParameter('klig',lig_par*kilojoules_per_mole/angstroms**2)
        
        positions= simulation.context.getState(getPositions=True).getPositions()
        self.run_MD(simulation,positions,nsteps=500,use_plumed=False,plumed_file=None,report_steps=False)



    def remove_last_line(self, file_path='COLVAR'):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        with open(file_path, 'w') as file:
            file.writelines(lines[:-1])

    def remove_line_change_rows(self, file_path='COLVAR'):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        with open(file_path, 'w') as file:
            file.writelines(lines[:-50])


    def combine_xtc_files(self, topology_file='step5_1.pdb', stride=2000):
        xtc_files = []
        iter = int(np.loadtxt('last_iter_value.txt'))
        for j in range(0, iter-stride, stride):
            file_name = f'simulation_prod_run_{j+stride}.xtc'
            xtc_files.append(file_name)
            
        topology = md.load_topology(topology_file) 
        combined_trajectory = md.load(xtc_files[0], top=topology)
        
        for file_name in xtc_files:
            trajectory = md.load(file_name, top=topology)
            combined_trajectory = md.join([combined_trajectory, trajectory])
        
        combined_trajectory.save('combined_trajectory.xtc')
        every_10th_frame = combined_trajectory[::10]
        every_10th_frame.save('every_10th_frame.xtc')



    def measure_execution_time(self,func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time


