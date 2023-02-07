#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bvani
"""
import numpy as np
from sys import stdout
from openmmplumed import PlumedForce
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
#from openmm.app import *
#from openmm import *
#from openmm.unit import *

import pdbfixer

def RegSpaceClustering(z, min_dist, max_centers=200, batch_size=100,randomseed=0):
    '''Regular space clustering.
    Args:
        data: ndarray containing (n,d)-shaped float data
        max_centers: the maximum number of cluster centers to be determined, integer greater than 0 required
        min_dist: the minimal distances between cluster centers
    '''
    num_observations, d = z.shape
    p = np.hstack((0,np.random.RandomState(seed=randomseed).permutation(num_observations-1)+1))
    data = z[p]
    center_list = data[0, :].copy().reshape(d,1)
    centerids=[p[0]+1]
    i = 1
    while i < num_observations:
        x_active = data[i:i+batch_size, :]
        distances = np.sqrt((np.square(np.expand_dims(center_list.T,0) - np.expand_dims(x_active,1))).sum(axis=-1))
        indice = tuple(np.nonzero(np.all(distances > min_dist, axis=-1))[0])
        if len(indice) > 0:
            # the first element will be used
            #print(center_list.shape,x_active.shape,x_active[indice[0]].shape)
            center_list = np.hstack((center_list, x_active[indice[0]].reshape(d,1)))
            centerids.append(p[i+indice[0]]+1)
            i += indice[0]
        else:
            i += batch_size
        if len(centerids) >= max_centers:
            print("%i centers: Exceeded the maximum number of cluster centers!\n"%len(centerids))
            print("Please increase dmin!\n")
            raise ValueError
    print("Found %i centers!"%len(centerids))
    return center_list,centerids

def fix_pdb(index):
  """
  fixes the raw pdb from colabfold using pdbfixer.
  This needs to be performed to cleanup the pdb and to start simulation 

  Fixes performed: missing residues, missing atoms and missing Terminals
  """
  raw_pdb=f'pred_{index}.pdb';

  # fixer instance
  fixer = pdbfixer.PDBFixer(raw_pdb)

  #finding and adding missing residues including terminals
  fixer.findMissingResidues()
  fixer.findMissingAtoms()
  fixer.addMissingAtoms()
  out_handle = open(f'fixed_{index}.pdb','w')
  PDBFile.writeFile(fixer.topology, fixer.positions, out_handle,keepIds=True)


def run_unbiased(on_gpu,plumedfile,dt,temp,freq,nstep,index):
  """
  Runs an unbiased simulation on the cluster center using openMM.
  The MD engine also uses plumed for on the fly calculations
  input : raw pdb from colabfold
  forcefields : amber03 and tip3p
  output : fixed_{index}.pdb, unb_{index}.pdb, COLVAR_unb
  """
  if plumedfile != "None":
    use_plumed=True
  
  outfreq = 0
  chkpt_freq=0
  save_chkpt_file=False
  
  print(f'We are at {os.getcwd()}')
  
  #fixing PDBs to avoid missing residue or terminal issues
  fix_pdb(index);
  pdb_fixed=f'fixed_{index}.pdb'
  
  #Get the structure and assign force field
  pdb = PDBFile(pdb_fixed) 
  forcefield = ForceField('amber03.xml', 'tip3p.xml')
  
  # Placing in a box and adding hydrogens, ions and water
  modeller = Modeller(pdb.topology, pdb.positions)
  modeller.addHydrogens(forcefield)
  modeller.addSolvent(forcefield, padding=0.5*nanometers, model='tip3p', neutralize=True, positiveIon='Na+', negativeIon='Cl-')

  #Create simulation system and assign integrator
  system = forcefield.createSystem(modeller.topology,nonbondedMethod=PME,nonbondedCutoff=1.2*nanometer,
        switchDistance=1.0*nanometer,constraints=HBonds)
  integrator = NoseHooverIntegrator(temp*kelvin, freq/picoseconds,
                                dt*picoseconds);
  if use_plumed:
    fid=open(plumedfile,'r')
    ff=fid.read()
    force=PlumedForce(ff)
    system.addForce(force)
    #system.addForce(MonteCarloBarostat(press*bar, temp*kelvin))   #Pressure control
    if on_gpu:
      platform = Platform.getPlatformByName('CUDA')
      properties = {'Precision': 'double','CudaCompiler':'/usr/local/cuda/bin/nvcc'}
      simulation = Simulation(modeller.topology, system, integrator, platform)
    else:
      platform = Platform.getPlatformByName('CPU')
      simulation = Simulation(modeller.topology, system, integrator, platform)
    
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    minim_positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
    PDBFile.writeFile(simulation.topology, minim_positions, open(f'minim_{index}.pdb', 'w'))
    if save_chkpt_file:
      simulation.reporters.append(CheckpointReporter(chkpt_fname, chkpt_freq))
    
    simulation.step(nstep)
    positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(f'unb_{index}.pdb', 'w'))

def make_biased_plumed(plumedfile,weights,colvar,height,biasfactor,width1,width2,gridmin1,gridmin2,gridmax1,gridmax2,temperature):
    f_unb=open(plumedfile)
    f=open('plumed_biased.dat','w')
    lines=f_unb.readlines()
    p=lines.pop(-2)
    w0=",".join([str(weights[0][i]) for i in range (len(weights[0]))])
    w1=",".join([str(weights[1][i]) for i in range (len(weights[1]))])
    lines.insert(-1,"\nsigma1: COMBINE ARG=%s COEFFICIENTS=%s PERIODIC=NO"%(colvar,w0))
    lines.insert(-1,"\nsigma2: COMBINE ARG=%s COEFFICIENTS=%s PERIODIC=NO"%(colvar,w1))

    lines.insert(-1,"\nMETAD ...\n \
      LABEL=metad\n \
      ARG=sigma1,sigma2\n \
      PACE=1000 HEIGHT=%f TEMP=%i\n \
      BIASFACTOR=%i\n \
      SIGMA=%f,%f\n \
      FILE=HILLS GRID_MIN=%f,%f GRID_MAX=%f,%f GRID_BIN=200,200\n \
      CALC_RCT RCT_USTRIDE=500\n \
      ... METAD\n"%(height,temperature,biasfactor,width1,width2,gridmin1,gridmin2,gridmax1,gridmax2))
  
    f.writelines(lines)
    f.write("\n PRINT ARG=%s,sigma1,sigma2,metad.rbias STRIDE=500 FILE=COLVAR_biased.dat"%colvar)

    f.close()
    
    
def run_biased(on_gpu,plumedfile,dt,temp,freq,nstep,index,save_chkpt_file=True,chkpt_freq=500000,restart=False):
  #Saves check point file after every 500000 steps (by default) 
  use_plumed=True
  outfreq = 5000

  #Pdb from previous unbiased run
  pdb = PDBFile(f'unb_{index}.pdb') 
  forcefield = ForceField('amber03.xml', 'tip3p.xml')

  system = forcefield.createSystem(pdb.topology,nonbondedMethod=PME,nonbondedCutoff=1.2*nanometer,
      switchDistance=1.0*nanometer,constraints=HBonds)
  integrator = NoseHooverIntegrator(temp*kelvin, freq/picoseconds,
                              dt*picoseconds)
  if use_plumed:
    fid=open(plumedfile,'r')
    ff=fid.read()
    force=PlumedForce(ff)
    system.addForce(force)
    #system.addForce(MonteCarloBarostat(press*bar, temp*kelvin)) #pressure control
    if on_gpu:
      platform = Platform.getPlatformByName('CUDA')
      properties = {'Precision': 'double','CudaCompiler':'/usr/local/cuda/bin/nvcc'}
      simulation = Simulation(pdb.topology, system, integrator, platform)
    else:
      platform = Platform.getPlatformByName('CPU')
      simulation = Simulation(pdb.topology, system, integrator, platform)
  simulation.context.setPositions(pdb.positions)
  if restart:
    simulation.loadCheckpoint('chkptfile.chk')
  if save_chkpt_file:
    simulation.reporters.append(CheckpointReporter('chkptfile.chk', chkpt_freq))
  simulation.reporters.append(StateDataReporter(stdout, outfreq, step=True))
  simulation.step(nstep)
  positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
  PDBFile.writeFile(simulation.topology, positions, open(f'final_{index}.pdb', 'w'))

    
#Functions for CSP demo only

def triginvert(x,sinx,cosx):
  if cosx<0:
    if sinx>0:
      x=np.pi-x
    elif sinx<0:
      x=-np.pi-x

  return x

def getTrp8(CVs):
  sinx=CVs[:,12]
  cosx=CVs[:,13]
  x=np.arcsin(sinx)
  chi1=  chi2=[triginvert(a,b,c) for (a,b,c) in zip(x,sinx,cosx)]

  sinx=CVs[:,116]
  cosx=CVs[:,117]
  x=np.arcsin(sinx)
  chi2=[triginvert(a,b,c) for (a,b,c) in zip(x,sinx,cosx)]

  return np.asarray(chi1),np.asarray(chi2)
