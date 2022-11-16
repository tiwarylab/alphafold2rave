#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bvani
"""
import numpy as np

def RegSpaceClustering(z, min_dist, max_centers=200, batch_size=100):
    '''Regular space clustering.
    Args:
        data: ndarray containing (n,d)-shaped float data
        max_centers: the maximum number of cluster centers to be determined, integer greater than 0 required
        min_dist: the minimal distances between cluster centers
    '''
    num_observations, d = z.shape
    p = np.hstack((0,np.random.permutation(num_observations-1)+1))
    data = z[p]
    center_list = data[0, :].copy().reshape(d,1)
    centerids=[p[0]]
    i = 1
    while i < num_observations:
        x_active = data[i:i+batch_size, :]
        distances = np.sqrt((np.square(np.expand_dims(center_list.T,0) - np.expand_dims(x_active,1))).sum(axis=-1))
        indice = tuple(np.nonzero(np.all(distances > min_dist, axis=-1))[0])
        if len(indice) > 0:
            # the first element will be used
            #print(center_list.shape,x_active.shape,x_active[indice[0]].shape)
            center_list = np.hstack((center_list, x_active[indice[0]].reshape(d,1)))
            centerids.append(p[i+indice[0]])
            i += indice[0]
        else:
            i += batch_size
        if len(center_list) >= max_centers:
            print("Exceed the maximum number of cluster centers!\n")
            print("Please increase dmin!\n")
            raise ValueError
    return center_list,centerids


def run_unbiased(on_gpu,plumedfile,dt,temp,freq,nstep,index):
    use_plumed=True
    outfreq = 0
    chkpt_freq=0
    
    t1 = time.perf_counter()

    os.chdir("/content/test_MD/")
    gro = GromacsGroFile('pred_%i.gro'%index)
    top = GromacsTopFile('topol.top', \
                         periodicBoxVectors=gro.getPeriodicBoxVectors(), \
                         includeDir='/content/Plumed-on-OpenMM-GPU/gromacsff')
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer, \
            switchDistance=1.0*nanometer,constraints=HBonds)

    #integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    #using NoseHooverIntegrator - Leapfrog integration
    integrator = NoseHooverIntegrator(temp*kelvin, freq/picosecond,
                                    dt*picoseconds);
    if use_plumed:
      fid=open(plumed_file,'r')
      ff=fid.read()
      force=PlumedForce(ff)
      system.addForce(force)

    if on_gpu:
      platform = Platform.getPlatformByName('CUDA')
      properties = {'Precision': 'double','CudaCompiler':'/usr/local/cuda/bin/nvcc'}
      simulation = Simulation(top.topology, system, integrator, platform)
    else:
      simulation = Simulation(top.topology, system, integrator, platform)

    simulation.context.setPositions(gro.positions)

    #simulation.minimizeEnergy()
    #simulation.reporters.append(PDBReporter('output.pdb', 1000))

    simulation.reporters.append(DCDReporter(outfname, outfreq))
    #simulation.reporters.append(StateDataReporter(stdout, outfreq, step=True,potentialEnergy=True, temperature=True))

    if save_chkpt_file:
      simulation.reporters.append(CheckpointReporter(chkpt_fname, chkpt_freq))

    #starts the MD simulation
    simulation.step(nstep)

    #timing the simulation
    t2 = time.perf_counter()
    print('\ntime taken to run:',(t2-t1)/60,' mins')

def make_biased_plumed(plumedfile,weights,colvar,height,biasfactor,width1,width2,gridmin1,gridmin2,gridmax1,gridmax2):
    f=open(plumedfile,"a")

    f.write("\n sigma1: COMBINE ARG=%s COEFFICIENTS=%s"%(colvar,weights[0]))
    f.write("\n sigma2: COMBINE ARG=%s COEFFICIENTS=%s"%(colvar,weights[1]))

    f.write("\nMETAD ...\n \
      LABEL=metad\n \
      ARG=sigma1,sigma2\n \
      PACE=1000 HEIGHT=%f TEMP=300\n \
      BIASFACTOR=%i\n \
      SIGMA=%f,%f\n \
      FILE=HILLS GRID_MIN=-%f,-%f GRID_MAX=%f,%f GRID_BIN=200,200\n \
      CALC_RCT RCT_USTRIDE=1000\n \
      ... METAD\n"%(height,biasfactor,width1,width2,gridmin1,gridmin2,gridmax1,gridmax2))


    f.close()
    
def run_biased(on_gpu,plumed_file,dt,temp,freq,nstep)
  
    use_plumed=True
    outfreq = 0
    chkpt_freq=0
    
    t1 = time.perf_counter()

    os.chdir("/content/test_MD/")
    gro = GromacsGroFile('pred_%i.gro'%index)
    top = GromacsTopFile('topol.top', \
                         periodicBoxVectors=gro.getPeriodicBoxVectors(), \
                         includeDir='/content/Plumed-on-OpenMM-GPU/gromacsff')
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer, \
            switchDistance=1.0*nanometer,constraints=HBonds)

    #integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    #using NoseHooverIntegrator - Leapfrog integration
    integrator = NoseHooverIntegrator(temp*kelvin, freq/picosecond,
                                    dt*picoseconds);
    if use_plumed:
      fid=open(plumed_file,'r') 
      ff=fid.read() 
      force=PlumedForce(ff) 
      system.addForce(force)

    if on_gpu:
      platform = Platform.getPlatformByName('CUDA')
      properties = {'Precision': 'double','CudaCompiler':'/usr/local/cuda/bin/nvcc'}
      simulation = Simulation(top.topology, system, integrator, platform)
    else:
      simulation = Simulation(top.topology, system, integrator, platform)

    simulation.context.setPositions(gro.positions)

    #simulation.minimizeEnergy()
    #simulation.reporters.append(PDBReporter('output.pdb', 1000))

    simulation.reporters.append(DCDReporter(outfname, outfreq))
    #simulation.reporters.append(StateDataReporter(stdout, outfreq, step=True,potentialEnergy=True, temperature=True))

    if save_chkpt_file:
      simulation.reporters.append(CheckpointReporter(chkpt_fname, chkpt_freq))

    #starts the MD simulation
    simulation.step(nstep)

    #timing the simulation
    t2 = time.perf_counter()
    print('\ntime taken to run:',(t2-t1)/60,' mins')


def scinvert(sinx,cosx):
  x=np.arcsin(sinx)
  if cosx<0:
    if sinx>0:
      x=np.pi-x
    elif sinx<0:
      x=-np.pi-x

  return x