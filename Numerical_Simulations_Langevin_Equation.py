# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:41:44 2023

@author: antoine.berut
"""

import numpy as np
import matplotlib.pyplot as plt

def colored_noise_simu(n_traj=1,dt=1e-5,duration=1,alpha=1,fc=100):
    '''
    colored_noise_simu(n_traj=1,dt=1e-5,duree=1,alpha=1,fc=100)
    
    Generates a colored noise X by integrating with Heun scheme [1] the overdamped Langevin equation:
        dX/dt = - wc * X + \sqrt(2*alpha*wc) * eta
    where wc = 2*pi*fc and eta is a Gaussian white noise with variance equal to 1
    
    Input parameters :
        - n_traj : number of noise samples one wants to create (int)
        - dt : time step for noise samples (s)
        - duration : duration of each sample (s)
        - alpha : finale variance of the colored noises
        - fc : cut-off frequency of the colored noises
        
    Output :
        - X : array of dimension "n_traj x int(duree/dt)" that contains n_traj independent samples of colored noise with the chosen duration
        
    [1] Mannella, R. (2002). Integration of stochastic differential equations on a computer. International Journal of Modern Physics C, 13(09), 1177-1194.
    '''
    
    wc = 2*np.pi*fc
    nb_time_increment=int(np.round(duration/dt))
    
    thermal_noise=np.sqrt(2*alpha*wc/dt)*np.random.randn(n_traj,nb_time_increment)
    
    traj=np.zeros((n_traj,nb_time_increment))
    traj_Euler=np.zeros((n_traj,nb_time_increment))
    for i_time in range(nb_time_increment-1):
        #Euler prediction
        traj_Euler[:,i_time+1] = traj[:,i_time] + dt * ( - wc * traj[:,i_time] + thermal_noise[:,i_time])
        #Heun prediction
        traj_loc = traj[:,i_time] + dt * (0.5*(- wc * traj[:,i_time] - wc * traj_Euler[:,i_time+1]) + thermal_noise[:,i_time])
        #Update trajectories
        traj[:,i_time+1] = traj_loc
    return traj

def Brownian_simu(nb_part=100,duration=1,dt=1e-5,x_0=[],r=1.5e-6,eta=1e-3,k_B=1.381e-23,T=298.15,ext_F=lambda x:np.zeros_like(x),ext_noise=[],on_plot=False,nb_plot=10):
    """
    Brownian_simu(nb_part=100,duration=1,dt=1e-5,x_0=[],r=1.5e-6,eta=1e-3,k_B=1.381e-23,T=298.15,ext_F=lambda x:np.zeros_like(x),ext_noise=[],on_plot=False,nb_plot=10)
    
    Computes the 1D trajectory of a Brownian particle, by integrating with Heun scheme [1] the overdamped Langevin equation:
        gamma * dX/dt = external_force + external_noise + thermal_noise
    where: 
    - X is the particle's position
    - t is the time
    - gamma is the Stokes drag coefficient (gamma = 6*pi*R*eta with R the particle's radius and eta the fluid's viscosity)
    - external_force is a deterministic function of X
    - external_noise is a term that may vary with time but is independent of X
    - thermal_noise is a Gaussian white noise accounting for the thermal agitation in the fluid
    
    Input parameters are separated in three categories:
    1) Simulation parameters:
        - nb_part: number of trajectories one wants to create (int)
        - duration: duration of each trajectory (s)
        - dt: time step for the numerical integration (s)
        - x_0: initial position for each trajectory (array of dimension "nb_part x 1", in m)
    2) Physical parameters:
        - r: particle's radius (m)
        - eta: fluid's viscosity (Pa.s)
        - k_B: Boltzmann constant (J/K)
        - T: temperature (K)
        - ext_F: external force function (must be a function of X that accepts an array of dimension "nb_part x 1" as an input and returns an array of same dimension as an output)
        - ext_noise: external noise values (must be an array of dimension "nb_part x np.round(duration/dt)")
    3) Graphical parameters:
        - on_plot: boolean, if True the script automatically plot a given number of trajectories (X as a function of t)
        - nb_plot: number of trajectories to plot if "on_plot" is True (int)
        
    All input parameters have default values, and therefore all can be considered as optional parameters.
    The default values for the parameters are:
        - nb_part: 100
        - duration: 1 (s)
        - dt: 1e-5 (s)
        - x_0: [] (initial position is X = 0 for all trajectories)
        - r: 1.5e-6 (m)
        - eta: 1e-3 (Pa.s)
        - k_B: 1.381e-23 (J/K)
        - T: 298.15 (K)
        - ext_F: lambda z:np.zeros_like(z) (no external force)
        - ext_noise: [] (no external noise)
        - on_plot: False
        - nb_plot: 10
    
    Outputs:
        - x_pos: array of dimension "nb_part x np.round(duration/dt)" containing nb_part independent trajectories of the chosen duration
        - time: array of dimension "1 x np.round(duration/dt)" containing the time vector (common to all trajectories)

    [1] Mannella, R. (2002). Integration of stochastic differential equations on a computer. International Journal of Modern Physics C, 13(09), 1177-1194.
    """
    #Physical parameters
    gamma = 6*np.pi*eta*r
    nb_time_increment=int(np.round(duration/dt))
    time=np.arange(nb_time_increment)*dt
    
    #Noises
    thermal_noise=np.sqrt(2*gamma*k_B*T/dt)*np.random.randn(nb_part,nb_time_increment)
    if np.size(ext_noise)==0:
        colored_noise=np.zeros_like(thermal_noise)
    elif np.shape(ext_noise)==(nb_part,nb_time_increment):
        colored_noise=ext_noise
    else:
        print('External noise has inconsistent dimension with number of particles and time duration.')
        return
    
    #Position
    x_pos=np.zeros((nb_part,nb_time_increment))
    if np.size(x_0)==nb_part:
        x_pos[:,0]=x_0
    elif np.size(x_0)>0:
        print('Initial positions has inconsistent dimension with number of particles')
        return
    x_pos_Euler=np.zeros((nb_part,nb_time_increment))

    for i_time in range(nb_time_increment-1):
        #Euler prediction
        x_pos_Euler[:,i_time+1] = x_pos[:,i_time] + dt/gamma * (ext_F(x_pos[:,i_time]) + thermal_noise[:,i_time] + colored_noise[:,i_time])
        #Heun prediction
        x_pos_loc = x_pos[:,i_time] + dt/gamma * (0.5*(ext_F(x_pos[:,i_time]) + ext_F(x_pos_Euler[:,i_time+1])) + thermal_noise[:,i_time] + colored_noise[:,i_time])
        #Update trajectories
        x_pos[:,i_time+1] = x_pos_loc
        
    #Graphical output
    if on_plot==True:
        nb_traj_plot_max=nb_plot
        fig, ax = plt.subplots()
        for i in range(np.minimum(nb_part,nb_traj_plot_max)):
            ax.plot(time,x_pos[i,:]*1e6)
        ax.set_title('Trajectories')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (µm)')
    
    return x_pos , time