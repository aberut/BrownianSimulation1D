# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:41:44 2023

@author: antoine.berut
"""

import numpy as np
import matplotlib.pyplot as plt
from inspect import signature

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

def Brownian_simu(nb_part=100,duration=1,dt=1e-5,x_0=[],r=1.5e-6,eta=1e-3,k_B=1.381e-23,T=298.15,ext_F=lambda x=[]:np.zeros_like(x),F_params=[],ext_noise=[],on_plot=False,nb_plot=10):
    """
    Brownian_simu(nb_part=100,duration=1,dt=1e-5,x_0=[],r=1.5e-6,eta=1e-3,k_B=1.381e-23,T=298.15,ext_F=lambda x=[]:np.zeros_like(x),F_params=[],ext_noise=[],on_plot=False,nb_plot=10)
    
    Computes the 1D trajectory of a Brownian particle, by integrating with Heun scheme [1] the overdamped Langevin equation:
        gamma * dX/dt = F(X,params) + G(t) + thermal_noise
    where: 
    - X is the particle's position
    - t is the time
    - gamma is the Stokes drag coefficient (gamma = 6*pi*R*eta with R the particle's radius and eta the fluid's viscosity)
    - F(X, params) is an external_force, which is a deterministic function of X and some optional parameters (that may themselves be time dependent)
    - G(t) is also an external force (called "external noise" to avoid confusion), which only varies with time but is independent of X
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
        - ext_F: external force function (must be a function that takes at least an array X of dimension "nb_part x 1" as an input, and returns an array of same dimension as an output)
        - F_params: in the case where ext_F takes two inputs (ext_F(X,params)), values of the parameters for the function ext_F at each time step (must be an array of dimension "nb_params x np.round(duration/dt)")
        - ext_noise: external noise values (must be an array of dimension "nb_part x np.round(duration/dt)")
    3) Graphical parameters:
        - on_plot: boolean, if True the script automatically plot a given number of trajectories (X as a function of t)
        - nb_plot: number of trajectories to plot if "on_plot" is True (int)
    
    Note: The main difference between using ext_F with time dependent parameters F_params, and using ext_noise, is that ext_F will have the same value for every trajectory that is computed (F_params is provided for each time step, but is the same for each trajectory),
    while ext_noise can be provided with a different value for each trajectory that is computed at each time step. See detailed examples for more information.
    
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
        - ext_F: lambda x: np.zeros_like(x) (no external force)
        - F_params: [] (no external parameter for the force ext_F)
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
    
    #Thermal Noise
    thermal_noise=np.sqrt(2*gamma*k_B*T/dt)*np.random.randn(nb_part,nb_time_increment)
    
    #External force
    check_params=len(signature(ext_F).parameters) #check the numbers of input parameters that the function ext_F takes
    if check_params==1:
        def ext_F_params(x,params=[]):
            return ext_F(x)
        F_params=np.zeros_like(thermal_noise)
    elif check_params==2:
        ext_F_params=ext_F
        if np.ndim(F_params)==0:
            F_params=np.zeros_like(thermal_noise)
        elif np.ndim(F_params)==1:
            if np.size(F_params)==nb_time_increment:
                F_params=np.reshape(F_params,(1,int(nb_time_increment)))
            else:
                print('Error: external force parameters F_params has inconsistent dimension with time duration (you must provide a value of the parameters for each time step).')
                return None, None
        elif np.ndim(F_params)==2:
            if np.shape(F_params)[1]!=nb_time_increment:
                print('Error: external force parameters F_params has inconsistent dimension with time duration (you must provide a value of the parameters for each time step, as an array of dimension "nb_params x np.round(duration/dt)").')
                return None, None
        else:
            print('Error: external force parameters F_params has inconsistent dimension (the function F_ext should have only two inputs, F_params should be an array of dimension "nb_params x np.round(duration/dt)").')
            return None, None
    else:
        print('Error : the external force function ext_F does not have the correct amount of input parameters (it should have either one input parameter "X", or two input parameters "X" and "params").')
        return None, None
    
    #External noise
    if np.size(ext_noise)==0:
        colored_noise=np.zeros_like(thermal_noise)
    elif np.ndim(ext_noise)==1:
        if np.size(ext_noise)==nb_part*nb_time_increment:
            colored_noise=np.reshape(ext_noise,(1,int(nb_part)*int(nb_time_increment)))
        else:
            print('Error: external noise ext_noise has inconsistent dimension with number of particles and time duration (you must provide a value of ext_noise for each time step).')
            return None, None
    elif np.shape(ext_noise)==(nb_part,nb_time_increment):
        colored_noise=ext_noise
    else:
        print('Error: external noise ext_noise has inconsistent dimension with number of particles and time duration (it should be an array of dimension "nb_part x np.round(duration/dt)").')
        return None, None
    
    #Position
    x_pos=np.zeros((nb_part,nb_time_increment))
    if np.size(x_0)==nb_part:
        x_pos[:,0]=x_0
    elif np.size(x_0)>0:
        print('Initial positions x0 has inconsistent dimension with number of particles (it should be an array of dimension "nb_part x 1").')
        return
    x_pos_Euler=np.zeros((nb_part,nb_time_increment))

    for i_time in range(nb_time_increment-1):
        #Euler prediction
        x_pos_Euler[:,i_time+1] = x_pos[:,i_time] + dt/gamma * (ext_F_params(x_pos[:,i_time],F_params[:,i_time]) + thermal_noise[:,i_time] + colored_noise[:,i_time])
        #Heun prediction
        x_pos_loc = x_pos[:,i_time] + dt/gamma * (0.5*(ext_F_params(x_pos[:,i_time],F_params[:,i_time]) + ext_F_params(x_pos_Euler[:,i_time+1],F_params[:,i_time+1]) + colored_noise[:,i_time] + colored_noise[:,i_time+1]) + thermal_noise[:,i_time])
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