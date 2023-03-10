# BrownianSimulation1D

[![DOI](https://zenodo.org/badge/587817112.svg)](https://zenodo.org/badge/latestdoi/587817112) [![CC BY 4.0][cc-by-shield]][cc-by]

A Python code that generates trajectories of Brownian particles, by integrating the overdamped [Langevin equation](https://en.wikipedia.org/wiki/Langevin_equation), following [Heun's method](https://en.wikipedia.org/wiki/Heun%27s_method).

The main function, called **Brownian_simu**, is available in the script [Numerical_Simulations_Langevin_Equation](./Numerical_Simulations_Langevin_Equation.py).  
It can be used to simulate 1D trajectories (position $x$ as a function of time $t$), of a Brownian particle that is submitted to an external force $F(x,t)$, in addition to the Gaussian white noise corresponding to the equilibrium thermal agitation $\xi$.  
The simulations is done by integrating the following overdamped Langevin equation:

$\gamma \frac{\mathrm{d}x}{\mathrm{d}t} = F_{ext}(x,t) + \xi(t)$

where, $\gamma$ is the Stokes friciton term (given by $\gamma= 6 \pi R \mu$ with $R$ the particle's radius, and $\mu$ the fluid's viscosity), and $\xi$ is a Gaussian white noise which verifies $\langle \xi(t) \rangle$ = 0 and $\langle \xi(t) \xi(t') \rangle = 2\gamma k_\mathrm{B}T$ $\delta(t-t')$ (with $k_\mathrm{B}$ the Botlzmann constant, $T$ the temperature, and $\delta$ the Dirac function).

The script also contains a function, called **colored_noise_simu** that also integrates an overdamped Langevin equation to generate exponentially correlated Gaussian noises, with a given variance $\alpha$ and correlation time $\tau_c$.

To use a functions, simply download the script in your working directory, then import in your Python scripts with:

```python
from Numerical_Simulations_Langevin_Equation import Brownian_simu
```

Both functions have a detailed help, available by typing:

```python
help(Brownian_simu)
```

# Tutorial

A step-by-step example is provided as a [Jupyter Notebook](./Examples.ipynb).

This example shows how to:

1. Compute the trajectories of free particles

2. Compute the trajectories of particles held in an optical trap

3. Compute the trajectories of particles submitted to a sinusoidal forcing

4. Compute the trajectories of particles held in an optical trap with a time-dependent trap stiffness

5. Compute the trajectories of particles held in an optical trap and submitted to a colored noise

*Note* : you can open this example with [Binder](https://mybinder.org/) to directly run it and interact with it in your internet browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aberut/BrownianSimulation1D/HEAD?labpath=Examples.ipynb)

# Example of use in scientific articles

These functions were used to compute the numerical results presented in the article Comment on "Harvesting information to control non-equilibrium states of active matter", [arXiv:2212.06825](https://arxiv.org/abs/2212.06825)

The complete code is provided as a [Jupyter Notebook](./arXiv-2212.06825.ipynb).

# Citation

If you would like to use this code in a scientific work, please cite as:  

> Antoine B??rut. aberut/BrownianSimulation1D: v1.9. (2023) [doi:10.5281/zenodo.7526269](https://doi.org/10.5281/zenodo.7526269)

# License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
