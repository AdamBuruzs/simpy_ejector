# The *simpy_ejector* python package

## fast 1 dimensional python simulator for ejectors

A 'simple' and fast 1 dimensional computational fluid dynamics simulator
made for simulating Ejectors for heat pump and
refrigerator applications. It includes modules for
simulating two phase supersonic nozzles, that are crucial parts
of supersonic ejectors.

Using this package you can easily calculate mass flow-rates of motive and suction nozzles of ejectors with 
the user defined ejector geometry and fluid specification. 
The provided functions solve simplified differential-algebraic equations of the fluid flow (mass, momentum and energy equations).
This package calculates and plots speed, pressure, quality, etc. profiles along the ejector axis. 
And lets you easily calculate ejector efficiency. 

You can play with the primary nozzle and ejector geometry, and update the calculation results with any fluid from the 
Refprop (https://www.nist.gov/srd/refprop) library. This makes it an ideal tool for a rough and fast ejector dimensioning.
You can calculate and evaluate with ease dozens of ejector geometries within minutes. And you know it, that it replaces hours and days 
of heavy calculations on commercial software packages!

This module can be used for 1 component, one or two phase flow simulations.
In the first version it applies the homogeneus equilibrium model (HEM) to calculate
material properties. (For the thermodynamic properties it needs the RefProp 9 or newer to be installed - not tested for older versions)  


For a short description of the method see the pdf in the Docs directory: 
[/Docs/](./Docs/Ejector_Buruzs.pdf) <br>
The english description of the method will be published soon, and you must reference to it if you use this package in any publication! 


To see, how it works, look into the jupyter notebooks located in the directory:
[src/simpy_ejector/useCases/](./src/simpy_ejector/useCases/EjectorMixerSingleShocking.ipynb)

Don't forget, that flows1d requires you to install the python-Refprop package ctREFPROP:
https://pypi.org/project/ctREFPROP/.
In windows you need to have the refprop.dll installed (for the development I have used version 9.1)
and set the environment variable 'RPprefix' to the path of the 'refprop.dll' file. 
For example if your REFPROP is installed into "C:/Program Files (x86)/REFPROP/", then open a command window, and type:
```
setx RPprefix "C:/Program Files (x86)/REFPROP/"
```
(or set the RPprefix user-environment variable  through windows menu)

**For the latest Documentation, Demo, Tutorials and Examples visit the project page on github:
https://github.com/AdamBuruzs/simpy_ejector**

## Installation

````
pip install simpy-ejector
````
And set the environment variable 'RPprefix' to the path of the 'refprop.dll' file. 
For example : "C:/Program Files (x86)/REFPROP/"

## Examples

simulation result for an ejector (with a simulated normal shock wave)
![simulation result for an ejector](./src/simpy_ejector/charts/ShockInMixer.png)

Pressure profiles in a nozzle with a simulated shock-wave.
![pressure profiles in a nozzle with shock-wave](./src/simpy_ejector/charts/Pressure_profiles_Nozzle.png)

Pressure profile comparison with experiment (in this case without shock-waves):

![pressure profiles comparison with experiment](./src/simpy_ejector/charts/PressureExp.png)