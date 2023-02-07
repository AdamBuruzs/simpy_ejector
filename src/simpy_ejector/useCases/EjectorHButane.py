#  Copyright (c) 2023.   Adam Buruzs
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! python -m site
### installing package in jupyter notebook
# import sys
# !{sys.executable} -m pip install C:\Users\BuruzsA\PycharmProjects\flows1d\dist\flows1d-1.0.0b1-py3-none-any.whl


# In[2]:


#help("modules")


# In[3]:


from simpy_ejector import nozzleSolver,NozzleParams, EjectorGeom, nozzleFactory, refProp, EjectorMixer, numSolvers


# In[4]:


import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import importlib
import logging,sys

logging.basicConfig(stream = sys.stdout, level = logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# get_ipython().run_line_magic('matplotlib', 'notebook')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 140)
np.core.arrayprint._line_width = 180
np.set_printoptions(linewidth= 180)


# In[5]:


nozzle = nozzleFactory.ConicConic(Rin = 1.0, Lcon=2.905, Rt = 0.2215, Ldiv = 1.4116, Rout = 0.345 )
nozzle.setFriction( 1.0e-2 )
ejector = EjectorGeom(nozzle, Dm = 1.4 )
ejector.setMixer(mixerstart=5.514, mixerLen=11.2, diffuserLen=25.2, diffuserHeight=1.80)
ejectorPlot = ejector.draw()


# In[6]:


## Butane ejector for heat pump see Schlemminger article
pin = 2140.0  # kPa  2000
Tin = 273.15 + 114.0  # Kelvin  380
Tsuc =  52.7 + 273.15 #  suction temperature
Psuc =  430 # suction pressure kPa
fluid = "BUTANE"
RP = refProp.setup(fluid)
[Din, hin] = refProp.getDh_from_TP(RP, Tin, pin)
[DinSuc, hinSuc] = refProp.getDh_from_TP(RP, Tsuc, Psuc)
hin, hinSuc


# In[7]:


nsolver = nozzleSolver.NozzleSolver(nozzle, fluid, 1, solver="AdamAdaptive", mode="basic")
nsolver.setFriction(1e-2)
vin_crit = nsolver.calcCriticalSpeed(pin, hin, 0.1, maxdev= 1e-3, chokePos="divergent_part" )


# In[8]:


vin_crit

sol_until_throat = nsolver.solveAdaptive1DBasic(vin_crit, pin, hin, 0.0, nsolver.nozzle.xt)
# In[9]:
dv_kick = 2.0
dp_kick = nsolver.pFromV_MassConst(v = sol_until_throat.iloc[-1]["v"], dv = dv_kick, p = sol_until_throat.iloc[-1]["p"], h = sol_until_throat.iloc[-1]["h"])
print(f"mass conserving artificial kick: dv = {dv_kick} m/s, dp = {dp_kick} kPa")

res_crit = nsolver.solveKickedNozzle(vin_crit, pin , hin, kicks = {'v': dv_kick , 'p': -dp_kick})
nsolver.plotsol(res_crit, title = "choked nozzle with friction = {} ".format(nsolver.frictionCoef))


nozzleExit =  res_crit.iloc[-1]
nozzleExit


# In[12]:


plt.figure()
plt.plot(res_crit['x'], res_crit['v'])
plt.plot(res_crit['x'], res_crit['c'])
plt.legend(["velocity", "speed of sound"])


# In[13]:


massFlowPrim = nozzleExit['v']* nozzleExit['d']* nozzle.Aprofile(nozzle.L) * 1e-4
massFlowPrim # in kg/sec


# Mixing calculations
# set up the mixer properties. TODO: what are the reasonable ranges?

mixingParams = { 'massExchangeFactor' : 1.e-4, 'dragFactor': 0.003, 'frictionInterface': 0.001, 'frictionWall': 0.015}
mixer = EjectorMixer.EjectorMixer(fluid, ejector, mixingParams)


calcSucMassFlow = True
if calcSucMassFlow == False:
    # Single Choking calculation. With externally set suction mass flow rate:
    MFlowSuc = 58.0
    mixer.setSuctionMassFlow(MFlowSuc) # set the suction mass flow rate in [g/s]b

    entrainment = mixer.massFlowSecond / 1000.0 / massFlowPrim
    entrainment

    preMix = mixer.premixWrapSolve(res_crit, Psuc ,Tsuc)
    preMix
    mixerinput = [ preMix[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]

    solMix = mixer.solveMix(mixerinput)
else:
    mixer.setSuctionMassFlow(None)
    mixer.setSingleChoke(True)
    mixer.ejector.Asi = 2* 1.1**2 * math.pi  # cm2 of the suction nozzle inlet.
    #mixer.ejector.Asi = 2*3.14*3.25*2.2
    preMix = mixer.premixWrapSolve(res_crit, Psuc ,Tsuc)
    preMix
    print(f"calculated secondary mass flow rate {preMix['massFlowSecond']} g/s")
    mixerinput = [ preMix[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]
    solMix = mixer.solveMix(mixerinput)


print(solMix)


mixer.plotSolution(res_crit, solMix, "single choking mode",  ejectorPlot)


Pdiffout = solMix.iloc[-1]['p']


Pdiffout


# Ejector Pressure Lift in kPa

Pdiffout - Psuc


# Ejector efficiency:

diffusorLast =  solMix.iloc[-1]
diffusorLast


# We need to calculate the average spec enthalpy, for that we need the mass flow rates

# In[28]:


primState =  refProp.getTD(RP, diffusorLast['hp'] , diffusorLast['p'])
secState = refProp.getTD(RP, diffusorLast['hs'] , diffusorLast['p'])
primState, secState


# The mass flow rates:

# In[29]:


massFPrim = primState['D'] * diffusorLast['vp']* diffusorLast['Ap']* 1e-4
massFSec = primState['D'] * diffusorLast['vs']* diffusorLast['As']* 1e-4
massFPrim , massFSec


# In[30]:


hAverage = (massFPrim * diffusorLast['hp'] + massFSec * diffusorLast['hs'] )/ (massFPrim + massFSec )
hAverage


# In[31]:


hin,hinSuc,Psuc


# In[32]:


efficiency = mixer.calcEfficiency(pin, Tin, Psuc, Tsuc, Pdiffout, massFPrim , massFSec )


# Ejector efficiency according to Elbel:

# In[33]:


efficiency

def specPlotSolution(ejector, RP, solNozzle, solMix, title="", ejectorPlot=True,
                     pressureExp = None):
    """ Plot the results of solveMix function
    :param solNozzle: solution of the nozzle
    :param solMix: the solution dataframe with data that we get from the solveMix function
    :return:
    """
    fig = plt.figure(figsize=[11, 8])
    fig.suptitle("results of the 1D flow ejector simulation \n " + title)
    if (ejectorPlot):
        plt.subplot(311)
        # plt.plot(ejectorPlot)
        nozzleWall = [ejector.nozzle.Rprofile(xi) for xi in solNozzle['x']]
        mixerWall = [ejector.mixerR(xi) for xi in solMix['x']]
        plt.plot(solNozzle['x'], nozzleWall)
        plt.plot(solMix['x'], np.sqrt(solMix['Ap'] / math.pi))
        plt.plot(solMix['x'], mixerWall)
        plt.legend(['wall of Nozzle', 'primary motive stream', 'wall of Mixing region'])
        plt.ylabel("R [cm]")

    if ejectorPlot:
        plt.subplot(312)
    else:
        plt.subplot(211)
    if pressureExp is None:
        plt.plot(solNozzle['x'], solNozzle['p'], color = 'C0')
        plt.plot(solMix['x'], solMix['p'], color='C0')
        # speeds
        plt.plot(solNozzle['x'], solNozzle['v'], color="#aa1111")
        plt.plot(solMix['x'], solMix['vp'], color="#aa1111")
        plt.plot(solMix['x'], solMix['vs'], color="#aa9911")
        ## enthalpy
        plt.plot(solMix['x'], solMix['hp'])
        plt.legend(['pressure [kPa]', 'p [kPa]',
                    'prim flow speed', 'prim flow speed', 'suction flow speed',
                    'hp'])
        plt.yscale("log")
        plt.xlabel("x [cm]")
    else:
        plt.plot(solNozzle['x'], solNozzle['p'], color='C0')
        plt.plot(solMix['x'], solMix['p'], color='C0')
        plt.plot(pressureExp['x'], pressureExp['p'], color = 'red', marker = 'x', linestyle = '')
        plt.legend(['pressure simulation [kPa]', '',
                    'pressure Experiments [kPa]' ] )
        plt.xlabel("x [cm]")
        plt.ylabel("kPa")
        plt.yscale("log")
        plt.grid(True, axis = 'y')
    ## Mach numbers :
    if ejectorPlot:
        plt.subplot(313)
    else:
        plt.subplot(212)
    plt.plot(solNozzle['x'], solNozzle['mach'], color = 'C0')
    cPrim = [refProp.getSpeedSound(RP, solMix.iloc[i]['hp'], solMix.iloc[i]['p']) for i in range(solMix.__len__())]
    cSec = [refProp.getSpeedSound(RP, solMix.iloc[i]['hs'], solMix.iloc[i]['p']) for i in
            range(solMix.__len__())]
    plt.plot(solMix['x'], solMix['vp'] / cPrim, color = 'C0')
    plt.plot(solMix['x'], solMix['vs'] / cSec, color = 'C1' )
    plt.legend(['primary Mach', 'primary Mach', 'secondary Mach'])
    plt.tight_layout()
    plt.axvline( x = ejector.nozzle.xt, linestyle = '--', color = '#444444' )
    plt.text(ejector.nozzle.xt, 0.0, 'motive nozzle throat', va = 'bottom', rotation = 90, alpha = 0.6)
    # text1 = plt.annotate('motive nozzle throat', xy=(ejector.nozzle.xt, 0.3), va='top', rotation = 90)
    # text1.set_alpha(.7)
    mixend = ejector.ejectorlength - ejector.diffuserLen
    plt.axvline( x = mixend, linestyle = '--', color = '#444444' )
    text2 = plt.annotate('diffuser', xy=(mixend, 0), va='bottom',alpha = 0.6)
    # text2.set_alpha(.7)

pressureExp = pd.DataFrame( {'x': [ 5.5, 11.1, 16.7, 29.3, 41.9],
                              'p' : [ 404.4, 461.7, 520.2, 579.2, 584.5]})
specPlotSolution(ejector, RP, res_crit, solMix, "single choking mode",  False, pressureExp)