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

import logging
import sys, os
import math

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import matplotlib.pyplot as plt
import pandas as pd
# sys.path.append("../..")
from simpy_ejector.useCases import ejectorSimulator
from simpy_ejector import refprop_material, materialFactory

# load Refprop for your fluid:
fluid = "Butane"
# RP = refProp.setup(fluid)
# RProps = refprop_material.MaterialProperties(fluid)
RProps = materialFactory.MaterialPropertiesFactory.create(material=fluid, library='coolprop')

# # set up geometry and flow state input parameters:
# nozzle = nozzleFactory.ConicConic(Rin=1.0, Lcon=2.905, Rt=0.2215, Ldiv=1.4116, Rout=0.345)
# nozzle.setFriction(1.0e-2)
# ejector = EjectorGeom(nozzle, Dm=1.4)
# ejector.setMixer(mixerstart=5.514, mixerLen=11.2, diffuserLen=25.2, diffuserHeight=1.80)
# ejectorPlot = ejector.draw()

## Butane ejector for heat pump see Schlemminger article
pin = 2140.0  # kPa  2000
Tin = 273.15 + 114.0  # Kelvin  380
Tsuc = 52.7 + 273.15  # suction temperature
Psuc = 430  # suction pressure kPa

# Lcon = (Rin - Rt) / math.tan(gamma_conv * math.pi / 180)
[Din, hinPrim] = RProps.getDh_from_TP(Tin, pin)
[DinSuc, hinSuc] = RProps.getDh_from_TP(Tsuc, Psuc)

diffuserHeight = 1.80
diffuserLen = 25.2
Dmix = 1.4
# (diffuserHeight - Dmix/2)/diffuserLen
gamma_diffusor = math.atan((diffuserHeight - Dmix/2)/diffuserLen) * 180/ math.pi # degree
R_suc = 1.1 # cm
A_suction = 2* R_suc**2 * math.pi  # cm2 of the suction nozzle inlet.

params = {"Rin": 1.0, "Rt": 0.2215, "Rout": 0.345, "Dmix": Dmix,
          "Lcon": 2.905, "Ldiv": 1.4116,  # "gamma_conv": 15.0, "gamma_div" : 6.0,
          "Pprim": pin, "hprim": hinPrim, "Psuc": Psuc, "hsuc": hinSuc,
          "Tprim" : Tin, "Tsuc" : Tsuc,
          "A_suction_inlet": A_suction,
          "mixerLen": 11.2, "gamma_diffusor": gamma_diffusor, "diffuserLen": diffuserLen }
## calculate Temperatures from specific enthalpy with Refprop:
# primQ = RProps.getTD(hm=params["hprim"], P=params["Pprim"])
# params["Tprim"] = primQ['T']
# params["Tsuc"] = RProps.getTD(hm=params["hsuc"], P=params["Psuc"])['T']

# set parameters of the mixing calculations:
# params["mixingParams"] = {'massExchangeFactor': 2.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.0,
#                         'frictionWall': 0.0015}
params["mixingParams"] = {'massExchangeFactor': 1.e-4, 'dragFactor': 0.003, 'frictionInterface': 0.001,
                          'frictionWall': 0.015}

# create a simulator object:
esim = ejectorSimulator.ejectorSimu(params, fluid=fluid, proplibrary="coolprop")
# plot the ejector geometry:
ejplot = esim.ejector.draw()
## calculate the primary mass flow rate:
esim.calcPrimMassFlow()

## calculate the critical (= choked flow) solution in the motive nozzle:
res_nozzle = esim.motiveSolver()
print(f"By the motive nozzle exit:\n {res_nozzle.iloc[-1]}")
# solve the pre-mixing equations:
esim.premix(res_nozzle)  # this sets the mixer, that is needed for the mixing calculation
print(f"suction MFR [g/s] {round(esim.mixerin['massFlowSecond'], 3)}")

# solve the mixer equations until the ejector outlet
esim.mixersolve()

esim.massFlowCheck()
outletValues = esim.solMix.iloc[-1].to_dict()  # .transpose()

average_h = (outletValues["MFRp"] * outletValues["hp"] + outletValues["MFRs"] * outletValues["hs"]) / (
            outletValues["MFRp"] + outletValues["MFRs"])
print(
    f"Ejector outlet specific enthalpy {round(average_h, 3)} kJ/kg, pressure {round(outletValues['p'] * 1e-2, 3)} bar,"
    f" vapor quality {round(outletValues['q_average'], 3)}  ")

esim.plotMixSolution(res_nozzle, esim.solMix, "simpy_ejector 1D Ejector flow solution")

efficiency = esim.calcEfficiency()
print(f"Elbel Efficiency = {efficiency}")

def specPlotSolution(esim : ejectorSimulator.ejectorSimu, title="", ejectorPlot=True,
                     pressureExp = None):
    """ Plot the results of solveMix function
    :param solNozzle: solution of the nozzle
    :param solMix: the solution dataframe with data that we get from the solveMix function
    :param pressureExp: experimental data
    :return:
    """
    fig = plt.figure(figsize=[11, 8])
    fig.suptitle("results of the 1D flow ejector simulation \n " + title)
    if (ejectorPlot):
        plt.subplot(311)
        # plt.plot(ejectorPlot)
        nozzleWall = [esim.ejector.nozzle.Rprofile(xi) for xi in esim.primNozzleFlow['x']]
        mixerWall = [esim.ejector.mixerR(xi) for xi in esim.solMix['x']]
        plt.plot(esim.primNozzleFlow['x'], nozzleWall)
        plt.plot(esim.solMix['x'], math.sqrt(esim.solMix['Ap'] / math.pi))
        plt.plot(esim.solMix['x'], mixerWall)
        plt.legend(['wall of Nozzle', 'primary motive stream', 'wall of Mixing region'])
        plt.ylabel("R [cm]")

    if ejectorPlot:
        plt.subplot(312)
    else:
        plt.subplot(211)
    if pressureExp is None:
        plt.plot(esim.primNozzleFlow['x'], esim.primNozzleFlow['p'], color = 'C0')
        plt.plot(esim.solMix['x'], esim.solMix['p'], color='C0')
        # speeds
        plt.plot(esim.primNozzleFlow['x'], esim.primNozzleFlow['v'], color="#aa1111")
        plt.plot(esim.solMix['x'], esim.solMix['vp'], color="#aa1111")
        plt.plot(esim.solMix['x'], esim.solMix['vs'], color="#aa9911")
        ## enthalpy
        plt.plot(esim.solMix['x'], esim.solMix['hp'])
        plt.legend(['pressure [kPa]', 'p [kPa]',
                    'prim flow speed', 'prim flow speed', 'suction flow speed',
                    'hp'])
        plt.yscale("log")
        plt.xlabel("x [cm]")
    else:
        plt.plot(esim.primNozzleFlow['x'], esim.primNozzleFlow['p'], color='C0')
        plt.plot(esim.solMix['x'], esim.solMix['p'], color='C0')
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
    plt.plot(esim.primNozzleFlow['x'], esim.primNozzleFlow['mach'], color = 'C0')
    cPrim = [esim.fluid.getSpeedSound( esim.solMix.iloc[i]['hp'], esim.solMix.iloc[i]['p']) for i in range(esim.solMix.__len__())]
    cSec = [esim.fluid.getSpeedSound( esim.solMix.iloc[i]['hs'], esim.solMix.iloc[i]['p']) for i in
            range(esim.solMix.__len__())]
    plt.plot(esim.solMix['x'], esim.solMix['vp'] / cPrim, color = 'C0')
    plt.plot(esim.solMix['x'], esim.solMix['vs'] / cSec, color = 'C1' )
    plt.legend(['primary Mach', 'primary Mach', 'secondary Mach'])
    plt.tight_layout()
    plt.axvline( x = esim.ejector.nozzle.xt, linestyle = '--', color = '#444444' )
    plt.text(esim.ejector.nozzle.xt, 0.0, 'motive nozzle throat', va = 'bottom', rotation = 90, alpha = 0.6)
    # text1 = plt.annotate('motive nozzle throat', xy=(ejector.nozzle.xt, 0.3), va='top', rotation = 90)
    # text1.set_alpha(.7)
    mixend = esim.ejector.ejectorlength - esim.ejector.diffuserLen
    plt.axvline( x = mixend, linestyle = '--', color = '#444444' )
    text2 = plt.annotate('diffuser', xy=(mixend, 0), va='bottom',alpha = 0.6)
    # text2.set_alpha(.7)

# measured experimental data
pressureExp = pd.DataFrame( {'x': [ 5.5, 11.1, 16.7, 29.3, 41.9],
                              'p' : [ 404.4, 461.7, 520.2, 579.2, 584.5]})

specPlotSolution( esim, "single choking mode",  False, pressureExp)