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
## this path is only needed for testing:
sys.path.append("../..")
from simpy_ejector.useCases import ejectorSimulator
from simpy_ejector import  materialFactory

# load Refprop for your fluid:
fluid = "Water"

#proplibrary = "refprop" # or "coolprop"
proplibrary = "coolprop"
# RP = refProp.setup(fluid)
# RProps = refprop_material.MaterialProperties(fluid)
RProps = materialFactory.MaterialPropertiesFactory.create(material=fluid, library=proplibrary)

# # set up geometry and flow state input parameters:
# nozzle = nozzleFactory.ConicConic(Rin=1.0, Lcon=2.905, Rt=0.2215, Ldiv=1.4116, Rout=0.345)
# nozzle.setFriction(1.0e-2)
# ejector = EjectorGeom(nozzle, Dm=1.4)
# ejector.setMixer(mixerstart=5.514, mixerLen=11.2, diffuserLen=25.2, diffuserHeight=1.80)
# ejectorPlot = ejector.draw()

## Exmple test case for a water ejector
pin = 1000 # kPa  
Psuc = 300  # suction pressure kPa

# Lcon = (Rin - Rt) / math.tan(gamma_conv * math.pi / 180)
# [Din, hinPrim] = RProps.getDh_from_TP(Tin, pin)
# [DinSuc, hinSuc] = RProps.getDh_from_TP(Tsuc, Psuc)
hinPrim = 750 
hinSuc = 2800

diffuserHeight = 1.80
diffuserLen = 25.2
Dmix = 1.4
# (diffuserHeight - Dmix/2)/diffuserLen
gamma_diffusor = math.atan((diffuserHeight - Dmix/2)/diffuserLen) * 180/ math.pi # degree
R_suc = 1.1 # cm
A_suction = 2* R_suc**2 * math.pi  # cm2 of the suction nozzle inlet.

params = {"Rin": 1.0, "Rt": 0.2, "Rout": 0.345, "Dmix": Dmix,
          "Lcon": 2.90, "Ldiv": 1.4,  # "gamma_conv": 15.0, "gamma_div" : 6.0,
          "Pprim": pin, "hprim": hinPrim, "Psuc": Psuc, "hsuc": hinSuc,
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
esim = ejectorSimulator.ejectorSimu(params, fluid=fluid, proplibrary = proplibrary)
# plot the ejector geometry:
ejplot = esim.ejector.draw()
esim.setup_solver( step0_mn = 0.001, maxStep_mn = 0.004)
esim.dv_kick = 0.004
## calculate the primary mass flow rate:
esim.calcPrimMassFlow(plotCrit0 = True, chokePos="divergent_part")
vcrit_tot = esim.params["vin_crit"]
print(f"critical speed motive nozzle {vcrit_tot}")

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

