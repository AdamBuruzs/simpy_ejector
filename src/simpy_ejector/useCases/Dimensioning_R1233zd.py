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
import sys
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
import matplotlib.pyplot as plt
import pandas as pd
from simpy_ejector.useCases import ejectorSimulator
from simpy_ejector import refprop_material, materialFactory

# load Refprop for your fluid:
fluid = "R1233zde"
# RP = refProp.setup(fluid)
# RProps = refprop_material.MaterialProperties(fluid)
RProps = materialFactory.MaterialPropertiesFactory.create( material = fluid, library='refprop' )

# set up geometry parameters:
params = { "Rin": 1.5, "Rt": 0.29, "Rout": 0.87, "gamma_conv": 15.0, "gamma_div" : 6.0, "Dmix": 2.67,
           "Pprim": 2007, "hprim" : 365.5, "hsuc": 437.1, "Psuc" : 276.3 , "A_suction_inlet" : 16 ,
           "mixerLen": 12 , "gamma_diffusor": 2.5, "diffuserLen": 10}
## calculate Temperatures from specific enthalpy with Refprop:
primQ = RProps.getTD( hm= params["hprim"], P=params["Pprim"] )
params["Tprim"] = primQ['T']
params["Tsuc"] = RProps.getTD( hm= params["hsuc"], P=params["Psuc"] )['T']

# set parameters of the mixing calculations:
params["mixingParams"] = {'massExchangeFactor': 2.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.0,
                        'frictionWall': 0.0015}

# create a simulator object:
esim = ejectorSimulator.ejectorSimu(params, fluid = "R1233zde", proplibrary= "refprop")
# plot the ejector geometry:
ejplot = esim.ejector.draw()
## calculate the primary mass flow rate:
esim.calcPrimMassFlow()

## calculate the critical (= choked flow) solution in the motive nozzle:
res_crit = esim.motiveSolver()
print(f"By the motive nozzle exit:\n {res_crit.iloc[-1]}")
# solve the pre-mixing equations:
esim.premix(res_crit) # this sets the mixer, that is needed for the mixing calculation
print(f"suction MFR [g/s] {round(esim.mixerin['massFlowSecond'],3)}")

# solve the mixer equations until the ejector outlet
esim.mixersolve()

esim.massFlowCheck()
outletValues = esim.solMix.iloc[-1].to_dict() # .transpose()

average_h = (outletValues["MFRp"] * outletValues["hp"] + outletValues["MFRs"] * outletValues["hs"])/ (outletValues["MFRp"] + outletValues["MFRs"] )
print(f"Ejector outlet specific enthalpy {round(average_h,3)} kJ/kg, pressure {round(outletValues['p']*1e-2,3)} bar,"
      f" vapor quality {round(outletValues['q_average'],3)}  ")

esim.plotMixSolution(res_crit, esim.solMix, "simpy_ejector 1D Ejector flow solution")

efficiency = esim.calcEfficiency()
print(f"Elbel Efficiency = {efficiency}")