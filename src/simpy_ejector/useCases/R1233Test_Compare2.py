# Test for R1233zde fluid. Comparison with Manuel Schieder's CFD results
# see https://doi.org/10.34726/hss.2023.108420 section 4.1!

import logging
import sys
logging.basicConfig(stream = sys.stdout, level = logging.INFO)

from simpy_ejector import  nozzleFactory, NozzleParams, nozzleSolver, EjectorGeom, refProp, EjectorMixer
from simpy_ejector.useCases import ejectorSimulator

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import importlib
import math

# get_ipython().run_line_magic('matplotlib', 'notebook')
#
# from IPython.display import Image
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
# display(HTML("<style>div.output_scroll { height: 44em; }</style>"))
from time import time
t0 = time()


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 140)
np.core.arrayprint._line_width = 180
np.set_printoptions(linewidth= 180)

# get_ipython().run_cell_magic('html', '', '<style type = "text/css">\n/* Any CSS style can go in here. */\nbody {\n font-family: sans-serif;\n font-size: 19px;}\n.dataframe th {\n    font-size: 18px;\n}\n.dataframe td {\n    font-size: 20px;\n}\n.CodeMirror{\nfont-size:20px;}\n\n</style>')

# params = { "Rin": 1.0, "Rt": 0.22, "Rout": 0.345, "gamma_conv": 15.0, "" "Dmix": 1.4,
#            "pin": 2140, "Tin" : 409, "Tsuc": 336.25, "Psuc" : 430  }
Rout = 18 * math.tan( 6.0 * math.pi/180) + 1.53

params = { "Rin": 0.97, "Rt": 0.153, "Rout": 0.3422, "gamma_conv": 15.7, "gamma_div" : 6.0, "Dmix": 2.0,
           "Pprim": 1800, "Tprim" : 400, "Tsuc": 317.25, "Psuc" : 150 ,
           "gamma_diffusor": 2.5, "mixerLen": 11.2, "diffuserLen" : 25.19 }


fluid = "R1233zde"  ## Manuel : "R1233zde" from Refprop 10.0 !
RP = refProp.setup(fluid)
[Din, hin] = refProp.getDh_from_TP(RP, params['Tprim'], params['Pprim'])
[Dsuc, hsuc] = refProp.getDh_from_TP(RP, params['Tsuc'], params['Psuc'])
subcooling = params['Tprim'] - RP.SATPdll(P = params["Pprim"], z= [1.0] , kph= 1 ).T
print(f"primary nozzle inlet subcooling {subcooling:.4f} Celsius")
suctionState = refProp.getTD(RP, hm=hsuc, P= params['Psuc'] )

esim = ejectorSimulator.ejectorSimu(params)

recalc = True
if recalc :
    esim.calcPrimMassFlow()
else:
    esim.params['vin_crit']= 0.3697265625
    esim.params['mass_flow_crit'] = 0.10295384643183664
    esim.makeEjectorGeom(params)
ejplot = esim.ejector.draw()
print(esim.params)


params['mixingParams'] = {'massExchangeFactor': 0.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.001,
                    'frictionWall': 0.00}
params["A_suction_inlet"] =2 * 1.979 * 0.475 * math.pi  # cm2 of the suction nozzle inlet.


res_crit = esim.motiveSolver()
print(f"By the motive nozzle exit:\n {res_crit.iloc[-1]}")
esim.setupMixer()
## test multiple secondary flow momentum equations
## Check the errors and warnings about the convergence, and select a method that gives you convergent and plausible results!
for momcType in [0,2,1]:
    esim.mixer.momCalcType = momcType
    esim.mixer.premixEqSimple = True
    esim.mixer.premixRootMethod = "hybr"
    esim.solvePremix(res_crit) # this sets the mixer, that is needed for the mixing calculation
    print(f"Momcalc {esim.mixer.momCalcType} suction MFR [g/s] {round(esim.mixerin['massFlowSecond'],3)}")
# sols = {}
# for Ni in range(1, 15):
#     print(f"calculate Ni = {Ni}")
#     esim.mixer.momCalcType = 2
#     esim.mixer.premixEqSimple = True
#     esim.mixer.premixRootMethod = "hybr"
#     esim.mixer.Nint = Ni
#     esim.solvePremix(res_crit)  # this sets the mixer, that is needed for the mixing calculation
#     sols[Ni] = esim.mixerin
# [print(si) for si in sols.items()]

esim.mixersolve()
diffusor_out = esim.solMix.iloc[-1]
out_prim = refProp.getTD(esim.RP, hm= diffusor_out["hp"], P=diffusor_out["p"] )
out_sec =  refProp.getTD(esim.RP, hm= diffusor_out["hs"], P=diffusor_out["p"] )
print(diffusor_out)
## we would need to reach this quality at the end: 1/q = 1 + entrainment_ratio
q_need = 1.0/ (1.0 + esim.mixerin["massFlowSecond"]/ esim.mixerin["massFlowPrim"]  )
print(f"needed vapor quality = {round(q_need, 3)}. calculated {round(esim.outlet_quality,3)}")

esim.mixer.plotSolution(res_crit, esim.solMix, "no shock in mixer",  ejplot)


Pdiffout = esim.solMix.iloc[-1]['p']
efficiency = esim.mixer.calcEfficiency(params["Pprim"], params['Tprim'],
                                  params['Psuc'], params['Tsuc'],
                                  Pdiffout,
                                  esim.mixerin["massFlowPrim"] * 1e-3,
                                  esim.mixerin["massFlowSecond"] * 1e-3)
print(f"Ejector Elbel Efficiency: {efficiency:.3f}")

P_CFD = pd.read_csv("./R1233zde_CFD_pressure1.csv",
                    sep=", ", na_values="undef").dropna()
plt.figure()
## subtract Lin = 3cm constant cross section part
plt.plot(P_CFD.iloc[:, 0] * 100 - 3, P_CFD.iloc[:, 1] * 1e-3)
plt.plot(res_crit['x'], res_crit['p'], color='#22AA22')
plt.plot(esim.solMix['x'], esim.solMix['p'], color='#22AA22')
plt.plot()
plt.ylabel("p[kPa]")
plt.legend(["CFD", "1D Simulation", "1D Simulation"])
plt.ylim((10, res_crit['p'][0] * 1.1))
plt.yscale('log')
plt.xlabel("x[cm]")
plt.title("Pressure profiles R1233zde CFD vs 1D Simulation")

