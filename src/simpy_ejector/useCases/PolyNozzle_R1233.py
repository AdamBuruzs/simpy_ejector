# Test for R1233zde fluid.
# example of not conic nozzle. The nozzle profile will be calculated with a polynomial

import logging
import sys
logging.basicConfig(stream = sys.stdout, level = logging.INFO)

from simpy_ejector import  nozzleFactory, NozzleParams, nozzleSolver, EjectorGeom, refProp, EjectorMixer

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
           "pin": 1800, "Tin" : 400, "Tsuc": 317.25, "Psuc" : 150  }

fluid = "R1233zde"  ## Manuel : "R1233zde" from Refprop 10.0 !
RP = refProp.setup(fluid)
[Din, hin] = refProp.getDh_from_TP(RP, params['Tin'], params['pin'])

def PolyNozzle(Rin, Lcon, Rt, Ldiv,Rout):
    """create a CD Nozzle with Cubic convergent, and quadratic divergent part.
    Each paramters are in cm!

    :param Rin: inlet Radius
    :param Lcon: length of convergent part
    :param Rt: throat radius
    :param Ldiv: length of divergent part
    :param Rout: radius of outlet exit
    :return: the nozzle
    """
    def nozzleRProfile(x):
        if (x < 0):
            return 0.0
        elif ((x >= 0) & (x <= Lcon)):
            b = -3 * (Rin - Rt)/Lcon**2
            a = - 2 *b / Lcon / 3
            return a * x**3 + b * x**2.0 + Rin
        elif ((x >= Lcon) & (x <= Lcon + Ldiv)):
            ad = (Rout - Rt)/Ldiv**2
            return Rt + ad * (x -Lcon)**2
        else:
            return 0

    nozzle = NozzleParams.NozzleParams.fromRProfile(nozzleRProfile, Lcon + Ldiv, Lcon)
    return nozzle


def calcMotiveNozzle(params):
    # Schlemminger Hochtemperatur Wärmepumpe mit Ejektor
    Rin = params["Rin"] # cm
    Rt = params["Rt"] # 0.22
    Rout = params["Rout"]
    gamma_conv = params["gamma_conv"] # grad
    gamma_div = params["gamma_div"] # grad
    Dmix = params["Dmix"]
    Lcon = (Rin-Rt) / math.tan(gamma_conv * math.pi/180)
    print(f"Primary nozzle: inlet Radius {Rin} cm,\n Throat radius {Rt} cm \n convergent lenght {Lcon} cm")
    Ldiv = (Rout-Rt) / math.tan(gamma_div * math.pi/180)
    print(f"Primary nozzle: Outlet Radius {Rout} cm,\n divergent length {Ldiv}")
    nozzle = PolyNozzle(Rin = Rin, Lcon=Lcon, Rt = Rt, Ldiv = Ldiv, Rout = Rout)
    print(f"Primary nozzle: Rin ={Rin}, Rout = {Rout} Lcon = {Lcon}, Rt = {Rt} cm,\n converg Len {round(Lcon,5)} divergent length {round(Ldiv,5)} ")
    nozzle.setFriction( 1.0e-3 )

    ejector = EjectorGeom(nozzle, Dm = params["Dmix"] )
    mixstart = Lcon + Ldiv + 1.1
    gamma_diffusor = 2.5 ## or the half of it??
    diffuserLen = 25.19
    Ddif = math.tan(2.5 * math.pi / 180) * diffuserLen * 2.0 + Dmix
    diffuserLen = (Ddif - Dmix)/ 2 / math.tan(gamma_diffusor * math.pi/180)
    print(f"mixer start {mixstart} cm, diffuser length {round(diffuserLen,3)} cm")
    ejector.setMixer(mixerstart=mixstart, mixerLen=11.2, diffuserLen=diffuserLen, diffuserHeight = Ddif/2.0)
    #ejectorPlot = ejector.draw()

    print(f" prim press {params['pin']} kPa, sec press {params['Psuc']} kPa ")

    fluid = "R1233zde" ## Manuel : "R1233zde" from Refprop 10.0 !
    RP = refProp.setup(fluid)
    [Din, hin] = refProp.getDh_from_TP(RP, params['Tin'], params['pin'])

    nsolver = nozzleSolver.NozzleSolver(nozzle, fluid, 1, solver="AdamAdaptive", mode="basic")
    nsolver.setFriction(1e-2)
    vin_crit = nsolver.calcCriticalSpeed( params['pin'], hin, 0.1, maxdev=1e-3, chokePos="divergent_part")

    # Manuel CFD: motive nozzle 0.1087 kg/s, suction nozzle 0.109889 kg/s

    nozzle_crit0 = nsolver.solveNplot(vin_crit, params['pin'], hin)
    #nsolver.plotsol(nozzle_crit0)
    # The choked flow in the ejector motive nozzle:

    print(f"calculated critical choking inlet speed = {round(vin_crit, 5)} m/s")
    mass_flow_crit = vin_crit * refProp.getTD(nsolver.RP, hin, params['pin'])['D'] * nsolver.nozzle.Aprofile(0) * 1e-4
    print(f"critical mass flow is {round(mass_flow_crit, 5)} kg/sec")
    results = params
    results["vin_crit"] = vin_crit
    results["mass_flow_crit"] = mass_flow_crit
    return ejector,results, nsolver


ejector, out, nsolver = calcMotiveNozzle(params)
ejector.draw()

if True:
    pin = params['pin']
    ## this would probably stay subcritical:
    # nozzle_crit0 = nsolver.solveNplot(out["vin_crit"], pin, hin)
    # nozzle_crit0 = nsolver.solveNplot(vin_crit, pin, hin)
    # ups, by the critical speed, the flow still stays subsonic! So let's apply a small kick in the throat:

    res_crit= None

    print("############# resolve with kicks ############ ")
    v01 = out["vin_crit"] #-0.004
    sol_1 = nsolver.solveAdaptive1DBasic(v01, pin, hin, 0.0, nsolver.nozzle.xt)
    vph_throat = sol_1.iloc[-1]
    v = vph_throat["v"]
    p = vph_throat["p"]
    h = vph_throat["h"]
    dv_kick = 7.0
    dp_kick = nsolver.pFromV_MassConst(v = vph_throat["v"], dv = dv_kick, p = vph_throat["p"], h = vph_throat["h"])
    print(f"mass conserving artificial kick: dv = {dv_kick} m/s, dp = {dp_kick} kPa")

    res_crit = nsolver.solveKickedNozzle(v01, pin , hin, kicks = {'v': dv_kick, 'p': -dp_kick},
                                         solver= "adaptive_implicit", step0 = 0.001, maxStep = 0.005)
    print(f"throat by {nsolver.nozzle.xt}")
    nsolver.plotsol(res_crit, title = f"choked nozzle with friction = {nsolver.frictionCoef}.\n with artifical kick by throat with {dv_kick} m/sec ")
    print(res_crit.tail(1))

    res_crit[140:150]

    print(res_crit.shape)
    res_crit.tail(5)


    mixingParams = { 'massExchangeFactor' : 2.e-4, 'dragFactor': 0.1, 'frictionInterface': 0.01, 'frictionWall': 0.01}
    mixer = EjectorMixer.EjectorMixer(fluid, ejector, mixingParams)

    mixer.setSuctionMassFlow(None)
    mixer.setSingleChoke(True)
    #mixer.ejector.Asi = 2* 1.1**2 * math.pi  # cm2 of the suction nozzle inlet.
    mixer.ejector.Asi = 2 * 1.979 * 0.475 * math.pi  # cm2 of the suction nozzle inlet.

    print(f"secondary nozzle inlet cross section Area {mixer.ejector.Asi} cm^2")

    # # calculate the Pre-mixing chamber
    for momcalctype in [0,1]:
        mixer.momCalcType = momcalctype
        mixerin = mixer.premixWrapSolve(res_crit, params['Psuc'] ,params['Tsuc'])
        print(f"Premixer momentum calculation Type {momcalctype}:")
        print(
            f"MomType{momcalctype} Mass Flow Rates primary : {round(mixerin['massFlowPrim'], 4)} g/sec secondary: {round(mixerin['massFlowSecond'], 4)} g/s")

    mixerinput = [ mixerin[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]

    solMix = mixer.solveMix(mixerinput)

    print(solMix)

    ejectorPlot = ejector.draw()
    mixer.plotSolution(res_crit, solMix, "no shock in mixer",  ejectorPlot)
    solNoShock = solMix

    print(f"elapsed time {round(time()-t0, 3)} sec")


    # diffuser outlet pressure [kPa]
    Pdiffout = solMix.iloc[-1]['p']
    efficiency = mixer.calcEfficiency(params["pin"], params['Tin'],
                                      params['Psuc'], params['Tsuc'],
                                      Pdiffout,
                                      mixerin["massFlowPrim"]*1e-3,
                                      mixerin["massFlowSecond"]*1e-3)
    print(f"Ejector Elbel Efficiency: {efficiency:.3f}")


    # #P_CFD = pd.read_csv("C:/Users/BuruzsA/Documents/projects/ETHP/ManuelCFD/pressure_CFD1.csv")
    P_CFD = pd.read_csv("./R1233zde_CFD_pressure1.csv",
                        sep=", ", na_values="undef").dropna()
    plt.figure()
    ## subtract Lin = 3cm constant cross section part
    plt.plot(P_CFD.iloc[:,0]*100 - 3,P_CFD.iloc[:,1]*1e-3 )
    plt.plot(res_crit['x'], res_crit['p'], color = '#22AA22')
    plt.plot(solMix['x'], solMix['p'], color = '#22AA22')
    plt.plot()
    plt.ylabel("p[kPa]")
    plt.legend(["CFD", "1D Simulation", "1D Simulation"])
    plt.ylim((10, res_crit['p'][0]*1.1))
    plt.yscale('log')
    plt.xlabel("x[cm]")
    plt.title("Pressure profiles R1233zde CFD vs 1D Simulation")
    #
    print(f"diffusor outlet state\n {solMix.iloc[-1]}")
