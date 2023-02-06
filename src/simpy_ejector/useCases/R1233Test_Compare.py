#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! python -m site
### installing package in jupyter notebook
# import sys # replace the path below with the location of your .whl file :
#!{sys.executable} -m pip install C:\Users\BuruzsA\PycharmProjects\flows1d\dist\flows1d-1.0.0b4-py3-none-any.whl
## if you want to uninstall for test/updates 


import logging
import sys
logging.basicConfig(stream = sys.stdout, level = logging.INFO)


#sys.path.append("C:/Users/BuruzsA/PycharmProjects/")
sys.path.append("C:/Users/BuruzsA/PycharmProjects/flows1d") ## this is not needed, if the package is installed in jupyter
from flows1d.core import nozzleFactory, NozzleParams, nozzleSolver, EjectorGeom, refProp, EjectorMixer


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

fluid = "R1233zd"  ## Manuel : "R1233zde" from Refprop 10.0 !
RP = refProp.setup(fluid)
[Din, hin] = refProp.getDh_from_TP(RP, params['Tin'], params['pin'])


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
    nozzle = nozzleFactory.ConicConic(Rin = Rin, Lcon=Lcon, Rt = Rt, Ldiv = Ldiv, Rout = Rout)
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

    fluid = "R1233zd" ## Manuel : "R1233zde" from Refprop 10.0 !
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

# case = "NONE"
# ## case A
# if case == "A":
#     pin = 1800.0  # kPa  2000
#     Tin =  400   # Kelvin 127 C
#     Tsuc = 317.25 #  suction temperature K -> 44C
#     Psuc =  150 # suction pressure in kPa
# elif case == "B":
#     pin = 2139.0  # kPa  2000
#     Tin =  409.32   # Kelvin  400
#     Tsuc = 336.63 #  suction temperature K
#     Psuc =  430 # suction pressure in kPa
# print(f" prim press {pin} kPa, sec press {Psuc} kPa ")
#
# fluid = "R1233zd"
# RP = refProp.setup(fluid)
# [Din, hin] = refProp.getDh_from_TP(RP, Tin, pin)
#
# matinfo = RP.INFOdll(1)
# print(matinfo)
#
# matinfo.Tc - 273.15 # critical temp. in Celsius
#
# nsolver = nozzleSolver.NozzleSolver(nozzle, fluid, 1, solver="AdamAdaptive", mode="basic")
# nsolver.setFriction(1e-2)
# vin_crit = nsolver.calcCriticalSpeed(pin, hin, 0.1, maxdev= 1e-3, chokePos="divergent_part" )
#
# # Manuel CFD: motive nozzle 0.1087 kg/s, suction nozzle 0.109889 kg/s
#
# # The choked flow in the ejector motive nozzle:
#
# print(f"calculated critical choking inlet speed = {round(vin_crit,5)} m/s")
# mass_flow_crit = vin_crit * refProp.getTD(nsolver.RP, hin, pin)['D'] * nsolver.nozzle.Aprofile(0)*1e-4
# print(f"critical mass flow is {round(mass_flow_crit,5)} kg/sec")

ejector, out, nsolver = calcMotiveNozzle(params)
ejector.draw()

if False:
    outs = []
    for pi in [1600, 1800, 2200]:
        for Ti in [380, 400, 420]:
            for Rti in [0.18, 0.22 ]:
                params = { "Rin": 1.0, "Rt": Rti, "Rout": 0.345, "gamma_conv": 15.0, "Dmix": 1.4,
                           "pin": pi, "Tin" : Ti, "Tsuc": 317.25, "Psuc" : 150  }

                ejector, out, nsolver = calcMotiveNozzle(params)
                print(f"critical sol {out}")
                outs.append(out)
    recs = pd.DataFrame.from_records(outs, index = range(len(outs)))
    recs.to_csv("R1233_results.csv")

if True:
    pin = params['pin']
    nozzle_crit0 = nsolver.solveNplot(out["vin_crit"], pin, hin)
    nozzle_crit0 = nsolver.solveNplot(vin_crit, pin, hin)


    # ups, by the critical speed, the flow still stays subsonic! So let's apply a small kick in the throat:

    # In[53]:


    res_crit= None


    # In[60]:


    print("############# resolve with kicks ############ ")
    v01 = vin_crit #-0.004
    sol_1 = nsolver.solveAdaptive1DBasic(v01, pin, hin, 0.0, nsolver.nozzle.xt)
    vph_throat = sol_1.iloc[-1]
    v = vph_throat["v"]
    p = vph_throat["p"]
    h = vph_throat["h"]
    dv_kick = 2.0
    dp_kick = nsolver.pFromV_MassConst(v = vph_throat["v"], dv = dv_kick, p = vph_throat["p"], h = vph_throat["h"])
    print(f"mass conserving artificial kick: dv = {dv_kick} m/s, dp = {dp_kick} kPa")


    res_crit = nsolver.solveKickedNozzle(v01, pin , hin, kicks = {'v': dv_kick, 'p': -dp_kick},
                                         solver= "adaptive_implicit", step0 = 0.001, maxStep = 0.005)
    print(f"throat by {nsolver.nozzle.xt}")
    nsolver.plotsol(res_crit, title = f"choked nozzle with friction = {nsolver.frictionCoef}.\n with artifical kick by throat with {dv_kick} m/sec ")
    print(res_crit.tail(1))


    # In[62]:


    res_crit[140:150]


    # In[18]:


    print(res_crit.shape)
    res_crit.tail(5)


    # In[63]:


    #nsolver.plotsol(res_crit, title = "choked nozzle with friction = {} ".format(nsolver.frictionCoef))


    # that looks better ...

    # ## Mixing calculations

    # #### set up the mixer properties.

    # In[64]:


    mixingParams = { 'massExchangeFactor' : 2.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.0, 'frictionWall': 0.0015}
    mixer = EjectorMixer.EjectorMixer(fluid, ejector, mixingParams)


    # #### NO: set the suction nozzle mass flow rate [g/s]: mixer.setSuctionMassFlow(80) NO

    # In[65]:


    mixer.setSuctionMassFlow(None)
    mixer.setSingleChoke(True)
    mixer.ejector.Asi = 2* 1.1**2 * math.pi  # cm2 of the suction nozzle inlet.


    # # calculate the Pre-mixing chamber

    # In[66]:


    mixerin = mixer.premixWrapSolve(res_crit, Psuc ,Tsuc)


    # the premixing output is the input for the mixer

    # In[23]:


    mixerinput = [ mixerin[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]


    # solve the initial value ODE:

    # In[24]:


    solMix = mixer.solveMix(mixerinput)


    # The 2 annular flows: p: primary (motive flow), and s for the secondary ( suction flow)

    # In[25]:


    print(solMix)


    # In[26]:


    mixer.plotSolution(res_crit, solMix, "no shock in mixer",  ejectorPlot)
    solNoShock = solMix


    # In[27]:


    print(f"elapsed time {round(time()-t0, 3)} sec")


    # In[28]:


    #P_CFD = pd.read_csv("C:/Users/BuruzsA/Documents/projects/ETHP/ManuelCFD/pressure_CFD1.csv")
    P_CFD = pd.read_csv("C:/Users/BuruzsA/Documents/projects/Ejector/ETHP/ManuelCFD/p_stat_CFD.csv")
    plt.figure()
    plt.plot(P_CFD.iloc[:,0]*100,P_CFD.iloc[:,1]*1e-3 )
    plt.plot(res_crit['x'], res_crit['p'], color = '#22AA22')
    plt.plot(solMix['x'], solMix['p'], color = '#22AA22')
    plt.plot()
    plt.ylabel("p[kPa]")
    plt.legend(["CFD", "1D Simulation", "1D Simulation"])
    plt.ylim((0, res_crit['p'][0]*1.1))


    # ## Diffusor outlet pressure vs suction mass flow rate

    # In[29]:


    # try various mass flow rates in g/sec:
    res = []
    for suctionMassFlow in np.arange(10, 200, 10):
        mixingParams = { 'massExchangeFactor' : 2.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.0, 'frictionWall': 0.0015}
        mixer = EjectorMixer.EjectorMixer(fluid, ejector, mixingParams)
        mixer.setSuctionMassFlow(suctionMassFlow)
        mixerin = mixer.premixWrapSolve(res_crit, Psuc ,Tsuc)
        mixerinput = [ mixerin[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]
        solMix = mixer.solveMix(mixerinput)
        pout = round(solMix['p'].iloc[-1],3)
        print(f"suction massflow: {suctionMassFlow}. pressure outlet : {pout}")
        res.append([suctionMassFlow, pout])
    mdp = pd.DataFrame(res, columns = ["massFlowRate[g/s]","pressure[kPa]"])


    # In[30]:


    mdp


    # # APPENDIX

    # without shock wave both flow will be supersonic in the mixer and also in the diffuser.
    # This gives us too low exit pressures. So let's assume that there is a shockwave at some x_sh, and afterwards the both flows are mixed:

    # In[31]:


    x_sh= 13.5 # let's say there is a normal shock wave at x_sh!


    # In[32]:


    solNoShock # this would be the solution without shock:


    # now we calculate, how the solution looks like with a shock wave in x_sh:

    # In[33]:


    beforeShock, afterShock, solMixShocked = mixer.mixingShock(x_sh, solNoShock, mergeflows = True)


    # let's plot the results:

    # In[34]:


    mixer.plotSolution(res_crit, solMixShocked, "normal shock at {} cm".format(x_sh),  ejectorPlot)


    # so the shock at x_sh kicks the flow down to subsonic region.
    # the values at the diffuser exit:

    # In[35]:


    solMixShocked.iloc[-1]


    #  x in cm, p in kPa, vp/vs = primary/secondary speed in m/s, hp, hs in kJ/kg/K, Ap,As in cm^2

    # In[36]:


    solMixShocked


    # In[37]:


    solMixShocked['Dp'] = [refProp.getTD(nsolver.RP, solMixShocked['hp'][i] ,solMixShocked['p'][i] )['D'] for i in range(solMixShocked.__len__()) ]
    solMixShocked['Ds'] = [refProp.getTD(nsolver.RP, solMixShocked['hs'][i] ,solMixShocked['p'][i] )['D'] for i in range(solMixShocked.__len__()) ]


    # Let's check the mass flow rates!

    # In[38]:


    MFp = solMixShocked['Dp'] * solMixShocked['vp'] *  solMixShocked['Ap']* 1e-4 # primary mass flow rate
    MFs = solMixShocked['Ds'] * solMixShocked['vs'] *  solMixShocked['As']* 1e-4 # secondary mass flow rate
    plt.figure()
    plt.plot(solMixShocked['x'], MFp+MFs)
    plt.axvline(x=x_sh, linestyle=':', color='b')
    plt.ylabel("mass flow [kg/sec]")
    plt.ylim(ymin = min(MFp+MFs)*0.95, ymax = max(MFp+MFs)*1.05)


    # Cheking if the total mass flow rate really constant:

    # ok, that looks good.

    # In[39]:


    print("pressure at the diffuser exit = {} kPa".format(solMixShocked.iloc[-1]['p']))


    # TODO: find the x_sh shock location in dependence of the exit pressure (= invert the x_sh -> p_{diffuser exit} function)

    # In[40]:


    MFp[0]


    # In[41]:


    Pdiffout = solMixShocked.iloc[-1]['p']
    Pdiffout


    # In[42]:


    efficiency = mixer.calcEfficiency(pin, Tin, Psuc, Tsuc, Pdiffout, MFp[0] , MFs[0] )


    # Ejektor Efficiency according to Elbel

    # In[43]:


    efficiency


    # ## Flow simulation from outlet pressure ##
    # Let's see the more realistic inverse problem: Usually we can not measure the shock-wave position, but we can measure pretty accurately the **pressure in the diffusor exit**. We can even set the diffusor exit-pressure, and thus it's one of the key control variables when we set up an ejector.
    # So our python package provides a function to calculate the double choking case shock-wave position based on the exit pressure of the diffusor:

    # In[44]:


    P_outlet = 230 # kPa for example
    shockFromP = mixer.getChokePosition(P_outlet, solNoShock)


    # In[45]:


    shockFromP


    # In[46]:


    #beforeShock, afterShock, solMixShockedP = mixer.mixingShock(shockFromP['shockPos'], solNoShock, mergeflows = True)


    # In[47]:


    #mixer.plotSolution(res_crit, solMixShockedP, "for outlet Pressure {} kPa \n calculated normal shock at {} cm".format(P_outlet, round(shockFromP['shockPos'],3) ),  ejectorPlot)






