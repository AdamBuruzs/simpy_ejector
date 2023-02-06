# File for dimensioning of an ejector with R1233zd cooling fluid.

# R1233zde		A
# 	Mprim [kg/s]	0.57
# 	Msec [kg/s]	0.265
# 	Pprim [bar]	20.07
# 	Psec	2.763
# 	Tprim
# 	Tsec
# 	hprim [kJ/kg]	365.5
# 	hsec [kJ/kg]	437.1
# 	q_out	0.6838


# ! python -m site
### installing package in jupyter notebook
# import sys # replace the path below with the location of your .whl file :
#!{sys.executable} -m pip install C:\Users\BuruzsA\PycharmProjects\flows1d\dist\flows1d-1.0.0b4-py3-none-any.whl
## if you want to uninstall for test/updates 


import logging
import sys
logging.basicConfig(stream = sys.stdout, level = logging.INFO)


#sys.path.append("C:/Users/BuruzsA/PycharmProjects/")
#sys.path.append("C:/Users/BuruzsA/PycharmProjects/flows1d") ## this is not needed, if the package is installed in jupyter
from simpy_ejector import nozzleFactory, NozzleParams, nozzleSolver, EjectorGeom, refProp, EjectorMixer


import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import importlib
import math
import logging

logging.basicConfig(stream=sys.stdout, level= logging.INFO)

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
# Rout = 18 * math.tan( 6.0 * math.pi/180) + 1.53



class ejectorSimu:

    def __init__(self, params, fluid = "R1233zd" ):
        """ ejector simulator object
        :params params: a dictionary like: params = { "Rin": 1.1, "Rt": 0.29, "Rout": 0.4, "gamma_conv": 15.7, "gamma_div" : 6.0, "Dmix": 2.0,
        "Pprim": 2007, "hprim" : 365.5, "hsuc": 437.1, "Psuc" : 276.3 , "A_suction_inlet" : 8 }
        params["mixingParams"] = {'massExchangeFactor': 2.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.0,
                        'frictionWall': 0.0015} -> these are the parameters of the mixer & secondary-primary fluid mixing
          R in cm, A in cm2, h in kJ/kg, P in kPa, gamma in degree.
        """
        self.params = params
        self.fluid =  fluid ## Manuel : "R1233zde" from Refprop 10.0 !

    def makeEjectorGeom(self, params):
        """ create the ejector geometry"""
        #params = self.params
        Rin = params["Rin"]  # cm
        Rt = params["Rt"]  # 0.22
        Rout = params["Rout"]
        gamma_conv = params["gamma_conv"]  # grad
        gamma_div = params["gamma_div"]  # grad
        Dmix = params["Dmix"]
        Lcon = (Rin - Rt) / math.tan(gamma_conv * math.pi / 180)
        print(f"Primary nozzle: inlet Radius {Rin} cm,\n Throat radius {Rt} cm \n convergent lenght {Lcon} cm")
        Ldiv = (Rout - Rt) / math.tan(gamma_div * math.pi / 180)
        print(f"Primary nozzle: Outlet Radius {Rout} cm,\n divergent length {Ldiv}")
        nozzle = nozzleFactory.ConicConic(Rin=Rin, Lcon=Lcon, Rt=Rt, Ldiv=Ldiv, Rout=Rout)
        print(
            f"Primary nozzle: Rin ={Rin}, Rout = {Rout} Lcon = {Lcon}, Rt = {Rt} cm,\n converg Len {round(Lcon, 5)} divergent length {round(Ldiv, 5)} ")
        nozzle.setFriction(1.0e-3)

        ejector = EjectorGeom(nozzle, Dm=params["Dmix"])
        mixstart = Lcon + Ldiv + 1.1
        gamma_diffusor = params["gamma_diffusor"]  ## or the half of it??
        diffuserLen = params["diffuserLen"]
        mixerLen = params["mixerLen"]
        Ddif = math.tan(gamma_diffusor * math.pi / 180) * diffuserLen * 2.0 + Dmix
        diffuserLen = (Ddif - Dmix) / 2 / math.tan(gamma_diffusor * math.pi / 180)
        print(f"mixer start {mixstart} cm, diffuser length {round(diffuserLen, 3)} cm")
        ejector.setMixer(mixerstart=mixstart, mixerLen=mixerLen, diffuserLen=diffuserLen, diffuserHeight=Ddif / 2.0)
        # ejectorPlot = ejector.draw()
        self.ejector = ejector
        ### set up the nozzle solver:
        self.RP = refProp.setup(self.fluid)
        [Din, hin] = refProp.getDh_from_TP(RP, self.params['Tprim'], self.params['Pprim'])
        self.nsolver = nozzleSolver.NozzleSolver(nozzle, self.fluid, 1, solver="AdamAdaptive", mode="basic")
        self.nsolver.setFriction(1e-2)

    def calcPrimMassFlow(self):
        """calculate the motive nozzle critical speed and choking mass flow rate
        This function sets the self.nsolver!!
        """
        self.makeEjectorGeom(self.params)
        nozzle = self.ejector.nozzle
        print(f" prim press {self.params['Pprim']} kPa, sec press {self.params['Psuc']} kPa ")
        # self.nsolver = nozzleSolver.NozzleSolver(nozzle, self.fluid, 1, solver="AdamAdaptive", mode="basic")
        # self.nsolver.setFriction(1e-2)

        RP = refProp.setup(self.fluid)
        [Din, hin] = refProp.getDh_from_TP(RP, self.params['Tprim'], self.params['Pprim'])


        vin_crit = self.nsolver.calcCriticalSpeed( self.params['Pprim'], hin, 0.1, maxdev=1e-3, chokePos="divergent_part")

        # Manuel CFD: motive nozzle 0.1087 kg/s, suction nozzle 0.109889 kg/s

        nozzle_crit0 = self.nsolver.solveNplot(vin_crit, self.params['Pprim'], hin)
        #nsolver.plotsol(nozzle_crit0)
        # The choked flow in the ejector motive nozzle:

        print(f"calculated critical choking inlet speed = {round(vin_crit, 5)} m/s")
        mass_flow_crit = vin_crit * refProp.getTD(self.nsolver.RP, hin, self.params['Pprim'])['D'] * self.nsolver.nozzle.Aprofile(0) * 1e-4
        print(f"critical mass flow is {round(mass_flow_crit, 5)} kg/sec")
        #results = params
        self.params["vin_crit"] = vin_crit
        self.params["mass_flow_crit"] = mass_flow_crit
        #return ejector,results, nsolver

    def motiveSolver(self):
        """ obtain the motive nozzle solution with kick-helper
        :param critical_res: the 2. output of the calcMotiveNozzle return value
        params, critical_res, nsolver
        """
        sol_1 = self.nsolver.solveAdaptive1DBasic(self.params["vin_crit"], self.params["Pprim"],
                                             self.params["hprim"], 0.0, self.nsolver.nozzle.xt)
        vph_throat = sol_1.iloc[-1]
        v = vph_throat["v"]
        p = vph_throat["p"]
        h = vph_throat["h"]
        dv_kick = 2.0
        dp_kick = self.nsolver.pFromV_MassConst(v = vph_throat["v"], dv = dv_kick, p = vph_throat["p"], h = vph_throat["h"])
        print(f"mass conserving artificial kick: dv = {dv_kick} m/s, dp = {dp_kick} kPa")
        res_crit = self.nsolver.solveKickedNozzle(self.params["vin_crit"], self.params["Pprim"], self.params["hprim"], kicks = {'v': dv_kick, 'p': -dp_kick},
                                             solver= "adaptive_implicit", step0 = 0.001, maxStep = 0.005)
        print(f"throat by {self.nsolver.nozzle.xt}")
        self.nsolver.plotsol(res_crit, title = f"choked nozzle with friction = {self.nsolver.frictionCoef}.\n with artifical kick by throat with {dv_kick} m/sec ")
        print(res_crit.tail(1))
        self.primNozzleFlow = res_crit
        return res_crit

    def premix(self, res_crit):
        """ solving the premix equations this will calculate the secondary mass flow rate"""
        mixingParams = self.params["mixingParams"]
        self.mixer = EjectorMixer.EjectorMixer(self.fluid, self.ejector, mixingParams)
        self.mixer.setSuctionMassFlow(None)
        self.mixer.setSingleChoke(True)
        #self.mixer.ejector.Asi = 2 * 1.1 ** 2 * math.pi  # cm2 of the suction nozzle inlet.
        self.mixer.ejector.Asi = params["A_suction_inlet"]

        self.mixerin = self.mixer.premixWrapSolve(res_crit, self.params["Psuc"], self.params["Tsuc"])

    def mixersolve(self):
        """ solve the mixer equations until the end """
        mixerinput = [ self.mixerin[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]
        # solve the initial value ODE:
        self.solMix = self.mixer.solveMix(mixerinput)
        self.diffout = self.solMix.iloc[-1] # diffuser output
        out_prim = refProp.getTD(self.RP, hm=self.diffout["hp"], P=self.diffout["p"])
        out_sec = refProp.getTD(self.RP, hm=self.diffout["hs"], P=self.diffout["p"])
        massFlowPrim = self.diffout["vp"] * self.diffout["Ap"] * out_prim['D'] * 1e-4
        massFlowSec =  self.diffout["vs"] * self.diffout["As"] * out_sec['D'] * 1e-4
        quality_tot =  (massFlowPrim * out_prim['q'] +  massFlowSec * out_sec['q'] ) / (massFlowPrim + massFlowSec)
        logging.info(f"diffuser outlet")
        logging.info(f"primary {round(self.diffout['vp'],2)} m/s with vapor q: { out_prim['q'] }. MFR {round(massFlowPrim,2)}")
        logging.info(f"secondary {round(self.diffout['vs'],2)} m/s with vapor q: { out_sec['q'] }. MFR {round(massFlowSec,2)}")
        logging.info(f"total q {quality_tot}")
        self.outlet_quality = quality_tot

    def massFlowCheck(self):
        """ verify the mass flow conservation. Validate if the sum stays constant in the mixer
        solMix = the flow solution in the mixer and the diffuser part of the ejector
        """
        solMix = self.solMix
        Dp = solMix.apply(lambda x: refProp.getTD(self.RP, hm=x['hp'], P=x['p'])['D'], axis=1)  # density primary flow
        qp = solMix.apply(lambda x: refProp.getTD(self.RP, hm=x['hp'], P=x['p'])['q'], axis=1) # vapor quality
        solMix["MFRp"] = Dp * solMix['vp'] * solMix['Ap']*1e-4 # primary mass flow rate
        solMix["qp"] = qp
        Ds = solMix.apply(lambda x: refProp.getTD(self.RP, hm=x['hs'], P=x['p'])['D'],  axis = 1) # density primary flow
        qs = solMix.apply(lambda x: refProp.getTD(self.RP, hm=x['hs'], P=x['p'])['q'], axis=1)
        solMix["MFRs"] = Ds * solMix['vs'] * solMix['As']*1e-4
        solMix["qs"] = qs
        logging.info(f"mixer first sec density {Ds.head(1)}")
        solMix["q_average"] = (solMix["MFRp"]* solMix["qp"] + solMix["MFRs"] * solMix["qs"]) / ( solMix["MFRp"] + solMix["MFRs"] )

    def plotMixSolution(self, solNozzle, solMix, title = "" ):
        """ Plot the results of solveMix function
        :param solNozzle: solution of the nozzle
        :param solMix: the solution dataframe with data that we get from the solveMix function
        :return:
        """
        fig, ax = plt.subplots(nrows = 4, ncols = 1, sharex = True)
        fig.suptitle( title)
        #plt.subplot(411, sharex = True)
        # plt.plot(ejectorPlot)
        nozzleWall = [self.ejector.nozzle.Rprofile(xi) for xi in solNozzle['x'] ]
        mixerWall = [self.ejector.mixerR(xi) for xi in solMix['x'] ]
        ax[0].plot(solNozzle['x'], nozzleWall)
        ax[0].plot(solMix['x'], np.sqrt( solMix['Ap']/ math.pi))
        ax[0].plot(solMix['x'], mixerWall)
        ax[0].legend(['wall of Nozzle',  'primary motive stream', 'wall of Mixing region'])
        ax[0].set_ylabel("R [cm]")
        ##################################
        ax[1].plot(solNozzle['x'], solNozzle['p'], color = 'blue')
        ax[1].plot(solMix['x'], solMix['p'], color = 'blue')
        # speeds
        ax[1].plot(solNozzle['x'], solNozzle['v'], color = "#aa1111")
        ax[1].plot(solMix['x'], solMix['vp'], color = "#aa1111")
        ax[1].plot(solMix['x'], solMix['vs'], color = "#aa9911")
        ## enthalpy
        ax[1].plot(solMix['x'], solMix['hp'])
        ax[1].legend([ 'pressure [kPa]', 'p [kPa]',
                     'prim flow speed', 'prim flow speed', 'suction flow speed',
                     'hp'])
        ax[1].set_yscale("log")
        ax[1].set_xlabel("x [cm]")
        ## Mach numbers : ################
        #plt.subplot(413)
        ax[2].plot(solNozzle['x'], solNozzle['mach'])
        cPrim = [refProp.getSpeedSound(self.RP, solMix.iloc[i]['hp'], solMix.iloc[i]['p']) for i in range(solMix.__len__())]
        cSec = [refProp.getSpeedSound(self.RP, solMix.iloc[i]['hs'], solMix.iloc[i]['p']) for i in
                range(solMix.__len__())]
        ax[2].plot(solMix['x'], solMix['vp'] / cPrim )
        ax[2].plot(solMix['x'], solMix['vs'] / cSec)
        ax[2].legend(['primary Mach', 'primary Mach', 'secondary Mach'])
        ## quality ##############################
        ax[3].plot(solMix['x'],solMix['qs'])
        ax[3].plot(solMix['x'], solMix['qp'])
        ax[3].plot(solMix['x'], solMix['q_average'])
        ax[3].title.set_text("vapor qualities")
        ax[3].set_ylim([0,1.0])
        ax[3].legend(["secondary flow ", "primary flow", "average quality"])
        plt.tight_layout()

    # def checkmixerSolution(self):
    #     """ verifying pre-mixer solution"""
    #     mpars = {"hsi" : self.params["hsuc"],
    #              "vo": , "ho": , "so": ,
    #              ""}
    #     mpars["Dsi"] = Dsi
    #     params["hsi"] = hsi
    #     params["hst"] = hsi # inlet speed is low, the stagnation enthlpy is approximated by the inlet enthalpy
    #     suctionProps = refProp.getTD(self.RP, hsi, params['psi'])
    #     params["sst"] = suctionProps["s"]
    #     fluido = refProp.getTD(self.RP, params["ho"], po)  # fluid properties by Nozzle exit
    #     Dinit = fluido['D']
    #     params['Asi'] = self.ejector.Asi
    #     self.premixEquationsSCMom(mpars, x)

fluid = "R1233zd"  ## Manuel : "R1233zde" from Refprop 10.0 !
RP = refProp.setup(fluid)
params = { "Rin": 1.5, "Rt": 0.29, "Rout": 0.87, "gamma_conv": 15.0, "gamma_div" : 6.0, "Dmix": 2.67,
           "Pprim": 2007, "hprim" : 365.5, "hsuc": 437.1, "Psuc" : 276.3 , "A_suction_inlet" : 8 ,
           "mixerLen": 12 , "gamma_diffusor": 2.5, "diffuserLen": 25}
primQ = refProp.getTD(RP, hm= params["hprim"], P=params["Pprim"] )
params["Tprim"] = primQ['T']
params["Tsuc"] = refProp.getTD(RP, hm= params["hsuc"], P=params["Psuc"] )['T']

[Din, hin] = refProp.getDh_from_TP(RP, params['Tprim'], params['Pprim'])


#params["Rin"] = 1.1  # cm
#params["Rt"]  = 0.29 # smaller throat -> smaller critical primary mass flow rate
params["Rout"] = 0.87 # 0.468  cm radius
params["Dmix"] = 2.67
params["A_suction_inlet"] = 16
params["gamma_conv"] = 15.0 # degree
params["gamma_div"] = 7.0
params["mixingParams"] = {'massExchangeFactor': 2.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.0,
                        'frictionWall': 0.0015}
#ejector = makeEjectorGeom(params)
esim = ejectorSimu(params)
recalc = True
if recalc :
    esim.calcPrimMassFlow()
else:
    esim.params['vin_crit']= 0.871875
    esim.params['mass_flow_crit'] = 0.587452
    esim.makeEjectorGeom(params)
ejplot = esim.ejector.draw()
print(esim.params)

esim.makeEjectorGeom(params)
res_crit = esim.motiveSolver()
print(f"By the motive nozzle exit:\n {res_crit.iloc[-1]}")
esim.premix(res_crit) # this sets the mixer, that is needed for the mixing calculation
print(f"suction MFR [g/s] {round(esim.mixerin['massFlowSecond'],3)}")

esim.mixer.mixingParams = {'massExchangeFactor': 0.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.001,
                    'frictionWall': 0.00}
params["gamma_diffusor"] = 2.0
params["diffuserLen"] = 5
esim.makeEjectorGeom(params)
esim.premix(res_crit)
esim.mixersolve()
diffusor_out = esim.solMix.iloc[-1]
out_prim = refProp.getTD(esim.RP, hm= diffusor_out["hp"], P=diffusor_out["p"] )
out_sec =  refProp.getTD(esim.RP, hm= diffusor_out["hs"], P=diffusor_out["p"] )
print(diffusor_out)
## we would need to reach this quality at the end: 1/q = 1 + entrainment_ratio
q_need = 1.0/ (1.0 + esim.mixerin["massFlowSecond"]/ esim.mixerin["massFlowPrim"]  )
print(f"needed vapor quality = {round(q_need, 3)}. calculated {round(esim.outlet_quality,3)}")

esim.mixer.plotSolution(res_crit, esim.solMix, "no shock in mixer",  ejplot)

## maybe something wrong with the mixing formula? Too high primary mass flow rate at the end!
esim.massFlowCheck()
esim.plotMixSolution(res_crit, esim.solMix, "simpy_ejector 1D Ejector flow solution")
print( f"MFR prim {esim.mixerin['massFlowPrim']} sec {esim.mixerin['massFlowSecond']} "
       f"sum {round(esim.mixerin['massFlowPrim'] + esim.mixerin['massFlowSecond'],3)} g/sec ")
print(f"  {(esim.solMix['MFRp'] + esim.solMix['MFRs']).head(3)}")
print(f" mixer prim sec MFR in kg/s {[round(esim.solMix[k].iloc[1],3) for k in ['MFRp', 'MFRs']] } ")
print(f"secondary density {round(esim.mixerin['Dsy'],3)}")
mixerin_sec =  refProp.getTD(esim.RP, hm= esim.mixerin["hsy"], P=esim.mixerin["py"] )
print(f"this should be equal with {mixerin_sec}")

mat_outlet = refProp.getTD(RP, hm= 388.2 , P= 359.2 )
print(f"refrigerant state at the outlet {mat_outlet}")

if False:
    outs = []
    for pi in [1600, 1800, 2200]:
        for Ti in [380, 400, 420]:
            for Rti in [0.18, 0.22 ]:
                params = { "Rin": 1.0, "Rt": Rti, "Rout": 0.345, "gamma_conv": 15.0, "Dmix": 1.4,
                           "Pprim": pi, "Tprim" : Ti, "Tsuc": 317.25, "Psuc" : 150  }

                ejector, out, nsolver = calcMotiveNozzle(params)
                print(f"critical sol {out}")
                outs.append(out)
    recs = pd.DataFrame.from_records(outs, index = range(len(outs)))
    recs.to_csv("R1233_results.csv")

if False:
    pin = params['Pprim']
    nozzle_crit0 = nsolver.solveNplot(out["vin_crit"], pin, hin)
    nozzle_crit0 = nsolver.solveNplot(vin_crit, Pprim, hin)


    # ups, by the critical speed, the flow still stays subsonic! So let's apply a small kick in the throat:

    res_crit= None

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

    mixerin = mixer.premixWrapSolve(res_crit, Psuc ,Tsuc)

    # the premixing output is the input for the mixer

    mixerinput = [ mixerin[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]

    # solve the initial value ODE:

    solMix = mixer.solveMix(mixerinput)

    # The 2 annular flows: p: primary (motive flow), and s for the secondary ( suction flow)

    print(solMix)


    mixer.plotSolution(res_crit, solMix, "no shock in mixer",  ejectorPlot)
    solNoShock = solMix


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






