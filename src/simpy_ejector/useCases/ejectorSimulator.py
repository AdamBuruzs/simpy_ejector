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


#sys.path.append("C:/Users/BuruzsA/PycharmProjects/")
#sys.path.append("C:/Users/BuruzsA/PycharmProjects/flows1d") ## this is not needed, if the package is installed in jupyter
from simpy_ejector import nozzleFactory, NozzleParams, nozzleSolver, EjectorGeom, refProp, EjectorMixer


import numpy as np
import math
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import importlib
import math

from time import time
t0 = time()

class ejectorSimu:

    def __init__(self, params, fluid = "R1233zd" ):
        """ ejector simulator object

        :param fluid: fluid name from refprop see https://pages.nist.gov/REFPROP-docs/#list-of-fluids
        :param params: a dictionary with fields <br>
         "Rin": primary nozzle inlet radius in [cm] ( example 1.1) <br>
         "Rt": primary nozzle throat radius in [cm] ( example 0.29) <br>
         "Rout": primary nozzle outlet/exit radius in [cm] ( example 0.4) <br>
         "gamma_conv": primary nozzle convergent part angle [degree] <br>
         "gamma_div" : primary nozzle divergent part angle [degree] <br>
         "Dmix": mixer diameter (2*radius) in [cm]   <br>
         "Pprim": primary nozzle inlet pressure in [kPa - kiloPascal!] <br>
         "hprim" : primary nozzle inlet specific enthalpy in kJ/kg <br>
         "hsuc": secondary/suction nozzle inlet specific enthalpy in kJ/kg <br>
         "Psuc" : secondary nozzle inlet pressure in [kPa - kiloPascal!] <br>
         "A_suction_inlet" : primary nozzle inlet cross section area [cm^2] <br>
         "mixerLen": length of the mixer [cm] <br>
         "gamma_diffusor": angle of the diffuser profile in [degree] <br>
         "diffuserLen": length of the diffuser [cm] <br>
         params["mixingParams"] = {'massExchangeFactor': 2.e-4, 'dragFactor': 0.01, 'frictionInterface': 0.0,
                        'frictionWall': 0.0015} -> these are the parameters of the mixer & secondary-primary fluid mixing
          R in cm, A in cm2, h in kJ/kg, P in kPa, gamma in degree.
        """
        self.params = params
        self.fluid =  fluid
        self.RP = refProp.setup(self.fluid)
        self.makeEjectorGeom(params)
        if not 'hprim' in params.keys():
            Dprim, hp = refProp.getDh_from_TP(self.RP, params['Tprim'], params['Pprim'])
            params['hprim'] = hp
        if not 'hsuc' in params.keys():
            Dsuc, hs = refProp.getDh_from_TP(self.RP, params['Tsuc'], params['Psuc'])
            params['hsuc'] = hs

    def makeEjectorGeom(self, params):
        """ create the ejector geometry.
        You can also use this function to update ejector geometry without reloading the whole object,
        and recalculating for example the critical speed"""
        #params = self.params
        Rin = params["Rin"]  # cm
        Rt = params["Rt"]  # 0.22
        Rout = params["Rout"]
        gamma_conv = params["gamma_conv"]  # grad
        gamma_div = params["gamma_div"]  # grad
        Dmix = params["Dmix"]
        Lcon = (Rin - Rt) / math.tan(gamma_conv * math.pi / 180)
        logging.info(f"Primary nozzle: inlet Radius {Rin} cm,\n Throat radius {Rt} cm \n convergent lenght {Lcon} cm")
        Ldiv = (Rout - Rt) / math.tan(gamma_div * math.pi / 180)
        logging.info(f"Primary nozzle: Outlet Radius {Rout} cm,\n divergent length {Ldiv}")
        nozzle = nozzleFactory.ConicConic(Rin=Rin, Lcon=Lcon, Rt=Rt, Ldiv=Ldiv, Rout=Rout)
        logging.info(
            f"Primary nozzle: Rin ={Rin}, Rout = {Rout} Lcon = {Lcon}, Rt = {Rt} cm,\n converg Len {round(Lcon, 5)} divergent length {round(Ldiv, 5)} ")
        nozzle.setFriction(1.0e-3)

        ejector = EjectorGeom(nozzle, Dm=params["Dmix"])
        mixstart = Lcon + Ldiv + 1.1
        gamma_diffusor = params["gamma_diffusor"]  ## or the half of it??
        diffuserLen = params["diffuserLen"]
        mixerLen = params["mixerLen"]
        Ddif = math.tan(gamma_diffusor * math.pi / 180) * diffuserLen * 2.0 + Dmix
        diffuserLen = (Ddif - Dmix) / 2 / math.tan(gamma_diffusor * math.pi / 180)
        logging.info(f"mixer start {mixstart} cm, diffuser length {round(diffuserLen, 3)} cm")
        ejector.setMixer(mixerstart=mixstart, mixerLen=mixerLen, diffuserLen=diffuserLen, diffuserHeight=Ddif / 2.0)
        # ejectorPlot = ejector.draw()
        self.ejector = ejector
        ### set up the nozzle solver:
        [Din, hin] = refProp.getDh_from_TP(self.RP, self.params['Tprim'], self.params['Pprim'])
        self.nsolver = nozzleSolver.NozzleSolver(nozzle, self.fluid, 1, solver="AdamAdaptive", mode="basic")
        self.nsolver.setFriction(1e-2)

    def calcPrimMassFlow(self):
        """calculate the motive nozzle critical speed and choking mass flow rate
        This function sets the self.nsolver!!
        """
        self.makeEjectorGeom(self.params)
        nozzle = self.ejector.nozzle
        logging.info(f" prim press {self.params['Pprim']} kPa, sec press {self.params['Psuc']} kPa ")
        # self.nsolver = nozzleSolver.NozzleSolver(nozzle, self.fluid, 1, solver="AdamAdaptive", mode="basic")
        # self.nsolver.setFriction(1e-2)

        RP = refProp.setup(self.fluid)
        [Din, hin] = refProp.getDh_from_TP(RP, self.params['Tprim'], self.params['Pprim'])

        vin_crit = self.nsolver.calcCriticalSpeed( self.params['Pprim'], hin, 0.1, maxdev=1e-3, chokePos="divergent_part")

        nozzle_crit0 = self.nsolver.solveNplot(vin_crit, self.params['Pprim'], hin, doPlot=False)

        logging.info(f"calculated critical choking inlet speed = {round(vin_crit, 5)} m/s")
        mass_flow_crit = vin_crit * refProp.getTD(self.nsolver.RP, hin, self.params['Pprim'])['D'] * self.nsolver.nozzle.Aprofile(0) * 1e-4
        logging.info(f"critical mass flow is {round(mass_flow_crit, 5)} kg/sec")
        #results = params
        self.params["vin_crit"] = vin_crit
        self.params["mass_flow_crit"] = mass_flow_crit
        #return ejector,results, nsolver

    def motiveSolver(self):
        """Obtain the motive nozzle solution with kick-helper.
        This kick will help to reach the supersonic flow in the primary nozzle at the throat.

        :return: pandas DataFrame with the flow parameters of the critical flow
        for each x (measured in cm) integration points
        """
        sol_1 = self.nsolver.solveAdaptive1DBasic(self.params["vin_crit"], self.params["Pprim"],
                                             self.params["hprim"], 0.0, self.nsolver.nozzle.xt)
        vph_throat = sol_1.iloc[-1]
        v = vph_throat["v"]
        p = vph_throat["p"]
        h = vph_throat["h"]
        dv_kick = 2.0 ## [m/s] increase this value, if the flow does not switch to supersonic after the throat
        dp_kick = self.nsolver.pFromV_MassConst(v = vph_throat["v"], dv = dv_kick, p = vph_throat["p"], h = vph_throat["h"])
        logging.info(f"mass conserving artificial kick: dv = {dv_kick} m/s, dp = {dp_kick} kPa")
        res_crit = self.nsolver.solveKickedNozzle(self.params["vin_crit"], self.params["Pprim"], self.params["hprim"], kicks = {'v': dv_kick, 'p': -dp_kick},
                                             solver= "adaptive_implicit", step0 = 0.001, maxStep = 0.005)
        logging.info(f"throat by {self.nsolver.nozzle.xt}")
        self.nsolver.plotsol(res_crit, title = f"choked nozzle with friction = {self.nsolver.frictionCoef}.\n with artifical kick by throat with {dv_kick} m/sec ")
        logging.info(res_crit.tail(1))
        self.primNozzleFlow = res_crit
        return res_crit

    def premix(self, res_crit):
        """ solving the premix equations. this will calculate the secondary mass flow rate"""
        mixingParams = self.params["mixingParams"]
        self.mixer = EjectorMixer.EjectorMixer(self.fluid, self.ejector, mixingParams)
        self.mixer.setSuctionMassFlow(None)
        self.mixer.setSingleChoke(True)
        #self.mixer.ejector.Asi = 2 * 1.1 ** 2 * math.pi  # cm2 of the suction nozzle inlet.
        self.mixer.ejector.Asi = self.params["A_suction_inlet"]

        self.mixerin = self.mixer.premixWrapSolve(res_crit, self.params["Psuc"], self.params["Tsuc"])

    def mixersolve(self):
        """ solve the mixer equations until the end of the ejector"""
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
        """ verify the mass flow conservation. Validate if the sum stays constant in the mixer. This is only used for debugging!
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
