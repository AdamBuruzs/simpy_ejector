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
from simpy_ejector.materialFactory import MaterialPropertiesFactory
from simpy_ejector import nozzleFactory, NozzleParams, nozzleSolver, EjectorGeom, EjectorMixer


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

    def __init__(self, params, fluid = "R1233zde", proplibrary= "refprop" ):
        """ ejector simulator object
        :param proplibrary: refprop or coolprop
        :param fluid: fluid name from refprop see https://pages.nist.gov/REFPROP-docs/#list-of-fluids or coolprop
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
        self.fluid = MaterialPropertiesFactory.create(library=proplibrary, material=fluid)
        # self.fluid =  fluid
        # self.RP = refProp.setup(self.fluid)
        self.makeEjectorGeom(params)
        if not 'hprim' in params.keys():
            Dprim, hp = self.fluid.getDh_from_TP(params['Tprim'], params['Pprim'])
            params['hprim'] = hp
        if not 'hsuc' in params.keys():
            Dsuc, hs = self.fluid.getDh_from_TP(params['Tsuc'], params['Psuc'])
            params['hsuc'] = hs
        self.dv_kick = 2.0
        self.setup_solver() # use the default values, or call after init the setup solver with your parameters!

    def makeEjectorGeom(self, params):
        """ create the ejector geometry.
        You can also use this function to update ejector geometry without reloading the whole object,
        and recalculating for example the critical speed"""
        #params = self.params
        Rin = params["Rin"]  # cm
        Rt = params["Rt"]  # 0.22
        Rout = params["Rout"]
        Dmix = params["Dmix"]
        if ("gamma_conv" in params) and ("gamma_div" in params):
            # calculate convergend and divergent part lengths from the angles
            gamma_conv = params["gamma_conv"]  # grad
            gamma_div = params["gamma_div"]  # grad
            Lcon = (Rin - Rt) / math.tan(gamma_conv * math.pi / 180)
            logging.info(f"Primary nozzle: inlet Radius {Rin} cm,\n Throat radius {Rt} cm \n convergent lenght {Lcon} cm")
            Ldiv = (Rout - Rt) / math.tan(gamma_div * math.pi / 180)
        else :
            Lcon = params["Lcon"]
            Ldiv = params["Ldiv"]
        logging.info(f"Primary nozzle: Outlet Radius {Rout} cm,\n divergent length {Ldiv}")
        nozzle = nozzleFactory.ConicConic(Rin=Rin, Lcon=Lcon, Rt=Rt, Ldiv=Ldiv, Rout=Rout)
        logging.info(
            f"Primary nozzle: Rin ={Rin}, Rout = {Rout} Lcon = {Lcon}, Rt = {Rt} cm,\n converg Len {round(Lcon, 5)} divergent length {round(Ldiv, 5)} ")
        if "motive_friction" in params:
            mot_friction = params["motive_friction"]
        else:
            mot_friction = 1.0e-3
        nozzle.setFriction(mot_friction)

        ejector = EjectorGeom(nozzle, Dm=params["Dmix"])
        if "dx_mixstart" in params:  # The distance of the Munday-Bagster hypothetical throat from the LAval nozzle end
            dx_mix = params["dx_mixstart"]
        else:
            dx_mix = 1.0 # a fix cm value
        mixstart = Lcon + Ldiv + dx_mix
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
        # [Din, hin] = self.fluid.getDh_from_TP(self.params['Tprim'], self.params['Pprim'])
        self.nsolver = nozzleSolver.NozzleSolver(nozzle, self.fluid, 1, solver="AdamAdaptive", mode="basic")
        self.nsolver.setFriction(mot_friction)

    def calcPrimMassFlow(self, plotCrit0 = False, chokePos="divergent_part"):
        """calculate the motive nozzle critical speed and choking mass flow rate
        This function sets the self.nsolver!!
        :param plotCrit0: plot the last converged (critical) flow? This is maybe still the last subsonic flow.
        """
        self.makeEjectorGeom(self.params)
        nozzle = self.ejector.nozzle
        logging.info(f" prim press {self.params['Pprim']} kPa, sec press {self.params['Psuc']} kPa ")
        # self.nsolver = nozzleSolver.NozzleSolver(nozzle, self.fluid, 1, solver="AdamAdaptive", mode="basic")
        # self.nsolver.setFriction(1e-2)

        # RP = refProp.setup(self.fluid)
        # this is not good if quality > 0 by entry:
        # [Din, hin] = self.fluid.getDh_from_TP( self.params['Tprim'], self.params['Pprim'])

        vin_crit = self.nsolver.calcCriticalSpeed( self.params['Pprim'], self.params['hprim'], v0 = 0.1, maxdev=1e-3,
                                                   maxStep = 0.05, chokePos="divergent_part")

        nozzle_crit0 = self.nsolver.solveNplot(vin_crit, self.params['Pprim'],  self.params['hprim'], doPlot=plotCrit0)

        logging.info(f"calculated critical choking inlet speed = {round(vin_crit, 5)} m/s")
        mass_flow_crit = vin_crit * self.fluid.getTD(  self.params['hprim'], self.params['Pprim'])['D'] * self.nsolver.nozzle.Aprofile(0) * 1e-4
        logging.info(f"critical mass flow is {round(mass_flow_crit, 5)} kg/sec")
        #results = params
        self.params["vin_crit"] = vin_crit
        self.params["mass_flow_crit"] = mass_flow_crit
        #return ejector,results, nsolver

    def setup_solver(self, step0_mn = 0.001, maxStep_mn = 0.005):
        """Setting parameters of the solver of the motive nozzle
        :param step0_mn: initial step size (dx) used for the motive nozzle solution
        :param maxStep_mn: maximal step size (dx) used for the motive nozzle solution
        """
        self.step0_mn = step0_mn
        self.maxStep_mn = maxStep_mn

    def motiveSolver(self):
        """Obtain the motive nozzle solution with kick-helper.
        This kick will help to reach the supersonic flow in the primary nozzle at the throat.

        :return: pandas DataFrame with the flow parameters of the critical flow
        for each x (measured in cm) integration points
        """
        sol_1 = self.nsolver.solveAdaptive1DBasic(self.params["vin_crit"], self.params["Pprim"],
                                             self.params["hprim"], 0.0, self.nsolver.nozzle.xt, step0 = self.step0_mn, maxStep = self.maxStep_mn )
        vph_throat = sol_1.iloc[-1]
        v = vph_throat["v"]
        p = vph_throat["p"]
        h = vph_throat["h"]
        logging.debug(f"critical solution at the throat : {sol_1.iloc[-1]}")
        #dv_kick = 2.0 ## [m/s] increase this value, if the flow does not switch to supersonic after the throat
        dp_kick = self.nsolver.pFromV_MassConst(v = vph_throat["v"], dv = self.dv_kick, p = vph_throat["p"], h = vph_throat["h"])
        logging.info(f"mass conserving artificial kick: dv = {self.dv_kick} m/s, dp = {dp_kick} kPa, throat pressure {p}")
        if abs(dp_kick) > p:
            logging.error("pressure kick is too large, reduce esim.dv_kick ! ")
        res_crit = self.nsolver.solveKickedNozzle(self.params["vin_crit"], self.params["Pprim"], self.params["hprim"], kicks = {'v': self.dv_kick, 'p': -dp_kick},
                                             solver= "adaptive_implicit", step0 = self.step0_mn, maxStep = self.maxStep_mn )
        logging.debug(f" dv = {self.dv_kick} at throat {self.nsolver.nozzle.xt}. res_crit at the end: \n{res_crit.iloc[-1]} ")
        if res_crit.iloc[-1]['v'] < sol_1.iloc[-1]['v']: # the flow did not became supersonic
            logging.info("velocity kick was too low, increasing it and try again")
            dv_step = 1 # m/sec
            for ii in range(20):
                self.dv_kick = self.dv_kick + dv_step # 1 m/s steps increase of the velocity kick
                dp_kick = self.nsolver.pFromV_MassConst(v=vph_throat["v"], dv=self.dv_kick, p=vph_throat["p"],
                                                        h=vph_throat["h"])
                logging.info(f"new guess for mass conserving artificial kick: dv = {self.dv_kick} m/s, dp = {dp_kick} kPa")
                res_crit = self.nsolver.solveKickedNozzle(self.params["vin_crit"], self.params["Pprim"],
                                                          self.params["hprim"], kicks={'v': self.dv_kick, 'p': -dp_kick},
                                                          solver="adaptive_implicit",  step0 = self.step0_mn, maxStep = self.maxStep_mn )
                logging.info(f"speed throat {sol_1.iloc[-1]['v']:.2f}, outlet {res_crit.iloc[-1]['v']:.2f}")
                if (res_crit.iloc[-1]['v'] > sol_1.iloc[-1]['v']):
                    break
        logging.info(f"throat by {self.nsolver.nozzle.xt}")
        self.nsolver.plotsol(res_crit, title = f"choked nozzle with friction = {self.nsolver.frictionCoef}.\n "
                                               f"with artifical kick by throat with {self.dv_kick} m/sec ")
        logging.info(f"-motiveSolver result tail {res_crit.tail(1)}")
        self.primNozzleFlow = res_crit
        return res_crit

    def setupMixer(self):
        """ just set up the mixer for the calculations, you can still modify the mixer calculation parameters"""
        mixingParams = self.params["mixingParams"]
        self.mixer = EjectorMixer.EjectorMixer(self.fluid, self.ejector, mixingParams)
        self.mixer.setSuctionMassFlow(None)
        self.mixer.setSingleChoke(True)
        # self.mixer.ejector.Asi = 2 * 1.1 ** 2 * math.pi  # cm2 of the suction nozzle inlet.
        self.mixer.ejector.Asi = self.params["A_suction_inlet"]

    def solvePremix(self, res_crit):
        """ just solving the pre-mix equations for secondary mass flow rate"""
        self.mixerin = self.mixer.premixWrapSolve(res_crit, self.params["Psuc"], self.params["Tsuc"])
        self.massFlowSec = self.mixerin["massFlowSecond"]
        self.massFlowPrim = self.mixerin["massFlowPrim"]

    def premix(self, res_crit):
        """ solving the premix equations. this will calculate the secondary mass flow rate"""
        mixingParams = self.params["mixingParams"]
        self.mixer = EjectorMixer.EjectorMixer(self.fluid, self.ejector, mixingParams)
        self.mixer.setSuctionMassFlow(None)
        self.mixer.setSingleChoke(True)
        if "premixMomCalcType" in self.params:
            self.mixer.momCalcType = self.params["premixMomCalcType"]
        #self.mixer.ejector.Asi = 2 * 1.1 ** 2 * math.pi  # cm2 of the suction nozzle inlet.
        self.mixer.ejector.Asi = self.params["A_suction_inlet"]

        self.mixerin = self.mixer.premixWrapSolve(res_crit, self.params["Psuc"], self.params["Tsuc"])
        self.massFlowSec = self.mixerin["massFlowSecond"]
        self.massFlowPrim = self.mixerin["massFlowPrim"]

    def mixersolve(self):
        """ solve the mixer equations until the end of the ejector"""
        mixerinput = [ self.mixerin[key] for key in [  'py', 'vpy', 'vsy', 'hpy', 'hsy', 'Apy', 'Asy']]
        # solve the initial value ODE:
        self.solMix = self.mixer.solveMix(mixerinput)
        self.diffout = self.solMix.iloc[-1] # diffuser output
        out_prim = self.fluid.getTD( hm=self.diffout["hp"], P=self.diffout["p"])
        out_sec = self.fluid.getTD( hm=self.diffout["hs"], P=self.diffout["p"])
        massFlowPrim = self.diffout["vp"] * self.diffout["Ap"] * out_prim['D'] * 1e-4 # kg/s
        massFlowSec =  self.diffout["vs"] * self.diffout["As"] * out_sec['D'] * 1e-4
        quality_tot =  (massFlowPrim * out_prim['q'] +  massFlowSec * out_sec['q'] ) / (massFlowPrim + massFlowSec)
        logging.info(f"diffuser outlet")
        logging.info(f"primary {round(self.diffout['vp'],2)} m/s with vapor q: { round(out_prim['q'],3) }. MFR {round(massFlowPrim,3)}")
        logging.info(f"secondary {round(self.diffout['vs'],2)} m/s with vapor q: { round(out_sec['q'],3) }. MFR {round(massFlowSec,3)}")
        logging.info(f"total q {quality_tot}")
        self.outlet_quality = quality_tot
        self.massFlowPrim_dout = massFlowPrim # at diffuser outlet
        self.massFlowSec_dout = massFlowSec

    def massFlowCheck(self):
        """ verify the mass flow conservation. Validate if the sum stays constant in the mixer. This is only used for debugging!
        solMix = the flow solution in the mixer and the diffuser part of the ejector
        """
        solMix = self.solMix
        Dp = solMix.apply(lambda x: self.fluid.getTD( hm=x['hp'], P=x['p'])['D'], axis=1)  # density primary flow
        qp = solMix.apply(lambda x: self.fluid.getTD( hm=x['hp'], P=x['p'])['q'], axis=1) # vapor quality
        solMix["MFRp"] = Dp * solMix['vp'] * solMix['Ap']*1e-4 # primary mass flow rate
        solMix["qp"] = qp
        Ds = solMix.apply(lambda x: self.fluid.getTD( hm=x['hs'], P=x['p'])['D'],  axis = 1) # density primary flow
        qs = solMix.apply(lambda x: self.fluid.getTD( hm=x['hs'], P=x['p'])['q'], axis=1)
        solMix["MFRs"] = Ds * solMix['vs'] * solMix['As']*1e-4
        solMix["qs"] = qs
        logging.info(f"mixer first sec density {Ds.head(1)}")
        solMix["q_average"] = (solMix["MFRp"]* solMix["qp"] + solMix["MFRs"] * solMix["qs"]) / ( solMix["MFRp"] + solMix["MFRs"] )

    def calcEfficiency(self):
        """Calculate Elbel efficiency """
        eff = self.mixer.calcEfficiency(pMnIn = self.params["Pprim"],
                             hinMn = self.params["hprim"],
                             pSucIn = self.params["Psuc"],
                             hinSuc = self.params["hsuc"],
                             pdiffOut = self.solMix["p"].values[-1] ,
                             massFlMn = self.massFlowPrim ,
                             massFlSuc= self.massFlowSec )
        return eff

    def calcPrimNozzleEfficiency(self ):
        """ Calculate the primary nozzle isentropic efficiency

        :return: ( h_{MN,in} - h_{MN,out} ) /(  h_{MN,in} - h_{MN,out,isentropic} )
        """
        # self.primNozzleFlow
        sInMn = self.fluid.getTD(self.primNozzleFlow.iloc[0]['h'], self.primNozzleFlow.iloc[0]['p'] )['s']
        d_h_real = self.primNozzleFlow.iloc[0]['h'] - self.primNozzleFlow.iloc[-1]['h']
        hMnIsentropic = self.fluid.get_from_PS( self.primNozzleFlow.iloc[-1]['p'], sInMn)['h']
        d_h_isentropic = self.primNozzleFlow.iloc[0]['h'] - hMnIsentropic
        eta_motive = d_h_real / d_h_isentropic
        return eta_motive

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
        cPrim = [self.fluid.getSpeedSound( solMix.iloc[i]['hp'], solMix.iloc[i]['p']) for i in range(solMix.__len__())]
        cSec = [self.fluid.getSpeedSound( solMix.iloc[i]['hs'], solMix.iloc[i]['p']) for i in
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
