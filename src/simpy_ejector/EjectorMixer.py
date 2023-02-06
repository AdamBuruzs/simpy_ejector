## original version by Adam Buruzs (aburuzs@gmail.com)
## Solving 0 and 1 D equations for Ejector flow and motive/suction mixing calculations
# flow media is 1 comhonent material, where 2 phase is allowed
# liquid -vapor phase transition is modelled by Homogeneus Equilibrium Model
# written by Dr Adam Buruzs
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import logging
import scipy.optimize
from simpy_ejector import EjectorGeom, numSolvers, refProp
from simpy_ejector.flowSolver import FlowSolver

class EjectorMixer(FlowSolver) :

    def __init__(self, fluid, ejector : EjectorGeom, mixingParams):
        """ class for ejector mixing region and diffuser calculation
        :param RP: refprop object
        """
        super().__init__(fluid)
        self.ejector = ejector
        self.mixingParams = mixingParams
        self.setAreaDeriv(ejector.mixerdAdx, ejector.mixerArea)
        self.massFlowSecond = None
        self.singleChoke = True

    def setSuctionMassFlow(self, massFlowSecond):
        """ set secondary/ suction mass flow rate
        :param massFlowSecond: mass flow of suction nozzle in g/s
        """
        self.massFlowSecond = massFlowSecond
        if massFlowSecond is None: # delete this attribute
            delattr(self, "massFlowSecond")

    def setSingleChoke(self, is_singleChoke : bool):
        """ set with True, if single Choking calculation is required. If False, then Double choking will be calculated.

        If massFlowSecond is not set, then it will be calculated (self.premixEquationsSCMom),
         otherwise the inputted massFlowSecond will be used (self.premixEquationsSingleChoked)
        """
        self.singleChoke = is_singleChoke


    def premixEquationsDoubleChoked(self, parameters: np.ndarray,
                        variables ):
        """ System of equations to be solved for the pre-mixing part of the Ejector
        Assumed a double choking i.e. choked suction flow

        | massFlowPrim: primary mass flow rate
        | ho: motive nozzle output specific enthalpy
        | vo: motive nozzle output velocity
        | so: motive nozzle output specific entropy
        | hst: suction stagnation specific enthalpy [kJ/kg]
        | sst: suction stagnation specific entropy
        | Am: mixer cross section area [cm^2]
        | py: pressure at the hypothetical throat
        | vpy: primary flow speed at the hypothetical throat
        | hpy: primary flow spec enthalpy at the hypothetical throat
        | Dpy: primary flow density at the hypothetical throat
        | Apy: primary flow cross section area at the hypothetical throat [cm^2]
        | vsy: scondary flow speed at the hypothetical throat
        | hsy:
        | Dsy:
        | Asy: secondary flow cross section area at the hypothetical throat [cm^2]
        | massFlowSecond: secondary mass flow rate
        :param RP: Nist Refprop object
        :return: a vector of size 10 that should be solved for root
        """
        # TODO: check dimensions
        RP = self.RP
        [massFlowPrim, ho, vo, so, hst, sst, Am] = parameters
        [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy, massFlowSecond] = variables
        massp = massFlowPrim - Dpy * vpy * Apy / 10.0  # [ g/s] Apy in cm^2
        enp = ho + math.pow(vo, 2.0) / 2.0 * 1e-3 - hpy - math.pow(vpy, 2.0) / 2.0 * 1e-3  # kJ/kg
        dp = Dpy - refProp.getTD(RP, hpy, py)['D']
        Sp = so - refProp.getTD(RP, hpy, py)['s'] #  [kJ/kg/K]
        masssec = massFlowSecond - Dsy * vsy * Asy / 10.0  # this line is in [ g/s], Asy is in cm^2
        ens = hst - hsy - math.pow(vsy, 2.0) / 2.0 * 1e-3 # kJ/kg
        ds = Dsy - refProp.getTD(RP, hsy, py)['D']
        Ss = sst - refProp.getTD(RP, hsy, py)['s']
        dA = Am - Asy - Apy # [cm^3]
        dc = vsy - refProp.getSpeedSound(RP, hsy, py) # [m/s]
        out = [massp, enp, dp, Sp, masssec, ens, ds, Ss, dA, dc]
        return out

    def premixEquationsSingleChoked(self, parameters: np.ndarray, massFlowSecond,
                        variables ):
        """| System of equations to be solved for the pre-mixing part of the Ejector
        | Assumed a single choking i.e. only the motive/primary flow is choked, but the suction flow is not choked
        so the secondary mass flow rate - massFlowSecond - is a parameter, not a variable

        | massFlowPrim: primary mass flow rate
        | ho: motive nozzle output pressure
        | vo: motive nozzle output velocity
        | so: motive nozzle output specific entropy
        | hst: suction stagnation specific enthalpy [kJ/kg]
        | sst: suction stagnation specific entropy
        | Am: mixer cross section area [cm^2]
        | py: pressure at the hypothetical throat
        | vpy: primary flow speed at the hypothetical throat
        | hpy: primary flow spec enthalpy at the hypothetical throat
        | Dpy: primary flow density at the hypothetical throat
        | Apy: primary flow cross section area at the hypothetical throat [cm^2]
        | vsy: scondary flow speed at the hypothetical throat
        | hsy:
        | Dsy:
        | Asy: secondary flow cross section area at the hypothetical throat [cm^2]
        | massFlowSecond: secondary mass flow rate
        :param RP: Nist Refprop object
        :return: a vector of size 10 that should be solved for root
        """
        # TODO: check dimensions
        RP = self.RP
        [massFlowPrim, ho, vo, so, hst, sst, Am] = parameters
        [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy ] = variables
        massp = massFlowPrim - Dpy * vpy * Apy / 10.0  # [ g/s] Apy in cm^2
        enp = ho + math.pow(vo, 2.0) / 2.0 * 1e-3 - hpy - math.pow(vpy, 2.0) / 2.0 * 1e-3  # kJ/kg
        dp = Dpy - refProp.getTD(RP, hpy, py)['D']
        Sp = so - refProp.getTD(RP, hpy, py)['s'] #  [kJ/kg/K]
        masssec = massFlowSecond - Dsy * vsy * Asy / 10.0  # this line is in [ g/s], Asy is in cm^2
        ens = hst - hsy - math.pow(vsy, 2.0) / 2.0 * 1e-3 # kJ/kg
        ds = Dsy - refProp.getTD(RP, hsy, py)['D']
        Ss = sst - refProp.getTD(RP, hsy, py)['s']
        dA = Am - Asy - Apy # [cm^3]
        ## No condition for the suction speed by y, no choking occurs.
        ## dc = vsy - refProp.getSpeedSound(RP, hsy, py) # [m/s]
        out = [massp, enp, dp, Sp, masssec, ens, ds, Ss, dA]
        return out

    def premixEquationsSCMom(self, pars: dict,
                        variables ):
        """| System of equations to be solved for the pre-mixing part of the Ejector
        | Assumed a single choking i.e. only the motive/primary flow is choked, but the suction flow is not choked
        an approximation of the momentum equation is used additionaly, so that the mass flow rate is also calculated

        SCMom: Single Choking with Momentum equation

        | massFlowPrim: primary mass flow rate
        | ho: motive nozzle output pressure
        | vo: motive nozzle output velocity
        | so: motive nozzle output specific entropy
        | hst: suction stagnation specific enthalpy [kJ/kg]
        | sst: suction stagnation specific entropy
        | Am: mixer cross section area [cm^2]
        | py: pressure at the hypothetical throat
        | vpy: primary flow speed at the hypothetical throat
        | hpy: primary flow spec enthalpy at the hypothetical throat
        | Dpy: primary flow density at the hypothetical throat
        | Apy: primary flow cross section area at the hypothetical throat [cm^2]
        | vsy: scondary flow speed at the hypothetical throat
        | hsy: secondary specific enthalpy
        | Dsy: suction flow density
        | Asy: secondary flow cross section area at the hypothetical throat [cm^2]
        | massFlowSecond: secondary mass flow rate
        :return: a vector of size 10 that should be solved for root
        """
        # TODO: check dimensions
        RP = self.RP # Nist Refprop object
        # [massFlowPrim, ho, vo, so, hst, sst, Am] = parameters
        [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy, massFlowSecond ] = variables
        massp = pars['massFlowPrim'] - Dpy * vpy * Apy / 10.0  # [ g/s] Apy in cm^2
        enp = pars['ho'] + math.pow(pars['vo'], 2.0) / 2.0 * 1e-3 - hpy - math.pow(vpy, 2.0) / 2.0 * 1e-3  # kJ/kg enthalpy primary
        dp = Dpy - refProp.getTD(RP, hpy, py)['D'] # density primary at Y
        Sp = pars['so'] - refProp.getTD(RP, hpy, py)['s'] #  [kJ/kg/K] entropy primary
        masssec = massFlowSecond - Dsy * vsy * Asy / 10.0  # this line is in [ g/s], Asy is in cm^2
        ens = pars['hst'] - hsy - math.pow(vsy, 2.0) / 2.0 * 1e-3 # kJ/kg enthalpy suction flow at Y
        ds = Dsy - refProp.getTD(RP, hsy, py)['D']
        Ss = pars['sst'] - refProp.getTD(RP, hsy, py)['s'] # entropy suction flow
        dA = pars['Am'] - Asy - Apy # [cm^2] the cross section are equation
        ## No condition for the suction speed by y, no choking occurs.
        ## dc = vsy - refProp.getSpeedSound(RP, hsy, py) # [m/s]
        # the momentum equation for the suction flow
        dMomS = 0.5 * (pars['Dsi'] * pars['Asi'] + Dsy * Asy) * \
                (1.0 - (Dsy * Asy  / pars['Dsi'] / pars['Asi'])** 2.0 ) * vsy ** 2.0 - \
                (pars['Asi'] + Asy) * (pars['psi'] - py) * 1000.0 # p is in kPa
        out = [massp, enp, dp, Sp, masssec, ens, ds, Ss, dA, dMomS]
        return out

    def solvePreMix(self, params: np.ndarray, po=100.0):
        """| solve the pre-mix chamber equations numerically.
        In case of external input secondary mass flow rate, or double choking condition
        this is the old version. For calculating secondary mass flow rate for single choking case use solvePreMixSingleChoke

        :param params: [massFlowPrim, ho, vo, so, hst, sst, Am]
        | massFlowPrim: primary mass flow rate
        | ho: motive nozzle output pressure
        | vo: motive nozzle output velocity
        | so: motive nozzle output specific entropy
        | hst: suction stagnation specific enthalpy [kJ/kg]
        | sst: suction stagnation specific entropy
        | Am: mixer cross section area [m^2]
        :param po: nozzle exit pressure, only used to calculate initial values
        :param massFlowSecond: suction mass flow rate. If none, then double choking is assumed, in this case the mass flow rate is calculated.
        :return:
        """
        [massFlowPrim, ho, vo, so, hst, sst, Am] = params
        fluido = refProp.getTD(self.RP, ho, po)  # fluid properties by Nozzle exit
        Dinit = fluido['D']
        if self.singleChoke and  hasattr(self,  'massFlowSecond'): # secondary mass flow rate is passed as known parameter
            print(f"suction mass flow rate {self.massFlowSecond} g/s is used for pre-mix calculations" )
            eqFun = lambda x: self.premixEquationsSingleChoked(params, self.massFlowSecond, x)
            xinit = [100, vo, ho, Dinit, Am / 2.0, vo / 2.0, ho, Dinit, Am / 2.0]
            premix = scipy.optimize.root(eqFun, x0=np.array(xinit), method='hybr')
            # [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy, massFlowSecond] = variables
            [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy] = premix.x
            massFlowSecond = self.massFlowSecond
        elif not hasattr(self,  'massFlowSecond'): # secondary mass flow rate is calculated
            print('double choking calculation')
            eqFun = lambda x: self.premixEquationsDoubleChoked(params, x)
            xinit = [100, vo, ho, Dinit, Am / 2.0, vo / 2.0, ho, Dinit, Am / 2.0, massFlowPrim / 2.0]
            premix = scipy.optimize.root(eqFun, x0=np.array(xinit), method='hybr')
            # [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy, massFlowSecond] = variables
            [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy, massFlowSecond] = premix.x
        else:
            logging.error("for single chocking calculation with suction massFlowRate calculation set the "
                          "massFlowSecond attribute to None")
        resdict = { 'py': py, 'vpy' :vpy, 'hpy': hpy, 'Dpy': Dpy,
                    'Apy': Apy, 'vsy' : vsy, 'hsy': hsy,
                    'Dsy': Dsy, 'Asy': Asy, 'massFlowSecond': massFlowSecond }
        return [resdict, premix.x]

    def solvePreMixSingleChoke(self, params: dict, po=100.0):
        """| solve the pre-mix chamber equations numerically in case of single choking inclusive the approximative calculation
        of the secondary mass flow rate

        :param params: dictionary with [massFlowPrim, ho, vo, so, psi, Tsi, Am]
        | massFlowPrim: primary mass flow rate
        | ho: motive nozzle output pressure
        | vo: motive nozzle output velocity
        | so: motive nozzle output specific entropy
        | psi : suction inlet pressure in kPa
        | Tsi : suction inlet temperature
        | hst: suction stagnation specific enthalpy [kJ/kg]
        | sst: suction stagnation specific entropy
        | Am: mixer cross section area [cm^2]
        | Asi : suction inlet area
        :param po: nozzle exit pressure, only used to calculate initial values
        :param massFlowSecond: suction mass flow rate. If none, then double choking is assumed, in this case the mass flow rate is calculated.
        :return:
        """
        #[massFlowPrim, ho, vo, so, hst, sst, Am] = params
        #[massFlowPrim, ho, vo, so, ps,Ts, Am, Asi] = params
        [Dsi, hsi] = refProp.getDh_from_TP(self.RP, params['Tsi'], params['psi'])
        params["Dsi"] = Dsi
        params["hsi"] = hsi
        params["hst"] = hsi # inlet speed is low, the stagnation enthlpy is approximated by the inlet enthalpy
        suctionProps = refProp.getTD(self.RP, hsi, params['psi'])
        params["sst"] = suctionProps["s"]
        fluido = refProp.getTD(self.RP, params["ho"], po)  # fluid properties by Nozzle exit
        Dinit = fluido['D']
        params['Asi'] = self.ejector.Asi
        logging.info(f"solving equation in solvePreMixSingleChoke with parameters:\n {params}")
        eqFun = lambda x: self.premixEquationsSCMom(params, x)
        xinit = [params['psi'], params['vo'], params['ho'], Dinit,
                 params['Am'] / 2.0, params['vo'] / 2.0, params['ho'], suctionProps['D'], params['Am'] / 2.0, params['massFlowPrim'] / 2.0]
        ## TODO: better initialize the Dsy with the suction inlet density!!!
        premix = scipy.optimize.root(eqFun, x0=np.array(xinit), method='lm', tol= 1e-5 )
        #logging.info(f"the solution quality: {eqFun(premix.x)}")
        # [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy, massFlowSecond] = variables
        [py, vpy, hpy, Dpy, Apy, vsy, hsy, Dsy, Asy, massFlowSecond] = premix.x
        if premix.success == False:
            logging.error(f"{premix.message}\n solvePreMixSingleChoke has not converged, try to change xinit initial values, or the parameters of the scipy.optimize.root")
        else:
            logging.info(f"premix calculation finished after {premix.nfev} function evaluations:\n {premix.message}")
        logging.info(f" error terms in solving the solvePreMixSingleChoke: \n "
                     f"{eqFun(premix.x)}\n "
                     f"for the equations: [massp, enp, dp, Sp, masssec, ens, ds, Ss, dA, dMomS]")
        resdict = { 'py': py, 'vpy' :vpy, 'hpy': hpy, 'Dpy': Dpy,
                    'Apy': Apy, 'vsy' : vsy, 'hsy': hsy,
                    'Dsy': Dsy, 'Asy': Asy, 'massFlowSecond': massFlowSecond }
        return [resdict, premix.x]


    def premixWrapSolve(self, res_crit :pd.DataFrame, Psuc, Tsuc):
        """ wrapping the Premix solver, use the nozzle solver output (res_crit) and the Suction nozzle input
        pressure and Temperature, and the Ejector geometry, and calculate the premixing equations.

        :param res_crit: critical flow parameters, where the last line is the nozzle outlet values
        :param Psuc: suction nozzle input pressure
        :param Tsuc: suction nozzle input temperature
        :param ejector: Ejector geometry object
        :param RP: Nist RefPRop object
        :return: A dictionary with flow parameters at the pre-mix end
        """
        RP = self.RP
        nozzle = self.ejector.nozzle
        print('critical flow nozzle outlet quantities')
        print(res_crit.tail(1).transpose())
        nozzle_out = res_crit.tail(1).to_dict('records')[0]
        so = refProp.getTD(RP, nozzle_out['h'], nozzle_out['p'])['s']
        vo = nozzle_out['v']
        massFlowPrim = vo * nozzle_out['d'] * nozzle.Ao * 1e-4 * 1e3 # [g/sec]
        [Ds, hst] = refProp.getDh_from_TP(RP, Tsuc, Psuc)
        prop = refProp.getTD(RP, hst, Psuc)
        sst = prop['s']
        logging.info(f"suction flow by inlet Dens {Ds} g/l, quality {prop['q']}")
        if self.singleChoke and not hasattr( self, 'massFlowSecond'):
            logging.info("calculation of single-chocking subcritical mode incl. suction mass flow rate calculations")
            parameters = { 'Tsi':Tsuc, 'psi': Psuc, 'massFlowPrim': massFlowPrim,
                           'ho': nozzle_out['h'], 'vo' : vo, 'so': so, 'Am':  self.ejector.Am }
            print(parameters)
            mixres, mixar = self.solvePreMixSingleChoke(parameters, nozzle_out['p'])
            logging.info("pre-mix calculations with single choking and suction mass flow rate calculation")
        else:
            parameters = [massFlowPrim, nozzle_out['h'], vo, so, hst, sst, self.ejector.Am]
            # [massFlowPrim, ho, vo, so, hst, sst, Am] = parameters
            print(parameters)
            mixres, mixar = self.solvePreMix(parameters, nozzle_out['p'])
        print('at Y point after pre-mixing')
        print(mixres)
        print('Pressure[kPa] suction {} . primary nozzle exit : {} '.format(round(Psuc, 3), round(nozzle_out['p'], 3)))
        print("equalized pressure after pre-mixing : {} kPa".format(round(mixres['py'])))

        print('cross section area nozzle exit {} Mixer full: {} cm^2'.format(round(nozzle.Ao, 3), round(self.ejector.Am, 3)))
        print('A at pre-mix end (Y) suction/secondary {} A primary flow : {} cm^2'.format(round(mixres['Asy'], 4),
                                                                                          round(mixres['Apy'], 3)))
        print('mass flow [g/sec]  suction/secondary {} primary flow : {} '.format(round(mixres['massFlowSecond'], 4),
                                                                                  round(massFlowPrim, 3)))
        print("mass flow rate, entrainment ratio = {}".format( mixres['massFlowSecond'] / massFlowPrim ))
        out = mixres
        out['massFlowPrim'] = massFlowPrim
        return out


    def mixerEquations(self, variables, x):
        """ equations governing the mixer region mass, momentum and energy mixing between the primary/motive
         and secondary/suction flow

        :param parameters: dictionary with calculation parameters for friction, and mixing
        :param variables:  p- pressure
         vp -primary/motive flow speed
         vs - secondary/suction flow speed
         hp - primary flow specific enthalpy
         hs - secondary flow specific enthalpy
         Ap -primary flow cross section area
         As -suction flow cross section area
        :return: the derivatives of the variables
        """
        p,vp,vs,hpk,hsk,Ap,As = variables
        if (As > 0) :
            # hpk and hsk in kJ/kg, Ap and As in cm^2
            hfac = 1.e3 # factor to bring enthalpy to SI
            pSI,hpSI,hsSI,ApSI,AsSI = [p*1.e3, hpk*1.e3, hsk*1.e3,Ap*1.e-4,As*1.e-4] # in SI units
            eMat = np.zeros((7,7))
            Dp = refProp.getTD(self.RP, hpk, p)['D']
            Ds = refProp.getTD(self.RP, hsk, p)['D']
            eps = 0.01
            dDdhp = (refProp.getTD(self.RP, hpk + eps, p)['D'] - Dp) / eps / 1000. # hp was in kJ/kg
            dDdhs = (refProp.getTD(self.RP, hsk + eps, p)['D'] - Ds) / eps / 1000.
            dDdpp = (refProp.getTD(self.RP, hpk, p + eps)['D'] - Dp) / eps / 1000.
            dDdps = (refProp.getTD(self.RP, hsk, p + eps)['D'] - Ds) / eps / 1000.
            # eMat is the matrix of the linear equation system
            # mass conservation equations TODO: take care of units! hfac is used to
            eMat[0,:] = [1/Dp * dDdpp, 1/vp , 0.0, 1./Dp * dDdhp , 0.0, 1/Ap, 0.0 ]
            eMat[1,:] = [1/Ds * dDdps, 0.0, 1/vs , 0.0, 1./Ds * dDdhs , 0.0, 1/As ]
            # momentum equations:
            eMat[2, :] = [ApSI , ApSI * Dp * vp, 0.0, 0.0, 0.0, 0.0, 0.0 ]
            eMat[3, :] = [AsSI, 0.0, AsSI * Ds * vs, 0.0,  0.0, 0.0, 0.0 ]
            # energy equations:
            eMat[4, :] =  [0.0, ApSI * Dp * vp**2.0 , 0.0, ApSI * Dp *vp ,  0.0, 0.0, 0.0 ]
            eMat[5, :] =  [0.0, 0.0, AsSI * Ds * vs**2.0 , 0.0, AsSI * Ds *vs ,  0.0, 0.0 ]
            # cross section constrain
            eMat[6, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 ]

            parameters = self.mixingParams
            # dgdx = parameters['dgdx']  # the exchange mass flow
            # mass flow rate from primary to secondary
            circumferenceIface =  2.0 * math.sqrt(ApSI * math.pi)
            dms2p = parameters['massExchangeFactor'] * vp * circumferenceIface
            # parameters['momdx'] momentum transfer rate
            dragFac = parameters['dragFactor']
            momexchange = dragFac * (vp - vs)* abs(vp-vs) * circumferenceIface

            bvec = [ 1./ (vp * Dp * ApSI) * dms2p ,
                     - 1./ (vs * Ds * AsSI) * dms2p ]
            bvec += [ (vs -vp) * dms2p - momexchange - parameters['frictionInterface'],
                      -(vs -vp) * dms2p + momexchange - parameters['frictionWall'] ]
            bvec += [ ( hsSI - hpSI + 0.5 * (vp**2.0 - vs**2.0))* dms2p,
                    -( hsSI - hpSI + 0.5 * (vp**2.0 - vs**2.0))* dms2p]
            eps = 0.001
            if( x > self.ejector.ejectorlength - eps ):
                eps = -1 * eps
            dAdx = (self.ejector.mixerArea(x +eps) - self.ejector.mixerArea(x)) / eps # cm^2/cm
            bvec += [dAdx]
            # solving the linear equations for the derivatives :
            # eMat * [dp,dvp,dvs,dhp,dhs, Ap,As ] = bvec
            # p and h are in kilo, A stays in cm^2, because dAdx is also in cm^2
            unitvec = [1.e3, 1.0, 1.0, 1.e3, 1.e3, 1.0, 1.0]
            ematUnit = eMat* unitvec
            derivatives = np.linalg.solve(ematUnit, bvec)
            return derivatives
        else: # As <= 0, so the secondary flow already dissolved
            # p, vp, vs, hpk, hsk, Ap, As
            vph_in = np.array([vp,p,hpk])
            dv, dp, dh = self.simple_dvdpdh(vph_in, x)
            # the secondary/suction flow does not change, just the primary flow changes. The primary flow fills the ejector
            derivs = [dp, dv, 0.0, dh, 0.0, self.dAdx(x), 0.0]
            return np.array(derivs)


    def solveMix(self,  input, nint = 200, x0 = None) :
        """ solving the mixer and diffuser 1D equations
         :param input: initial variables
         :param nint: number of integration points
         :param x0: start of integration, if None then it automatically start by the mixer head
         """
        ## TODO: what if As goes under zero ?
        variables0 = input
        if x0 is None:
            x0 = self.ejector.mixerStart
        evolveFun = lambda v,x : self.mixerEquations(v,x)
        xvals = np.linspace(x0, self.ejector.ejectorlength, nint )
        sol = numSolvers.RungeKutta4(evolveFun, variables0, xvals)
        ## TODO : return a pandas DF with column names!
        solution = pd.DataFrame(sol, columns= ['x', 'p', 'vp', 'vs', 'hp', 'hs', 'Ap', 'As'] )
        return solution

    def plotSolution(self, solNozzle, solMix, title = "", ejectorPlot = None ):
        """ Plot the results of solveMix function
        :param solNozzle: solution of the nozzle
        :param solMix: the solution dataframe with data that we get from the solveMix function
        :return:
        """
        fig = plt.figure(figsize=[11,8])
        fig.suptitle("results of the 1D flow ejector simulation \n " + title)
        plt.subplot(311)
        # plt.plot(ejectorPlot)
        nozzleWall = [self.ejector.nozzle.Rprofile(xi) for xi in solNozzle['x'] ]
        mixerWall = [self.ejector.mixerR(xi) for xi in solMix['x'] ]
        plt.plot(solNozzle['x'], nozzleWall)
        plt.plot(solMix['x'], np.sqrt( solMix['Ap']/ math.pi))
        plt.plot(solMix['x'], mixerWall)
        plt.legend(['wall of Nozzle',  'primary motive stream', 'wall of Mixing region'])
        plt.ylabel("R [cm]")

        plt.subplot(312)
        plt.plot(solNozzle['x'], solNozzle['p'], color = 'blue')
        plt.plot(solMix['x'], solMix['p'], color = 'blue')
        # speeds
        plt.plot(solNozzle['x'], solNozzle['v'], color = "#aa1111")
        plt.plot(solMix['x'], solMix['vp'], color = "#aa1111")
        plt.plot(solMix['x'], solMix['vs'], color = "#aa9911")
        ## enthalpy
        plt.plot(solMix['x'], solMix['hp'])
        plt.legend([ 'pressure [kPa]', 'p [kPa]',
                     'prim flow speed', 'prim flow speed', 'suction flow speed',
                     'hp'])
        plt.yscale("log")
        plt.xlabel("x [cm]")
        ## Mach numbers :
        plt.subplot(313)
        plt.plot(solNozzle['x'], solNozzle['mach'])
        cPrim = [refProp.getSpeedSound(self.RP, solMix.iloc[i]['hp'], solMix.iloc[i]['p']) for i in range(solMix.__len__())]
        cSec = [refProp.getSpeedSound(self.RP, solMix.iloc[i]['hs'], solMix.iloc[i]['p']) for i in
                range(solMix.__len__())]
        plt.plot(solMix['x'], solMix['vp'] / cPrim )
        plt.plot(solMix['x'], solMix['vs'] / cSec)
        plt.legend(['primary Mach', 'primary Mach', 'secondary Mach'])
        plt.tight_layout()


    def mixingShock(self, x_sh, solNoShock, mergeflows = True, solver = "adaptiveImplicit" ):
        """ calculate mixing with a shock wave in xsh
        :param x_sh: shock wave position
        :param solNoShock: solution without a Normal Shock
        :param mergeflows: if True, flow downstream will be only 1 flow with 1 velocity.
         TODO : what if the 2 flows are not merged, how to calculate with 2 flows further?
         Maybe calculate 2 flows with variable cross section, and common pressure?
        :param solver: which solver to use,
        1. "adaptiveImplicit" - implicit equation flowsolver.vph_equation solved by adaptive integration
        2. "RK4" - explicit first order ODE : flowsolver.simple_dvdpdh solved by Runge Kutta 4.
        3. "BDF" - explicit first order ODE : flowsolver.simple_dvdpdh solved by scipy BDF method
        """
        assert solver in ["adaptiveImplicit", "RK4", "BDF"]
        assert x_sh >= self.ejector.mixerStart , " x_sh too small"
        q_upstream = FlowSolver.interpolate(solNoShock, x_sh, 'x')
        vph_prim = q_upstream[['x', 'vp', 'p', 'hp']]
        vph_sec = q_upstream[['x', 'vs', 'p', 'hs']]
        # print('upstream quantities \n' + str( q_upstream) )
        # downstream quantities
        #nsolver = NozzleSolver(self.RP,self.fl)
        # vph primary downstream
        [vp, p_p , hp, dp ] = self.calcNormalShock(vph_prim[1:])
        [vs, p_s , hs, ds ] = self.calcNormalShock(vph_sec[1:])
        # print('primary downstream quantities:')
        # print([vp, p_p , hp, dp ] )
        # print('secondary downstream quantities:')
        # print([vs, p_s , hs, ds ] )
        ## they have very different pressure downstream!!

        # run the solver from the shocked start
        shockUpStream = solNoShock[ solNoShock['x'] < x_sh]
        #secondpart = self.solveMix()
        [pU, vpU, vsU, hpU, hsU, Ap, As]  = q_upstream[['p', 'vp', 'vs', 'hp', 'hs', 'Ap', 'As']]
        DpU = refProp.getTD(self.RP, hpU, pU)['D']
        DsU = refProp.getTD(self.RP, hsU, pU)['D']
        if(mergeflows) :
            ##  calculate mass flow rate, p, and h
            totA = Ap + As
            j = (vpU * DpU * Ap + vsU * DsU * As ) / (Ap + As)
            ptermA = Ap * (pU + DpU * vpU**2.0 * 1.e-3 ) + As * (pU + DsU * vsU**2.0 * 1.e-3 )
            pterm = ptermA / totA
            htermA = (hpU + 0.5 * math.pow(vpU, 2.0) * 1.0e-3) * Ap + (hsU + 0.5 * math.pow(vsU, 2.0) * 1.0e-3) * As
            hterm = htermA / totA
            jph = [j, pterm, hterm]
            fun2solve = lambda vph: self.shock_jph(vph) - jph
            vphsc = scipy.optimize.root(fun2solve, x0=np.array([0.0, (vpU + vsU)/ 2.0 , (hpU + hsU)/2 ]),
                                        method='hybr')
            vdo, pdo, hdo = vphsc.x # the quantities after the shock, (shock downstream)
            print("after the shock starting with v,p,h = " + str([vdo, pdo, hdo]))
            if( x_sh == self.ejector.ejectorlength):
                # the shock is exactly at the end of the ejector, this is a corner case:
                allq = self.vph2all( np.array([x_sh, vdo, pdo, hdo]).reshape((1,-1)) )
                shockDstream = allq
                solMixShocked = shockUpStream.append(shockDstream, ignore_index=True)
                return (shockUpStream, shockDstream, solMixShocked)
            ## TODO : solve 1D flow for the rest of the diffuser
            dAdx = self.ejector.mixerdAdx
            AFun = lambda x : self.ejector.mixerR(x)**2.0 * math.pi
            self.setAreaDeriv(dAdx, AFun)
            self.setFriction( self.mixingParams['frictionWall'])
            ## this solution has problem with mass-flow conservation!: TODO : but why ? which solution is better?
            if (solver == "adaptiveImplicit"):
                shockDstream = self.solveAdaptive1DBasic(vdo, pdo, hdo, x0 = x_sh, endx= self.ejector.ejectorlength,
                                                        step0 = 0.02, maxStep= 0.1)
            elif (solver == "RK4"):
                shockDstream =self.solve1D(vdo, pdo, hdo, 200, endx = self.ejector.ejectorlength, startx= x_sh, odesolver= "RK4")
            elif (solver == "BDF"):
                shockDstream = self.solve1D(vdo, pdo, hdo, 200, endx=self.ejector.ejectorlength, startx=x_sh,
                                            odesolver="BDF")
            else :
                print ("Error, no such solver in EjectorMixer.mixingShock : {}".format(solver))
            print(shockDstream.head(5))
            afterP = pd.DataFrame({'x': shockDstream['x'], 'p': shockDstream['p'], 'vp': shockDstream['v'],
                                   'vs': shockDstream['v'],
                                   'hp': shockDstream['h'],
                                   'hs': shockDstream['h'],
                                   'Ap': [self.ejector.mixerArea(xi) for xi in shockDstream['x']],
                                   'As': np.repeat(0.0,shockDstream.shape[0])})
            solMixShocked = shockUpStream.append(afterP, ignore_index=True)
            return (shockUpStream, shockDstream, solMixShocked)

    def getChokePosition(self, pOutlet, solNoShock):
        """Calculate the position of the shockwave based on the diffuser outlet pressure
        :param pOutlet: diffuser outlet pressure in kPa
        :solNoShock: the hypothetical solution of the flow without normal shockwave
        """
        lastnoShock = solNoShock.iloc[-1]
        pmin = lastnoShock['p']
        if (pOutlet < pmin):
            message = "pressure is smaller than minimal pressure, underexpanded Ejector"
            xshock = None
            return {"message": message, "shock" : xshock }
        # shock wave at the very end:
        shockUpStream, shockDstream, solLast = self.mixingShock(self.ejector.ejectorlength, solNoShock, mergeflows=True, solver="adaptiveImplicit")
        # shock wave at the beginning of the mixer:
        shockUpStream, shockDstream, solFirst = self.mixingShock(self.ejector.mixerStart, solNoShock, mergeflows=True,
                                                                solver="adaptiveImplicit")
        pmax = solFirst.iloc[-1]['p']
        pSh = solLast.iloc[-1]['p']
        print("pressure critical points at the diffuser outlet: pmin {} pLastShock {} pmax {}".format(pmin, pSh, pmax))
        criticalPressures = {"pmin": pmin, "pLastShock" : pSh, "pmax" : pmax }
        if (pOutlet < pSh):
            message = "pressure is between minimal pressure {} and minimal shocked pressure {}, " \
                      "overexpanded Ejector".format(pmin, pSh)
            xshock = None
            return {"message": message, "shock": xshock, "pmin": pmin, "pLastShock" : pSh, "pmax" : pmax}
        if (pOutlet == pSh):
            return {"message": "shock at the beginning", "shock": self.ejector.mixerStart, "pmin": pmin, "pLastShock": pSh, "pmax": pmax}
        if (pOutlet > pmax):
            message = "pressure is over the maximal achieveable pressure"
            xshock = None
            return {"message": message, "shock": xshock, "pmin": pmin, "pLastShock" : pSh, "pmax" : pmax}
        if ((pOutlet < pmax) &  (pOutlet > pSh) ):
            message = "normal shock in the mixer"
            # calculating shock position
            transformer = lambda x : self.mixingShock(min(x, self.ejector.ejectorlength), solNoShock, mergeflows=True, solver="adaptiveImplicit")[2]
            condition = lambda sol : sol.iloc[-1]['p'] > pOutlet
            tol = 0.01
            xshock = numSolvers.findMaxTrue(self.ejector.mixerStart, transformer, condition, tol)
            return {"message": message, "shockPos": xshock, "pmin": pmin, "pLastShock" : pSh, "pmax" : pmax }

    def calcEfficiency(self, pMnIn, TMnIn, pSucIn, TSucIn, pdiffOut, massFlMn, massFlSuc ):
        """ Calculate the Ejector efficiency formula according to Elbel,Hrnjak 2008.

        :param pMnIn: motive nozzle inlet pressure
        :param TMnIn: motive nozzle inlet temperature
        :param pSucIn: suction nozzle inlet pressure
        :param TSucIn: suction nozzle inlet temperature
        :param pdiffOut: diffuser outlet pressure
        :param massFlMn: mass flow rate of the motive nozzle in kg/sec
        :param massFlSuc: mass flow rate of the suction nozzle in kg/sec
        :return: efficiency ratio
        """
        [DinMn, hinMn]  = refProp.getDh_from_TP(self.RP, TMnIn, pMnIn)
        sInMn = refProp.getTD(self.RP, hinMn, pMnIn)['s']

        [DinSuc, hinSuc] = refProp.getDh_from_TP(self.RP, TSucIn, pSucIn)
        sInSuc = refProp.getTD(self.RP, hinSuc, pSucIn)['s']

        hMnIsentrop = refProp.get_from_PS(self.RP, pdiffOut, sInMn)['h']
        hSucIsentrop = refProp.get_from_PS(self.RP, pdiffOut, sInSuc)['h']

        suctionWork = massFlSuc * (hSucIsentrop - hinSuc)
        maxPotential = massFlMn * (hinMn - hMnIsentrop )

        print(" Expansion work recovered : {} Watt. \n "
              " max expansion work recovery potential {} Watt".format(suctionWork, maxPotential))

        efficiency = suctionWork / maxPotential
        return efficiency



