## implementation of matprop_gen with refprop fuctions

import matplotlib.pyplot as plt
import math
import os, numpy as np

import pandas as pd
from simpy_ejector.matprop_gen import MaterialProperties

class RefpropProperties(MaterialProperties):
    def __init__(self, material ):
        import ctREFPROP.ctREFPROP as RP
        import ctREFPROP
        RP = ctREFPROP.ctREFPROP.REFPROPFunctionLibrary(os.environ['RPPREFIX'])
        RP.SETPATHdll(os.environ['RPPREFIX'])
        # print(RP.RPVersion())
        r = RP.SETUPdll(1, material + ".FLD", "HMX.BNC", "DEF")
        assert (r.ierr == 0)
        self.RP = RP

    def getTD(self, hm=100.0, P=100.0, debug=False):
        ''' get Temperature and Density from enthalpy and pressure

        :param RP: Refprop pointer
        :param hm: ethalpy in kJ/kg
        :param P: pressure in kPa
        :return: Temp K, Density in g/liter, quality, speed of sound in a dict
        '''
        MM = self.RP.WMOLdll([1.0])  ## molar mass kg/kMol
        h = hm * MM
        res = self.RP.PHFLSHdll(P, h, [1.0])
        # print(res)

        density = res.D * MM
        quality = max(0.0, min(res.q, 1.0))
        speedsound = res.w
        specEntropy = res.s / MM  # [J/g/K] = [kJ/kg/K]
        if debug:
            print('Temp(K):{0}, Density {1} g/liter, quality: {2} Speed of Sound {3}'.
                  format(res.T, density, quality, speedsound))
        # c w--speed of sound [m/s]
        # c Cp, w are not defined for 2-phase states
        # c in such cases, a flag = -9.99998d6 is returned
        if speedsound < 0:
            speedsound = None
        return {"T": res.T, "D": density, "q": quality, "c": speedsound, "s": specEntropy}

    def getDh_from_TP(self, T, p, ):
        ''' get Density and specific enthalpy [kJ/kg] from Temperature and pressure
        :param p: pressure in kPa!
        :return : [Density in kg/m^3, spec enthalpy in kJ/kg]
        '''
        inprops = self.RP.TPFLSHdll(T, p, [1.0])
        MM = self.RP.WMOLdll([1.0])
        hin = inprops.h / MM
        Din = inprops.D * MM
        return [Din, hin]

    def get_from_PS(self, p, s):
        """Get quantities from pressure and specific etropy

        :param p: pressure in kPa
        :param s: specific entropy [kJ/kg/K]
        :return: dict with "T" : Temp [K], "D": density [kg/m^3] , "h" : spec enthalpy [kJ/kg]
        """
        MM = self.RP.WMOLdll([1.0])  ## molar mass kg/kMol
        smol = s * MM  # [[J/mol-K]
        res = self.RP.PSFLSHdll(p, smol, [1.0])
        density = res.D * MM
        hmass = res.h / MM
        print(' h = {}'.format(res.h))
        return {"T": res.T, "D": density, "h": hmass}

    def getTransport(self, T, D):
        """ get transport properties, viscosity and thermal conductivity

        :param T: Temperature (K)
        :param D: density g/l
        :return: eta - dynamical viscosity(uPa.s) <br />
               tcx - thermal conductivity(W / m.K)
        """
        MM = self.RP.WMOLdll([1.0])  ## molar mass kg/kMol
        Dmol = D / MM  # molar density
        eta, tcx, ierr, herr = self.RP.TRNPRPdll(T, Dmol, [1.0])
        # eta - -viscosity(uPa.s)
        # tcx - -thermal conductivity(W / m.K)
        return eta, tcx

    def getSpeedSound(self, hm=100.0, P=100.0):
        ''' | get Speed of Sound  from enthalpy and pressure.
        | If the medium is 2 phase, the speed of sound is calculated with the Homogeneus Equilibrium Model
        | \\frac{1}{\rho^2c^2}=\frac{1-x}{\rho_l^2c_l^2}+\frac{x}{\rho_g^2c_g^2}
        :param hm: ethalpy in kJ/kg
        :param P: pressure in kPa
        '''
        res0 = self.getTD( hm, P)
        if res0['c'] is not None:
            return (res0['c'])
        else:  # the media is 2 phase, calculate the quality (mass ratio first)
            q = res0['q']  # gas mass ratio (x)
            T = res0['T']  # temperature
            MM = self.RP.WMOLdll([1.0])
            # liquid = RP.TPFLSHdll( T,P , [1.0])
            densMol_liquid = self.RP.TPRHOdll(T, P, [1.0], 1, 0, 0).D
            densMol_vapor = self.RP.TPRHOdll(T, P, [1.0], 2, 0, 0).D
            self.RP.TPFL2dll(T, P, [1.0])  # not good
            therm_vap = self.RP.THERMdll(T, densMol_vapor, [1.0])
            therm_liq = self.RP.THERMdll(T, densMol_liquid, [1.0])
            cv = therm_vap.w  # speed of sound vapor
            cl = therm_liq.w
            mix = q / math.pow(densMol_vapor * MM * cv, 2.0) + (1 - q) / math.pow(densMol_liquid * MM * cl, 2.0)
            cmix = math.sqrt(1.0 / mix / math.pow(res0['D'], 2.0))  ## speed of sound for the 2 phase mixture
            return cmix

