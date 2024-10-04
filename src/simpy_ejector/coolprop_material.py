# general material properties abstract class
## implementation in coolprop_material and refprop_material
import logging
import os
import sys
import math

from simpy_ejector.matprop_gen import MaterialProperties
import CoolProp
from CoolProp.CoolProp import PropsSI



class CoolpropProperties(MaterialProperties):
    def __init__(self, material ):
        self.material = material

    def getTD(self, hm, P, debug=False):
        ''' get Temperature and Density from enthalpy and pressure

        :param hm: ethalpy in kJ/kg
        :param P: pressure in kPa
        :return: Temp K, Density in g/liter, quality, speed of sound, specific entropy in a dictionary
        '''
        # raise NotImplementedError("This method should be overridden by subclasses")
        Temp =  PropsSI('T','H', hm*1e3, 'P', P*1e3, self.material)
        dens =  PropsSI('D','H', hm*1e3, 'P', P*1e3, self.material)
        quality = PropsSI('Q','H', hm*1e3, 'P', P*1e3, self.material)
        try:
            speedsound = PropsSI('SPEED_OF_SOUND','H', hm*1e3, 'P', P*1e3, self.material)
        except:
            logging.error(f"coolprop: {sys.exc_info()}")
            speedsound = None
        specEntropy = PropsSI('SMASS', 'H', hm * 1e3, 'P', P * 1e3, self.material)
        return {"T": Temp, "D": dens, "q": quality, "c": speedsound, "s": specEntropy}



    def getDh_from_TP(self, RP, T, p, ):
        ''' get Density and specific enthalpy [kJ/kg] from Temperature and pressure
        be careful, when you use it in two phase region (partly melt)
        :param p: pressure in kPa!
        :return : [Density in kg/m^3, spec enthalpy in kJ/kg]
        '''
        dens = PropsSI('D', 'T', T, 'P', p * 1e3, self.material)
        enth = PropsSI('H', 'T', T, 'P', p * 1e3, self.material) /1.0e3 # in kJ/kg
        return [dens, enth]


    def get_from_PS(self, RP, p, s):
        """Get quantities from pressure and specific etropy

        :param RP:
        :param p:
        :param s: specific entropy [kJ/kg/K]
        :return: dict with "T" : Temp [K], "D": density [kg/m^3] , "h" : spec enthalpy [kJ/kg]
        """
        temp = PropsSI('T', 'S', s, 'P', p * 1e3, self.material)
        density = PropsSI('D', 'S', s, 'P', p * 1e3, self.material)
        hmass = PropsSI('H', 'S', s, 'P', p * 1e3, self.material) /1.0e3 # in kJ/kg
        return {"T": temp, "D": density, "h": hmass}

    def getTransport(self, T, D):
        """ get transport properties, viscosity and thermal conductivity

        :param RP:
        :param T: Temperature (K)
        :param D: density g/l
        :return: eta - dynamical viscosity(uPa.s) <br />
               tcx - thermal conductivity(W / m.K)
        """
        raise NotImplementedError("This method should be overridden by subclasses")
        visco =  PropsSI('VISCOSITY', 'T', T, 'D', D, self.material) * 1e6
        condu = PropsSI('CONDUCTIVITY', 'T', T, 'D', D, self.material) * 1e6
        return visco, condu


    def getSpeedSound(self, hm=100.0, P=100.0):
        ''' | get Speed of Sound  from enthalpy and pressure.
        | If the medium is 2 phase, the speed of sound is calculated with the Homogeneus Equilibrium Model
        | \\frac{1}{\rho^2c^2}=\frac{1-x}{\rho_l^2c_l^2}+\frac{x}{\rho_g^2c_g^2}
        :param RP: Refprop pointer
        :param hm: ethalpy in kJ/kg
        :param P: pressure in kPa
        '''
        res0 = self.getTD( hm, P)
        if res0['c'] is not None:
            return res0['c']
        else:  # the media is 2 phase, calculate the quality (mass ratio first)
            q = res0['q']  # gas mass ratio (x)
            T = res0['T']  # temperature
            hvap = PropsSI('H','T',T ,'Q',1,self.material)
            hliq = PropsSI('H', 'T', T, 'Q', 0, self.material)
            densvap = PropsSI('D', 'T', T, 'Q', 1, self.material)
            densliq = PropsSI('D', 'T', T, 'Q', 0, self.material)
            cvap = PropsSI('SPEED_OF_SOUND', 'T', T, 'Q', 1, self.material)
            cliq = PropsSI('SPEED_OF_SOUND', 'T', T, 'Q', 0, self.material)
            mix = q / math.pow(densvap * cvap, 2.0) + (1 - q) / math.pow(densliq * cliq, 2.0)
            cmix = math.sqrt(1.0 / mix / math.pow(res0['D'], 2.0))  ## speed of sound for the 2 phase mixture
            return cmix


if __name__ == "__main__" :
    allfluids = CoolProp.CoolProp.FluidsList()
    print(CoolProp.CoolProp.get_global_param_string("incompressible_list_pure"))
    ## 'IsoButane' , "water", "R1233zde",
    prop = CoolpropProperties('butane')
    for hp in range(0,700, 80):
        res = prop.getTD(hm=270.0 + hp, P=350.0)
        print(f"material data: {res}")

