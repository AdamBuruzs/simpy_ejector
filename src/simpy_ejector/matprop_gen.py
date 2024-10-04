# general material properties abstract class
## implementation in coolprop_material and refprop_material
import os


class MaterialProperties:

    def getTD(hm, P, debug=False):
        ''' get Temperature and Density from enthalpy and pressure

        :param RP: Refprop pointer
        :param hm: ethalpy in kJ/kg
        :param P: pressure in kPa
        :return: Temp K, Density in g/liter, quality, speed of sound, specific entropy in a dictionary
        '''
        raise NotImplementedError("This method should be overridden by subclasses")

    def getDh_from_TP(RP, T, p, ):
        ''' get Density and specific enthalpy [kJ/kg] from Temperature and pressure
        be careful, when you use it in two phase region (partly melt)
        :param p: pressure in kPa!
        :return : [Density in kg/m^3, spec enthalpy in kJ/kg]
        '''
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_from_PS(RP, p, s):
        """Get quantities from pressure and specific etropy

        :param RP:
        :param p:
        :param s: specific entropy [kJ/kg/K]
        :return: dict with "T" : Temp [K], "D": density [kg/m^3] , "h" : spec enthalpy [kJ/kg]
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def getTransport(RP, T, D):
        """ get transport properties, viscosity and thermal conductivity

        :param RP:
        :param T: Temperature (K)
        :param D: density g/l
        :return: eta - dynamical viscosity(uPa.s) <br />
               tcx - thermal conductivity(W / m.K)
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def getSpeedSound(RP, hm=100.0, P=100.0):
        ''' | get Speed of Sound  from enthalpy and pressure.
        | If the medium is 2 phase, the speed of sound is calculated with the Homogeneus Equilibrium Model
        | \\frac{1}{\rho^2c^2}=\frac{1-x}{\rho_l^2c_l^2}+\frac{x}{\rho_g^2c_g^2}
        :param RP: Refprop pointer
        :param hm: ethalpy in kJ/kg
        :param P: pressure in kPa
        '''
        raise NotImplementedError("This method should be overridden by subclasses")