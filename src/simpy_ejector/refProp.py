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
## original version by Adam Buruzs
# interface of simpy_ejector to ctrefprop
import numpy as np
import matplotlib.pyplot as plt
import math
import os, numpy as np
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 100)

def NBP_test():
    ''' test function from the tutorial'''
    print('The REFPROP directory:' + os.environ['RPPREFIX'])
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    print(RP.RPVersion())
    #    MOLAR_BASE_SI = RP.GETENUMdll(0, "MOLAR BASE SI").iEnum
    #    r = RP.REFPROPdll("PROPANE", "PQ", "T", MOLAR_BASE_SI, 0, 0, 101325, 0, [1.0])
    #    print(r.ierr, r.herr, r.Output[0])
    r = RP.SETUPdll(1, "BUTANE.FLD", "HMX.BNC", "DEF")
    assert (r.ierr == 0)
    # pressure quality flash
    print(RP.PQFLSHdll(101.325, 0, [1.0], 0))
    P = 150.0  # kPa
    ##  h = 1000.0 # kJ/kMol
    MM = RP.WMOLdll([1.0])  ## molar mass kg/kMol
    hm = 261.0  # kJ/kg
    h = hm * MM
    res = RP.PHFLSHdll(P, h, [1.0])
    print(res)
    print('Temp(K):{0} Density {1} g/liter'.format(res.T, res.D * MM))


def setup(material="BUTANE"):
    # print('The REFPROP directory:' + os.environ['RPPREFIX'])
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    # print(RP.RPVersion())
    r = RP.SETUPdll(1, material + ".FLD", "HMX.BNC", "DEF")
    assert r.ierr == 0, r.herr
    return RP


def getTD(RP, hm=100.0, P=100.0, debug=False):
    ''' get Temperature and Density from enthalpy and pressure

    :param RP: Refprop pointer
    :param hm: ethalpy in kJ/kg
    :param P: pressure in kPa
    :return: Temp K, Density in g/liter, quality, speed of sound in a dict
    '''
    MM = RP.WMOLdll([1.0])  ## molar mass kg/kMol
    h = hm * MM
    res = RP.PHFLSHdll(P, h, [1.0])
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


def getDh_from_TP(RP, T, p):
    ''' get Density and specific enthalpy [kJ/kg] from Temperature and pressure
    :param T: temperature in Kelvin
    :param p: pressure in kPa!
    :return : [Density in kg/m^3, spec enthalpy in kJ/kg]
    '''
    inprops = RP.TPFLSHdll(T, p, [1.0])
    MM = RP.WMOLdll([1.0])
    hin = inprops.h / MM
    Din = inprops.D * MM
    return [Din, hin]

def get_from_PS(RP, p , s):
    """Get quantities from pressure and specific etropy

    :param RP:
    :param p:
    :param s: specific entropy [kJ/kg/K]
    :return: dict with "T" : Temp [K], "D": density [kg/m^3] , "h" : spec enthalpy [kJ/kg]
    """
    MM = RP.WMOLdll([1.0])  ## molar mass kg/kMol
    smol = s * MM # [[J/mol-K]
    res = RP.PSFLSHdll(p, smol, [1.0])
    density = res.D * MM
    hmass = res.h / MM
    print(' h = {}'.format(res.h))
    return {"T": res.T, "D": density, "h": hmass }

def getTransport(RP, T,D):
    """ get transport properties, viscosity and thermal conductivity

    :param RP:
    :param T: Temperature (K)
    :param D: density g/l
    :return: eta - dynamical viscosity(uPa.s) <br />
           tcx - thermal conductivity(W / m.K)
    """
    MM = RP.WMOLdll([1.0])  ## molar mass kg/kMol
    Dmol = D / MM # molar density
    eta, tcx, ierr, herr = RP.TRNPRPdll(T,Dmol,[1.0])
    # eta - -viscosity(uPa.s)
    # tcx - -thermal conductivity(W / m.K)
    return eta, tcx


def getSpeedSound(RP, hm=100.0, P=100.0):
    ''' | get Speed of Sound  from enthalpy and pressure.
    | If the medium is 2 phase, the speed of sound is calculated with the Homogeneus Equilibrium Model
    | \\frac{1}{\rho^2c^2}=\frac{1-x}{\rho_l^2c_l^2}+\frac{x}{\rho_g^2c_g^2}
    :param RP: Refprop pointer
    :param hm: ethalpy in kJ/kg
    :param P: pressure in kPa
    '''
    res0 = getTD(RP, hm, P)
    if res0['c'] is not None:
        return (res0['c'])
    else:  # the media is 2 phase, calculate the quality (mass ratio first)
        q = res0['q']  # gas mass ratio (x)
        T = res0['T']  # temperature
        MM = RP.WMOLdll([1.0])
        # liquid = RP.TPFLSHdll( T,P , [1.0])
        densMol_liquid = RP.TPRHOdll(T, P, [1.0], 1, 0, 0).D
        densMol_vapor = RP.TPRHOdll(T, P, [1.0], 2, 0, 0).D
        RP.TPFL2dll(T, P, [1.0])  # not good
        therm_vap = RP.THERMdll(T, densMol_vapor, [1.0])
        therm_liq = RP.THERMdll(T, densMol_liquid, [1.0])
        cv = therm_vap.w  # speed of sound vapor
        cl = therm_liq.w
        mix = q / math.pow(densMol_vapor * MM * cv, 2.0) + (1 - q) / math.pow(densMol_liquid * MM * cl, 2.0)
        cmix = math.sqrt(1.0 / mix / math.pow(res0['D'], 2.0))  ## speed of sound for the 2 phase mixture
        return cmix


########### Functions for testing ::::

def speedSound_plot():
    fluid = "BUTANE"
    RP = setup(fluid)
    Ptest = 100.0  # kPa
    xv = list(range(100, 1000, 10))
    temp_v = [getTD(RP, xi, Ptest)["T"] for xi in xv]
    density_v = [getTD(RP, xi, Ptest)["D"] for xi in xv]
    quality_v = [getTD(RP, xi, Ptest)["q"] for xi in xv]
    plt.plot(xv, density_v)
    plt.plot(xv, quality_v)

    plt.figure(2)
    pvals = [100, 500, 2000]
    plt.subplot(211)
    for Ptest in pvals:
        csound_v = [getSpeedSound(RP, xi, Ptest) for xi in xv]
        plt.plot(xv, csound_v)
    plt.title(" Speed of sound for {} in frozen homogeneus equilibrium model ".format(fluid))
    plt.xlabel('specific enthalpy kJ/kg')
    plt.ylabel('speed of sound (m/s)')
    plt.legend(pvals, title="pressure kPa")
    plt.subplot(212)
    for Ptest in pvals:
        quality = [getTD(RP, xi, Ptest)['q'] for xi in xv]
        plt.plot(xv, quality)
    plt.ylabel('vapour mass rate')


def test2(RP, T, p):
    D , h = getDh_from_TP(RP, T,p)
    mu, tcx =  getTransport(RP,T,D)
    print("dyn viscosity : {} microPa*sec".format(mu))
    print("tcx : {} W/mK".format(tcx))

def densityDerivativesTest(RP, p = 1000, hmax = 400):
    # hx = np.linspace(300, 900, 500)
    hx = np.linspace(350, hmax, 500)
    res = pd.DataFrame()
    for hi in hx:
        props = getTD(RP, hi, p)
        D = props['D']
        c = props['c']
        cHEM = getSpeedSound(RP, hi, p)
        mu, k = getTransport(RP, props['T'], D)
        eps = 0.00001
        dDdh = (getTD(RP, hi + eps, p)['D'] - D) / eps / 1000.  # h in kJ/kg
        dDdp = (getTD(RP, hi, p + eps)['D'] - D) / eps / 1000.  # p in kPa
        out = {'p': [p] , 'h': [hi], 'D': [D], 'c': [c], 'cHEM' : [cHEM] , 'dDdh': [dDdh], 'dDdp': [dDdp], 'mu': [mu] }
        outpd = pd.DataFrame(out)
        res = res.append(outpd )
    plt.figure(3, figsize=(10, 8))
    plt.subplot(211)
    plt.title( 'pressure {} kPa'.format(p))
    plt.plot(res['h'], res['c'], linewidth = 2)
    plt.plot(res['h'], res['cHEM'], '-.')
    plt.plot(res['h'], res['D'])
    plt.xlabel(' h [kJ/kg]')
    plt.legend(['speed of sound', 'sound speed HEM' , 'density'])
    plt.subplot(212)
    plt.plot(res['h'], res['dDdh'])
    plt.plot(res['h'], res['dDdp'])
    plt.legend(['dDdh', 'dDdp'])
    plt.xlabel(' h [kJ/kg]')
    return(res)

def densityTest2(RP, eps = 0.001, h= 400, p = 1000, dp = 1, dh= 0.5):
    """ can we use partial derivatives ? """
    T = getTD(RP, h,p)['T']
    D = getTD(RP, h, p)['D']
    dDdh = (getTD(RP, h + eps, p)['D'] - D) / eps / 1000.  # h in kJ/kg
    dDdp = (getTD(RP, h, p + eps)['D'] - D) / eps / 1000.  # p in kPa
    diff1 = getTD(RP, h+dh, p + dp)['D'] - D
    diff2 = dDdh * dh * 1e3  + dDdp * dp * 1e3 #
    print("derivs : dDdh {} dDdp {}".format(dDdh, dDdp))
    print("diffs 1: {} 2 . {}".format(diff1, diff2) )

def satLines(RP):
    """ saturation lines """
    info = RP.INFOdll(1)
    MM = RP.WMOLdll([1.0])
    # info.Tc -critical Temp , info.Pc critical pressure in kPa
    crit = RP.TPFLSHdll(info.Tc, info.Pc, z= [1.0])
    crith = crit.h / MM
    pcrit = info.Pc
    satlines = pd.DataFrame()
    pPoints0 = np.power(10.0, np.linspace( np.log10(1.0), np.log10(info.Pc - 10), 20) )
    pPoints1 = np.linspace(pPoints0[-2], pPoints0[-1], 10)
    pPoints =np.unique( np.sort( np.append(pPoints0,pPoints1) ) )
    for pi in pPoints: # pi pressure in kPa
        satLT = RP.SATPdll(pi, [1.0], 1).T # saturation temperature

        liqL =  RP.PQFLSHdll(pi, 0, z= [1.0], kq = 1)
        vapL = RP.PQFLSHdll(pi, 1.0, z=[1.0], kq=1)
        hL = liqL.h / MM
        hV = vapL.h / MM
        props = RP.TPFLSHdll(T, pi, [1.0])
        out = {'p': [pi] , 'hL': [hL], 'hV': [hV] }
        outpd = pd.DataFrame(out)
        satlines = satlines.append(outpd )
    critLine = pd.DataFrame( { 'p': [info.Pc] , 'hL': [crith], 'hV': [crith] })
    satlines = satlines.append(critLine)
    return satlines

if __name__ == '__main__':
    # If the RPPREFIX environment variable is not already set by your installer (e.g., on windows),
    # then uncomment this line and set the absolute path to the location of your install of
    # REFPROP
    # os.environ['RPPREFIX'] = r'D:\Code\REFPROP-cmake\build\10\Release\\'

    # Print the version of REFPROP in use and the NBP
    # NBP_test()
    T= 250 # K
    p = 100 # kPa
    RP = setup('butane')
    test2(RP, T,p)

    testcase = 1

    if (testcase == 1):
        vals = densityDerivativesTest(RP, 1e3)

        plt.figure()
        plt.plot(vals['h'], vals['cHEM'])
        plt.plot(vals['h'], 1 / np.sqrt(vals['dDdp']))
        plt.legend(['cHEM formula', '1/sqrt(dDdp)'])


        for hi in np.linspace(260,350, 100):
            pc = 300
            print(' quality{} dens {} '.format( getTD(RP, hi, pc)['q'], getTD(RP, hi, pc) ['D']))
            densityTest2(RP, eps=0.0001, h=hi, p=pc, dp=1, dh=0.5)

    if (testcase == 2):
        pin = 2100
        Tin = 350
        RP = setup('CO2')
        [Din, hin] = getDh_from_TP(RP, Tin, pin)
        print( "D: {} h: {} ".format(Din, hin) )
        q1 = getTD(RP, hin,pin)
        print(" spec entropy {}".format(q1['s']) )

        q2 = get_from_PS(RP, pin, q1['s'])
        print(q2)
        satlines = satLines(RP)
        plt.plot( satlines['hL'], satlines ['p'])
        plt.plot( satlines['hV'], satlines ['p'])
        plt.yscale("log")