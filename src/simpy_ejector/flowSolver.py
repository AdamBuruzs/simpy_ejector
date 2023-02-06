## original version by Adam Buruzs
import scipy.optimize
import scipy.integrate
import numpy as np
import pandas as pd
import math
from simpy_ejector import numSolvers, refProp
import logging

class FlowSolver(object):

    def __init__(self, fluid = "BUTANE") :
        """ general object for 1D fluid flow simulations
        containing function for normal shock wave solutions"""
        self.fluid = fluid
        self.RP = refProp.setup(fluid)

    def setFriction(self, friction):
        """| set the friciton coefficient that is the friction of the flow with the wall """
        self.frictionCoef = friction
        print("friction of the flow calc set to {}".format(friction))



    def setAreaDeriv(self , dAdxFun , AFun ):
        """ set the geometric constrains of the 1D fluid flow:
        the spatial derivative of the cross section (of the wall)

        :param dAdxFun: a function dAdx(x)
        :param AFun: function of the cross section area in cm^2 A(x)
        :return:
        """
        self.dAdx = dAdxFun
        self.AFun = AFun


    @staticmethod
    def interpolate(dataframe, xval, col='x'):
        '''linear interpolation of values'''
        rowl = (dataframe[col] > xval).idxmax() - 1
        rowr = rowl + 1
        a = dataframe.iloc[rowl]
        b = dataframe.iloc[rowr]
        xa = a[col]
        xb = b[col]
        res = a + (xval - xa) / (xb - xa) * (b - a)
        return res

    def shock_jph( self, vph: np.array):
        ''' The equation to be solved for the shock wave calculation
                :param vph: a vector with (velocity,pressure,enthalpy)
                :return: the right side of the equation, mass flux, pressure term, energy term'''
        [v, p, h] = vph
        c = refProp.getSpeedSound(self.RP, h, p)
        D = refProp.getTD(self.RP, h, p)['D']
        j = v * D
        pterm = p + D * math.pow(v, 2.0) * 1.0e-3  # kPa
        hterm = h + 0.5 * math.pow(v, 2.0) * 1.0e-3  # kJ/kg
        return np.array([j, pterm, hterm])

    def calcNormalShock(self, vph_upstream):
        """calculate normal shock wave v, p,h jumps
        :param vph_upstream: vector of upstream speed, pressure, enthalpy
        :return : vector with [vph and density] downstream after the shock wave front
        """
        leftside = self.shock_jph(vph_upstream)
        # print('upstream left side: ' + str(leftside))
        fun2solve = lambda vph: self.shock_jph(vph) - leftside
        # vph2 = numSolvers.NewtonND(fun2solve, x0 = np.array([0.0, vph_upstream[1] * 2.0 , vph_upstream[2] ]),
        #                            maxdev= 0.1, debug = True)
        vphsc = scipy.optimize.root(fun2solve, x0=np.array([0.0, vph_upstream[1] * 2.0, vph_upstream[2]]),
                                    method='hybr')
        v2, p2, h2 = vphsc.x
        D2 = refProp.getTD(self.RP, h2, p2)['D']
        return np.array([v2, p2, h2, D2])



    def basic_dvdpdh(self, vph: np.array, x: float):
        '''| DEPRECATED: does not conserve massflow rate!
        | the basic differential equation of  [speed, pressure, enthalpy] evolution
         for a 1D fluid flow without viscosity with a friction force on the wall
        :param vph: [velocity, pressure, spec enthalpy] in m/s, kPa, kJ/kg
        :param x: spatial location
        :return: [dv/dx, dp/dx, dh/dx] 3 dim numpy array
        '''
        [v, p, h] = vph
        c = refProp.getSpeedSound(self.RP, h, p)
        D = refProp.getTD(self.RP, h, p)['D']
        eps = 0.0001
        dAdx = self.dAdx(x)
        dvdx = v / (math.pow(v, 2.0) / math.pow(c, 2.0) - 1.0) * dAdx / self.AFun(x)
        if hasattr(self, 'frictionCoef'):
            dvdx = dvdx - self.frictionCoef * math.sqrt(math.pi / self.AFun(x)) * math.pow(v, 3.0) / (
                    math.pow(v, 2.0) - math.pow(c, 2.0))
        dpdx = -1 * D * v * dvdx / 1.e3  # kPascal
        dhdx = -1 * v * dvdx / 1.e3  # kJ/kg
        out = np.array([dvdx, dpdx, dhdx])
        return out

    def simple_dvdpdh(self, vph: np.array, x: float, capprox = True):
        '''| the basic differential equation of  [speed, pressure, enthalpy] evolution
         for a 1D fluid flow without viscosity with a friction force on the wall
        :param vph: [velocity, pressure, spec enthalpy] in m/s, kPa, kJ/kg
        :param x: spatial location
        :param capprox: use dp/dr = c^2 approximation and HEM speed of sound formula for 2 phase mixture ?
        :return: [dv/dx, dp/dx, dh/dx] 3 dim numpy array
        '''
        [v, p, h] = vph
        c = refProp.getSpeedSound(self.RP, h, p)
        D = refProp.getTD(self.RP, h, p)['D']
        eps = 0.001
        dDdh = (refProp.getTD(self.RP, h + eps, p)['D'] - D) / eps / 1000.  # h in kJ/kg
        dDdp = (refProp.getTD(self.RP, h, p + eps)['D'] - D) / eps / 1000.  # p in kPa
        dAdx = self.dAdx(x)
        left1 = - dAdx / self.AFun(x)
        if hasattr(self, 'frictionCoef'):
            #print("using friction {}".format(self.frictionCoef))
            left2 = - self.frictionCoef *  math.sqrt(math.pi / self.AFun(x) / 1.e-4) * math.pow(v, 2.0) * D / 100.
        else:
            left2 = 0.0
        leftside = [left1, left2, 0.0]
        eqMat = np.zeros((3, 3))  # the matrix of the equation system
        if not capprox:
            #print("calc with density derivatives")
            eqMat[0, :] = [ 1.0/ v, 1. / D * dDdp, 1. / D * dDdh]
            ## just for testing :
            # print('x {} left0 {} '.format(x, left1))
        else :
            eqMat[0, :] = [1.0 / v, 1. / D * 1./c**2.0, 0.0]
        eqMat[1, :] = [ v * D, 1.0, 0.0]
        eqMat[2, :] = [v, 0.0 , 1.0]
        # dvdx = v / (math.pow(v, 2.0) / math.pow(c, 2.0) - 1.0) * dAdx / self.AFun(x)
        # if hasattr(self, 'frictionCoef'):
        #     dvdx = dvdx - self.frictionCoef * math.sqrt(math.pi / self.AFun(x)) * math.pow(v, 3.0) / (
        #             math.pow(v, 2.0) - math.pow(c, 2.0))
        # dpdx = -1 * D * v * dvdx / 1.e3  # kPascal
        # dhdx = -1 * v * dvdx / 1.e3  # kJ/kg
        try:
            res = np.linalg.solve(eqMat, leftside)
            # print(eqMat)
            # print('leftvec : '+ str(leftside))
            # print('x : {:.4E} '.format(round(x, 10) ) +"res v',p',h',w' " + str(res) )
        except np.linalg.LinAlgError as err:
            print("linalg error occured at point x ={}, equtaion Matrix :".format(x))
            print(eqMat)
            print("v {} p {} h {} w {}".format(v, p, h))
            print("dDdp {} dDdh{}".format(dDdp, dDdh))
            print(err)
            print('massflow {}'.format(v * D * self.AFun(x) * 1e-4))
            res = np.zeros(4)* np.nan
            return res
        dvdxSI, dpdxSI, dhdxSI = res
        # print( 'massflow {}'.format(v * D * self.AFun(x) * 1e-4 ))
        [dvdx, dpdx, dhdx] = [dvdxSI, dpdxSI/ 1.0e3, dhdxSI/ 1.0e3 ] # we need p in kPa, h in kJ/kg
        out = np.array([dvdx, dpdx, dhdx])
        ## debugging:
        # print("derivs "+ str(out))
        # dvp = 1/v * dvdx + 1/D * dDdp * dpdx * 1e3 + 1/D * dDdh * dhdx * 1e3
        # print("dvdp = {}".format(dvp))
        # Dx = 0.01
        # dr1 = Dx * ( dDdp * dpdx * 1e3 + dDdh * dhdx * 1e3)
        # dr2 = (refProp.getTD(self.RP, h + dhdx* Dx, p + dpdx * Dx)['D'] - D)
        # print("density changes : {} vs {}".format(dr1, dr2))
        return out

    def vph_equation(self, vphold, xlast, deltaX, vphnew, constants):
        """Euler-like update of the vph [speed, pressure, enthalpy] vector based on mass,
        momentum and energy conservation. Mass flow conservation implicitly satisfied

        :param vphold: last values. v in [m/s], p in [kPa], h in [kJ/kg]
        :param xlast: last x value . in cm !
        :param deltaX: x value change . in cm!
        :param vphnew: new values
        :param constants: a dict with the "massflow" field in kg/sec
        """
        #print("sovle eq with friction {} ".format(self.frictionCoef))
        [vl,pl,hl] = vphold # last values v in [m/s] p in [kPa]
        [vn,pn,hn] = vphnew
        #frictionCoefficient = self.frictionCoef
        if hasattr(self, 'frictionCoef'):
            frictionCoefficient = self.frictionCoef
        else:
            frictionCoefficient = 0.0
        xnew = xlast + deltaX
        massflow = constants['massflow'] # in kg/sec
        Dnew = refProp.getTD(self.RP, hn, pn)['D']
        eq1 = vn - constants['massflow']/ self.AFun(xnew) / 1e-4 / Dnew
        # deltaX / 100. because deltaX is measured in cm, and the equation is in SI
        eq2 = constants['massflow']/ self.AFun(xnew) / 1e-4 * (vn-vl) + (pn-pl) * 1e3 + \
          deltaX /100. * frictionCoefficient * math.sqrt(math.pi / self.AFun(xlast) / 1.e-4) * constants['massflow']/ self.AFun(xnew) / 1e-4 * (vl + vn) / 2.0
        eq3 = 0.5 * ( math.pow(vn, 2.0) - math.pow(vl,2.0) ) + (hn-hl)*1e3
        return np.array([eq1,eq2, eq3])

    def vph_update(self, vphold, xlast, deltaX, constants, solver= 'myNewton'):
        """Euler like update of the vph flow quantities vector
        :return: the solution, and the error. If error is big, then the solution failed!!
        """
        eq2solve = lambda vph : self.vph_equation(vphold, xlast, deltaX, vph, constants)
        solver = 'myNewton' # or scipy
        if( solver == 'scipy'):
            vphnew = scipy.optimize.root(eq2solve, x0=np.array(vphold), method='broyden1', tol = 1e-7)
            out = np.array(vphnew.x)
            if (not vphnew.success):
                print(f" scipy root finding failed in vph_update at {xlast} + {deltaX}")
        elif(solver == 'myNewton'):
            (solND, Xvals) = numSolvers.NewtonND(eq2solve, np.array(vphold), alpha = 1.0, maxdev= 1e-5, debug = False)
            if solND is None:
                logging.info(f"Newton root finding failed at {xlast} + {deltaX}")
            out = solND
        if (out is not None):
            error = np.abs(eq2solve(out)).sum()
        else:
            error = None
        # if(error > 1e-2):
        #     print(" solver could not solve the update equation")
        #     print( "x {} the equation solved".format(round(xlast,4)) + str( eq2solve(out) ) )
        #     return None
        # else:
        return (out, error )

    def stepCorrection( vphD ):
            maxchange = np.abs(vphD).max()
            try:
                if( (maxchange < 10.) & (maxchange > 0.1 ) ) :
                    return 1 # stepsize was good
                elif (maxchange >= 10.): # stepsize too big
                    return 5.0 / maxchange
                elif maxchange == 0.0: # no change
                    # print("stepsize does not change")
                    return 1
                else: # maxchange < 0.1 we need bigger stepsize
                    out = 1.0/ maxchange
                    return out
            except RuntimeWarning:
                print(" warning occured in stepCorrection, maxchange {}".format(maxchange) )
                return None



    def solveAdaptive1DBasic(self, vin, pin, hin, x0, endx, step0 = 0.05, maxStep = 0.1):
        """| solving the vph_equation, vph_update equations step by step
        | using numSolvers.adaptive_implicit solver with adaptive stepsize
        :param vin: inlet speed
        :param pin: inlet pressure
        :param hin: inlet spec enthalpy
        :param xo,endx: solve the equations from x0 until endx
        :param step0: initial stepsize
        :param maxStep: maximal stepsize
        :return:
        """
        v0 = [vin, pin, hin]
        # updaterFunction(lastY, xlast, dX, params)
        Din =  refProp.getTD(self.RP, hin, pin)['D']
        massflow = vin * self.AFun(x0) * 1e-4 * Din
        params = {'massflow': massflow}
        print(params)
        updater = lambda vph, xlast, dX : self.vph_update(vph, xlast, dX, params, 'scipy') # 'myNewton'
        stopCondition = lambda vph : (vph <= 0.0).sum() > 0 # any element negative
        sol = numSolvers.adaptive_implicit(x0, endx, step0, maxStep, updaterFunction = updater,
                                           initY = v0, stepCorrection = FlowSolver.stepCorrection,
                                           stopCondition = stopCondition)
        # print(sol)
        densfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['D']
        ## speed of sound from the HEM model:
        cfunc = lambda row: refProp.getSpeedSound(self.RP, hm=row[3], P=row[2])
        qfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['q']
        density = np.apply_along_axis(densfunc, 1, sol)
        speed_sound = np.apply_along_axis(cfunc, 1, sol)
        quality = np.apply_along_axis(qfunc, 1, sol)
        solution = pd.DataFrame({'x': sol[:, 0], 'v': sol[:, 1], 'p': sol[:, 2], 'h': sol[:, 3], 'd': density,
                            'c': speed_sound, 'mach': sol[:, 1] / speed_sound, 'quality': quality})
        return solution


    def solve1D(self,  vin, pin, hin, nint, endx, startx=0.0, odesolver= "RK4"):
        """ solve initial value problem specified by self.basic_dvdpdh

        :param vin:
        :param pin:
        :param hin:
        :param nint: number of integration points
        :param endx: last x value
        :param startx: starting x value
        :param odesolver: RK4 or BDF
        :return:
        """
        # fun = lambda vph, x: self.basic_dvdpdh(vph, x)
        if hasattr(self, 'capprox'):
            capprox = self.capprox
        else:
            capprox = True
        fun = lambda vph, x: self.simple_dvdpdh(vph, x, capprox)
        v0 = [vin, pin, hin]
        if odesolver == "RK4":
            xvals = np.linspace(startx, endx, nint)
            sol = numSolvers.RungeKutta4(fun, v0, xvals)
        elif odesolver == "BDF":
            funrev = lambda x,vph: self.simple_dvdpdh(vph, x, capprox)
            vBDF = scipy.integrate.solve_ivp(funrev, [startx, endx], v0, method='BDF')
            sol = np.hstack((vBDF.t.reshape((-1, 1)), vBDF.y.transpose()))
                                             # events=[negPress, negEnthalpy], atol=self.extraParams['atol'])
        densfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['D']
        ## speed of sound from the HEM model:
        cfunc = lambda row: refProp.getSpeedSound(self.RP, hm=row[3], P=row[2])
        qfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['q']
        density = np.apply_along_axis(densfunc, 1, sol)
        speed_sound = np.apply_along_axis(cfunc, 1, sol)
        quality = np.apply_along_axis(qfunc, 1, sol)
        res = pd.DataFrame({'x': sol[:, 0], 'v': sol[:, 1], 'p': sol[:, 2], 'h': sol[:, 3], 'd': density,
                            'c': speed_sound, 'mach': sol[:, 1] / speed_sound, 'quality': quality})
        return res

    def vph2all(self, xvph: np.array):
        ''' fill a dataframe, calculate density, c and quality numbers
        :param xvph: a 2d numpy array with columns : x-location, velocity, pressure, enthalpy
        :return: pandas dataframe
        '''
        densfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['D']
        cfunc = lambda row: refProp.getSpeedSound(self.RP, hm=row[3], P=row[2])
        qfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['q']
        density = np.apply_along_axis(densfunc, 1, xvph)
        speed_sound = np.apply_along_axis(cfunc, 1, xvph)
        quality = np.apply_along_axis(qfunc, 1, xvph)
        res = pd.DataFrame({'x': xvph[:, 0], 'v': xvph[:, 1], 'p': xvph[:, 2], 'h': xvph[:, 3], 'd': density,
                            'c': speed_sound, 'mach': xvph[:, 1] / speed_sound, 'quality': quality})
        return res

    def pFromV_MassConst(self, v, dv, p, h):
        """ Take a flow with a given speed(v), pressure(p) and TEmperature (T), search the the other pressure p+dp,
        for which the mass flow remains the same, so
         Density(p,h) * v = Density(p + dp, h ) * (v+dv)
        :param v: velocity [m/s]
        :param dv: velocity change
        :param p: pressure in kPa
        :param h: spec enthalpy in kJ/kg
        :return : dp pressure difference in kPa
        """
        diffMF = lambda dpr : refProp.getTD(self.RP, h, p + dpr)['D']*(v+dv) - refProp.getTD(self.RP, h, p)['D'] * v
        sol_dp = scipy.optimize.fsolve(diffMF, x0 = 0)[0]
        return sol_dp

