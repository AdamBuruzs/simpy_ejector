## original version by Adam Buruzs
## Solving 0 and 1 D equations for nozzle flow calculations
# flow media is 1 component material, where 2 phase is allowed
# liquid -vapor phase transition is modelled by Homogeneus Equilibrium Model
# written by Adam Buruzs
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy.optimize
import scipy.integrate
from simpy_ejector.flowSolver import FlowSolver
from simpy_ejector import NozzleParams, numSolvers, refProp
import logging

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 100)
np.core.arrayprint._line_width = 180

class NozzleSolver(FlowSolver):
    """ Solving 0 and 1 D equations for nozzle flow calculations

    flow media is 1 component material, where 2 phase is allowed

    liquid -vapor phase transition is modelled by Homogeneus Equilibrium Model
    """

    def __init__(self, nozzle : NozzleParams, fluid="BUTANE", nint: int = 300, solver="RK4", mode="basic"):
        """ An object with numerical algorithms to solve the 2 phase fluid flow equations.

        :param nozzle: a nozzle object, containing the geometrical data, instance of class NozzleParams
        :param fluid: a fluid string , see refprop
        :param nint: an integer parameter to control the grid/stepsize by numerical initial value problem integration
        this has no effect by solver = "AdamAdaptive" and "BDFvarstep"
        :param solver: the numerical solver for the initial value problem. RK4: Runge Kutta 4. order,RK2 : Runge Kutta 2.,
            Euler : Euler method, "BDFvarstep": scipy adaptive BDF method
            "AdamAdaptive": a simple variable stepsize (adaptive) method, implemented by Adam Buruzs
        :param mode: "basic", "advanced", or "viscoLinear"   \\
            the "basic" mode calculates without viscosity and heat convection". \\
            "viscoLinear" takes into account the viscosity, but just keeps the linear terms that are easier to solve.
             heat convection is neglected by "viscoLinear" mode \\
            "Advanced" solver with accounting for viscosity term and heat conduction term
        """
        super().__init__(fluid)
        # self.fluid = fluid
        # self.RP = refProp.setup(fluid)
        self.nint = nint
        self.nozzle = nozzle
        self.setAreaDeriv(nozzle.dAdxNum, nozzle.Aprofile)
        assert solver in ["RK2", "RK4", "Euler", "BDFvarstep", "AdamAdaptive"], "no such solver!"
        self.solver = solver
        self.mode = mode

    def setExtraParams(self, params: dict):
        self.extraParams = params

    def nozzle_dvdpdh(self, vph: np.array, x: float):
        '''DEPRECATED! Do not use this !
        |  the function specifying the differential equation of the initial value problem
        :param RP: REFPROPInstance
        :param vph: [velocity, pressure, spec enthalpy]
        :param x: spatial location
        :return: [dv/dx, dp/dx, dh/dx] 3 dim numpy array
        '''
        [v, p, h] = vph
        c = refProp.getSpeedSound(self.RP, h, p)
        D = refProp.getTD(self.RP, h, p)['D']
        eps = 0.0001
        nozzle = self.nozzle
        dAdx = nozzle.dAdxNum(x)
        dvdx = v / (math.pow(v, 2.0) / math.pow(c, 2.0) - 1.0) * dAdx / nozzle.Aprofile(x)
        if hasattr(nozzle, 'frictionCoef'):
            dvdx = dvdx - nozzle.frictionCoef * math.sqrt(math.pi / nozzle.Aprofile(x)) * math.pow(v, 3.0) / (
                    math.pow(v, 2.0) - math.pow(c, 2.0))
        dpdx = -1 * D * v * dvdx / 1.e3  # kPascal
        dhdx = -1 * v * dvdx / 1.e3  # kJ/kg
        out = np.array([dvdx, dpdx, dhdx])
        return out

    def nozzle_dvdpdhdw(self, vphw: np.array, x: float):
        ''' the function specifying the differential equation of the initial value problem with
        viscosity and heat conduction terms. See eq 6.2C
        :param RP: REFPROPInstance
        :param vphw: [velocity, pressure, spec enthalpy, w] w is an auxiliary variable which is defined to convert the
        second order differential equation to first order
        :param x: spatial location
        :return: [dv/dx, dp/dx, dh/dx, dw/dx] 3 dim numpy array
        '''
        [v, p, h, w] = vphw
        c = refProp.getSpeedSound(self.RP, h, p)
        rpqs = refProp.getTD(self.RP, h, p)
        D = rpqs['D']
        T = rpqs['T']
        eps = 0.0001
        nozzle = self.nozzle
        if (x + eps < self.nozzle.L):
            dAdx = (nozzle.Aprofile(x + eps) - nozzle.Aprofile(x)) / eps
        else:  # at the end:
            dAdx = -1 * (nozzle.Aprofile(x - eps) - nozzle.Aprofile(x)) / eps
        dvdx = v / (math.pow(v, 2.0) / math.pow(c, 2.0) - 1.0) * dAdx / nozzle.Aprofile(x)
        if hasattr(nozzle, 'frictionCoef'):
            dvdx = dvdx - nozzle.frictionCoef * math.sqrt(math.pi / nozzle.Aprofile(x)) * math.pow(v, 3.0) / (
                    math.pow(v, 2.0) - math.pow(c, 2.0))
        dpdx = -1 * D * v * dvdx / 1.e3  # kPascal
        mu, k = refProp.getTransport(self.RP, T, D)  # mu is microPa*sec
        dh = 0.001
        print("x : {}".format(x))
        dTdh = (refProp.getTD(self.RP, h + dh, p)['T'] - T) / dh
        print("dTdh : {}".format(dTdh))
        alpha = 3.0 / 4.0 * mu / D * 1.0e-6 * 100.0  # mu in mPa alpha/beta should be m/s
        beta = k * dTdh / D / v * 1.0e-3 * 100.0  # dh in kJ/kg. beta in cm
        # beta = 1 # just for test!!!
        # dhdx = -1 * v * dvdx / 1.e3  # kJ/kg
        #### dimensions : [h] = kJ/kg , [w] = kJ/kg
        ## beta is in meter, but dhdx we need in kJ/kg/cm ! that causes factor 1e-2
        dhdx = w / beta * 1.0e-2 + alpha / beta * dvdx * 1.0e-3
        # debuggung variables:
        term1, term2 = [w / beta * 1.0e-2, alpha / beta * dvdx * 1.0e-3]
        print("x : {}".format(x))
        print(" w/beta {} beta {}".format(w / beta * 1.0e-2, beta))
        print(" dhdx2 {} \n ".format(alpha / beta * dvdx * 1.0e-3))
        dwdx = dhdx + v * dvdx * 1.0e-3
        print("w {}, h {} , dhdx {} dwdx {}".format(w, h, dhdx, dwdx))
        out = np.array([dvdx, dpdx, dhdx, dwdx])
        return out

    def nozzle_dvdpdh_visco_linear(self, vphw: np.array, x: float, abstrick = True ):
        ''' the function specifying the linear differential equation of the initial value problem with
        viscosity without heat conduction terms. This only contains the linear terms, so that it can be solved
        with the linear solver of np.linalg.solve
        :param RP: REFPROPInstance
        :param vphw: [velocity, pressure, spec enthalpy, w] w = dv/dx
        :param x: spatial location
        :param abstrick: shall we use the trick with abs(v') to keep v' positive?
        :return: [dv/dx, dp/dx, dh/dx, dw/dx] 3 dim numpy array
        '''
        # physical quantities in used unit:
        [vu, pu, hu, wu] = vphw  # m/s ,kPa, kJ/kg, m/s/m = 1/sec
        # print("next step x {:.5E} start v,p,h, w values: ".format(x))
        # print(str(vphw))
        # print('pval {}'.format(vphw[1]))
        # convert the variables into SI units
        [v, p, h, w] = [vu, pu * 1000, hu * 1000, wu ]
        c = refProp.getSpeedSound(self.RP, hu, pu)
        rpqs = refProp.getTD(self.RP, hu, pu)
        D = rpqs['D']  # density (\rho)
        T = rpqs['T']
        mu, k = refProp.getTransport(self.RP, T, D)  # mu is microPa*sec
        muSI = mu * 1.0e-6  # mu in Pa*sec System International
        nuSI = muSI/ D # kinematic viscosity
        eps = 0.0001
        dDdh = (refProp.getTD(self.RP, hu + eps, pu)['D'] - D) / eps / 1000.  # h in kJ/kg
        dDdp = (refProp.getTD(self.RP, hu, pu + eps)['D'] - D) / eps / 1000.  # p in kPa
        nozzle = self.nozzle
        Ax = nozzle.Aprofile(x) * 1.0e-4  # in m^2
        dAdxu = nozzle.dAdxNum(x)
        dAdx = dAdxu / 100.0
        # mass conservation
        if abstrick:
            wa = math.fabs(w)
        else:
            wa = w
        leftside = [wa, - dAdx / Ax - wa / v ]
        lf3a = nozzle.frictionCoef * math.sqrt(math.pi / Ax) * math.pow(v, 2.0) * D
        lf3b = muSI / 3.0 * v / math.pow(Ax, 2.0) * math.pow(dAdx, 2.0)
        lf3c = wa * (D * v - muSI / 3 / Ax * dAdx)
        leftside.append(-lf3a - lf3b -lf3c)
        lf4b = wa * ( D * math.pow(v, 2.0) + 5.0 / 3.0 * dAdx * v * muSI / Ax )
        leftside.append(4.0 / 3.0 * math.pow(dAdx, 2.0) / math.pow(Ax, 2.0)
                        * muSI * math.pow(v, 2.0) - lf4b )
        left = np.array(leftside)
        # equation left = eqMat * [v',p',h',w']
        eqMat = np.zeros((4, 4))  # the matrix of the equation system
        eqMat[0, 0] = 1
        ### TODO ! shouldn't dDdp = 1/c^2 ?
        eqMat[1, :] = [0.0, 1. / D * dDdp, 1. / D * dDdh, 0.0]
        eqMat[2, :] = [0.0, 1.0, 0.0, - muSI * 4.0 / 3.0]
        eqMat[3, :] = [0.0, 0.0, D * v, -4.0 / 3.0 * muSI * v]
        # and solve the equation for the derivatives:
        try:
            res = np.linalg.solve(eqMat, left)
            # print(eqMat)
            # print('leftvec : '+ str(leftside))
            # print('x : {:.4E} '.format(round(x, 10) ) +"res v',p',h',w' " + str(res) )
        except np.linalg.LinAlgError as err:
            print("linalg error occured at point x ={}, equtaion Matrix :".format(x))
            print(eqMat)
            print("v {} p {} h {} w {}".format(v, p, h, w))
            print("dDdp {} dDdh{}".format(dDdp, dDdh))
            print(err)
            res = np.zeros(4)
            return res
        dvdx, dpdx, dhdx, dwdx = res
        # convert back to [ m/s  ,kPa, kJ/kg, 1/s ] / cm !
        out = np.array([dvdx / 100, dpdx / 1.0e5, dhdx / 1.0e5, dwdx / 100])
        # print("out " + str(out))
        return out

    def visco_lin_Mat(self, vphw: np.array, x: float, abstrick = True ):
        """ This function is for debugging !
        it just calculates the matrix of the equation to be solved in the linear viscous model"""
        [vu, pu, hu, wu] = vphw  # m/s ,kPa, kJ/kg, m/s/m = 1/sec
        # convert the variables into SI units
        [v, p, h, w] = [vu, pu * 1000, hu * 1000, wu]
        c = refProp.getSpeedSound(self.RP, hu, pu)
        rpqs = refProp.getTD(self.RP, hu, pu)
        D = rpqs['D']  # density (\rho)
        T = rpqs['T']
        mu, k = refProp.getTransport(self.RP, T, D)  # mu is microPa*sec
        muSI = mu * 1.0e-6  # mu in Pa*sec System International
        nuSI = muSI / D  # kinematic viscosity
        eps = 0.0001
        dDdh = (refProp.getTD(self.RP, hu + eps, pu)['D'] - D) / eps / 1000.  # h in kJ/kg
        dDdp = (refProp.getTD(self.RP, hu, pu + eps)['D'] - D) / eps / 1000.  # p in kPa
        nozzle = self.nozzle
        Ax = nozzle.Aprofile(x) * 1.0e-4  # in m^2
        dAdxu = nozzle.dAdxNum(x)
        dAdx = dAdxu / 100.0
        # mass conservation
        if abstrick:
            wa = math.fabs(w)
        else:
            wa = w
        leftside = [wa, - dAdx / Ax - wa / v]
        lf3a = nozzle.frictionCoef * math.sqrt(math.pi / Ax) * math.pow(v, 2.0) * D
        lf3b = muSI / 3.0 * v / math.pow(Ax, 2.0) * math.pow(dAdx, 2.0)
        lf3c = wa * (D * v - muSI / 3 / Ax * dAdx)
        leftside.append(-lf3a - lf3b - lf3c)
        lf4b = wa * (D * math.pow(v, 2.0) + 5.0 / 3.0 * dAdx * v * muSI / Ax)
        leftside.append(4.0 / 3.0 * math.pow(dAdx, 2.0) / math.pow(Ax, 2.0)
                        * muSI * math.pow(v, 2.0) - lf4b)
        left = np.array(leftside)
        # equation left = eqMat * [v',p',h',w']
        eqMat = np.zeros((4, 4))  # the matrix of the equation system
        eqMat[0, 0] = 1
        eqMat[1, :] = [0.0, 1. / D * dDdp, 1. / D * dDdh, 0.0]
        eqMat[2, :] = [0.0, 1.0, 0.0, - muSI * 4.0 / 3.0]
        eqMat[3, :] = [0.0, 0.0, D * v, -4.0 / 3.0 * muSI * v]
        return (eqMat, left)

    def calcW0Init(self, vph):
        """Calculate the innitial w input value for the advanced solver
        :return: - beta * vin * dv/dx| in
        """
        [v, p, h] = vph
        c = refProp.getSpeedSound(self.RP, h, p)
        rpqs = refProp.getTD(self.RP, h, p)
        D = rpqs['D']
        T = rpqs['T']
        x0 = 0.0
        eps = 0.0001
        nozzle = self.nozzle
        dAdx = (nozzle.Aprofile(x0 + eps) - nozzle.Aprofile(x0)) / eps
        dvdx = v / (math.pow(v, 2.0) / math.pow(c, 2.0) - 1.0) * dAdx / nozzle.Aprofile(x0)
        mu, k = refProp.getTransport(self.RP, T, D)  # mu is microPa*sec
        dh = 0.001
        dTdh = (refProp.getTD(self.RP, h + dh, p)['T'] - T) / dh
        dhdT = 1.0 / dTdh
        alpha = 3.0 / 4.0 * mu / D * 1.0e-6  # mu in mPa
        beta = k / dhdT / D / v * 1.0e-3  # dh in kJ/kg
        w0 = - beta * v * dvdx
        return w0

    def startW(self, vin, startx):
        """ return - v dA/dx """
        eps = 0.0001
        nozzle = self.nozzle
        if (startx > (nozzle.L - eps)):
            eps = - 0.0001
        dAdx = (nozzle.Aprofile(startx + eps) - nozzle.Aprofile(startx)) / eps
        dvdx0 = - vin / nozzle.Aprofile(startx) * dAdx
        dvdx0m = dvdx0 * 100
        return dvdx0m # cm -> m conversion


    def solve1dNozzle(self, vin, pin, hin, nint, startx=0.0, gridType='invA'):
        ''' solve the discretized initial value problem
        :param vin,pin, hin: inital values of velocity, pressure, enthalpy
        :param RP: REFPROP pointer use RP = refProp.setup(fluid)
        :param nint: number of intervals for the discretization
        :param gridType: invA or linear
        :return: the solution: a numpy 2 dim array with [x,v,p,h] values for each x point
        '''
        if (self.mode == "basic") & (self.solver == "AdamAdaptive"):
            if not hasattr(self, "extraParams"):
                step0, maxStep = 0.05, 0.1
            else :
                if 'step0' in self.extraParams.keys():
                    step0 = self.extraParams["step0"]
                else:
                    step0 = 0.05
                if 'maxStep' in self.extraParams.keys():
                    maxStep = self.extraParams["maxStep"]
                else:
                    maxStep = 0.1
            sol = self.solveAdaptive1DBasic(vin, pin, hin, startx, self.nozzle.L, step0, maxStep)
        elif  self.mode == "basic":
            v0 = [vin, pin, hin]
            fun = lambda vph, x: self.simple_dvdpdh(vph, x, capprox= False)
        elif self.mode == "advanced":
            w0 = self.calcW0Init([vin, pin, hin])
            v0 = [vin, pin, hin, w0]
            fun = lambda vph, x: self.nozzle_dvdpdhdw(vph, x)
        elif self.mode == "viscoLinear":
            w0 = self.startW(vin, startx)
            v0 = [vin, pin, hin, 0.0 ]
            fun = lambda vph, x: self.nozzle_dvdpdh_visco_linear(vph, x)
        else:
            print("ERROR : {} mode is not implemented yet".format(self.mode))
        funrev = lambda x, vph : fun(vph, x)
        if (gridType == "linear"):
            xvals = np.linspace(startx, nozzle.L, nint)
        elif (gridType == "invA"):
            xvals = self.nozzle.gridLinvA(startx, nint)
        else:
            print("ERROR! no such grid-Type! {}".format(gridType))
        if (startx < self.nozzle.xt) & ((xvals == self.nozzle.xt).sum() == 0):
            xvals = np.sort(np.append(xvals, self.nozzle.xt))
        if (self.solver == "RK2"):
            sol = numSolvers.RungeKutta2(fun, v0, xvals)
        elif (self.solver == "RK4"):
            sol = numSolvers.RungeKutta4(fun, v0, xvals)
        elif (self.solver == "Euler"):
            sol = numSolvers.IVEuler(fun, v0, xvals)
        elif ((self.solver == "BDFvarstep") & (self.mode == "basic")):
            vBDF = scipy.integrate.solve_ivp(funrev, [xvals.min(), self.nozzle.L], v0, method='BDF')
            sol = np.hstack((vBDF.t.reshape((-1, 1)), vBDF.y.transpose()))
        elif ((self.solver == "BDFvarstep") & (self.mode != "basic")):
            # first solve it until the throat
            print('running BDF calculation')
            rpqs = refProp.getTD(self.RP, hin, pin)
            D = rpqs['D']  # density (\rho)
            T = rpqs['T']
            mu, k = refProp.getTransport(self.RP, T, D)  # mu is microPa*sec
            muSI = mu * 1.0e-6  # mu in Pa*sec System International
            nuSI = muSI / D  # kinematic viscosity
            print('nuSI {}'.format(nuSI) )
            # first_step = nuSI / 10.
            v0 = [vin, pin, hin, self.extraParams['w0'] ]
            # TODO : handle negative pressure and singular matrix
            negPress = lambda x,vph : vph[1] # pressure crossing 0
            negPress.terminal = True
            negPress.direction = -1
            negEnthalpy = lambda x,vph : vph[2] # enthalpy gets negative
            negEnthalpy.terminal = True
            negEnthalpy.direction = 1

            if 'atol' in self.extraParams.keys():
                vBDF = scipy.integrate.solve_ivp(funrev, [xvals.min(), self.nozzle.xt], v0, method='BDF',
                                                 events=[negPress, negEnthalpy], atol = self.extraParams['atol'])
            else:
                # atol_used = 1.e-6 # the default atol
                vBDF = scipy.integrate.solve_ivp(funrev, [xvals.min(), self.nozzle.xt], v0, method='BDF',
                                                 events=[negPress,negEnthalpy])

            # , events= [negPress]

            sol = np.hstack((vBDF.t.reshape((-1, 1)), vBDF.y.transpose()))
            if( sol[-1,0] < self.nozzle.xt):
                print('BDF solver could not reach the end. probably the phase transition starts here. ')
                print(' x = {} num points {}'.format(sol[-1,0], sol.__len__()))
                print( vBDF.message )
                vphlast = sol[-1, 1:]
                # let's kick the ass of the solution, otherwise it will be so stiff that we can't solve it, so:
                # we manually decrease the pressure with 1 kPa ! :
                vphKicked = np.append( [ vphlast[0 ], vphlast[1] - 1 ] ,  vphlast[2:] )
                # and with this kicked result as initial value restart the solver:
                vBDF2 = scipy.integrate.solve_ivp(funrev, [sol[-1,0], self.nozzle.xt], vphKicked, method='BDF',
                                                  atol=1e-4, rtol=1e-2, events=[negPress])
                sol2 =  np.hstack((vBDF2.t.reshape((-1, 1)), vBDF2.y.transpose()))
                print(vBDF2.message)
                sol = np.append(sol,  sol2, axis = 0 ) # merge the two solutions together

            print(sol)
            print(vBDF)
        # sol = numSolvers.initialProblem(fun, v0, xvals)
        return sol

    # def negPressure( x, vph): # event to shot down the BDF ODE solver
    #     pressure = vph[1]
    #     return pressure
    #
    # negPressure.terminal = True

    def solveKickedNozzle(self, vin, pin, hin, kicks = {"v" : 1 , "p" : 1 }, solver= "adaptive_implicit", **kwargs):
        """Solve the flow equations for a supersonic Nozzle,
        with kicking the pressure and flow speed at the throat, so to help the supersonic transition in the divergent part
        :param kicks: a dict with the kicks (the flow velocity and pressure is artificially changed in the throat)
        """
        sol_1 = self.solveAdaptive1DBasic(vin, pin, hin, 0.0, self.nozzle.xt)
        pkick = kicks["p"]
        vkick = kicks["v"]
        vph_throat = sol_1.iloc[-1]
        logging.debug(f"last values by throat {sol_1.iloc[-1]} \n solving for the rest, with args : {kwargs}")
        if solver == "adaptive_implicit":
            sol_2 = self.solveAdaptive1DBasic(vph_throat['v'] + vkick, vph_throat['p'] - pkick, vph_throat['h'],
                                                     self.nozzle.xt, self.nozzle.L, **kwargs)
        elif solver == "BDF":
            sol_2 = self.solve1D(vph_throat['v'] + vkick, vph_throat['p'] - pkick, vph_throat['h'],
                                 0, endx=self.nozzle.L, startx=self.nozzle.xt,
                           odesolver="BDF")
        sol_full = sol_1.append(sol_2.iloc[1:], ignore_index=True)
        return sol_full

    def sonicPoint(self, vin, pin, hin):
        ''' DEPRECATED:
        determine at which x point does the flow reaches the sonic speed

        :param vin: inlet speed
        :return: the point xs where the flow gets supersonic, or a pseudo point, if the flow is everywhere subsonic
        '''
        propx = self.solveNplot(vin, pin, hin, doPlot=False)
        maxmach = max(propx['mach'])
        pointBeforeThroat = propx['x'][propx['x'] < self.nozzle.xt].max()
        ## search for dropping speed in convergent part:
        vchange = propx[propx['x'] < self.nozzle.xt]['v'].diff()
        firstdrop = np.argmax( (vchange < 0.0 ))
        if (firstdrop > 0 ): # there is such a speed drop in the convergent nozzle
            fd = firstdrop -1 # position of first local maximum
            # calculate Mach=1 position from extrapolation the mach number
            xs = propx['x'][fd] + (1 - propx['mach'][fd]) * \
                 (propx['x'][fd] - propx['x'][fd - 1]) / (propx['mach'][fd] - propx['mach'][fd - 1])
            return xs
        if (maxmach == 1.0):
            ## perfect solution
            xs = propx.iloc[propx['mach'].idxmax()]['x']
        elif (maxmach < 1.0):
            ## flow stays everywhere subsonic
            # maximum is by the throat:
            if ( propx['x'][propx['mach'].idxmax()] >= pointBeforeThroat ):
                T = propx.iloc[(propx['x'] >= self.nozzle.xt).idxmax()]
                X = propx.iloc[(propx['x'] >= self.nozzle.xt - self.nozzle.L / 10.0).idxmax()]
                # linear interpolation
                xs = T['x'] + (1.0 - T['mach']) * (T['x'] - X['x']) / (T['mach'] - X['mach'])
            else :
                # maximum Mach number inside the divergent part. The sonic speed reached there, just
                # stays subsonic because of numeric reasons
                im = propx['mach'].idxmax() # index of maximal Mach position
                # calculate mach=1 postion from extrapolating from im and im-1 points
                xs = propx['x'][im] + (1 - propx['mach'][im]) * \
                     (propx['x'][im] -propx['x'][im-1])/ (propx['mach'][im] -propx['mach'][im-1])
        elif (maxmach > 1.0):
            ## flow has supersonic region:
            ss = propx.iloc[(propx['mach'] > 1.0).idxmax()]  # first supersonic point
            ssm1 = propx.iloc[(propx['mach'] > 1.0).idxmax() - 1]  # one point before
            # linear interpolation
            xs = ss['x'] - (ss['x'] - ssm1['x']) * (ss['mach'] - 1.0) / (ss['mach'] - ssm1['mach'])
        return xs

    def calcCriticalSpeed_old(self, pin, hin, v0=1.0, maxdev=0.1):
        ''' Iteratively calculate at which inlet speed the flow gets supersonic
        :return: the inlet velocity, where the flow gets supersonic
        '''
        t0 = time.time()
        sonicpoint = lambda vin: self.sonicPoint(vin, pin, hin)
        throatSonic = lambda vin: sonicpoint(vin) - self.nozzle.xt
        # use a big eps, because the RG4 solution is apparently not monotonic!
        vin_crit = numSolvers.NewtonRaphson(throatSonic, v0, maxdev, 20, sign=-1, eps = 0.01)
        print('critical speed calculation finished in {} sec'.format(round(time.time() - t0, 3)))
        print('critical inlet velocity is {} m/s'.format(vin_crit))
        return vin_crit

    def calcCriticalSpeed(self, pin, hin, v0=0.1, maxdev=0.01, chokePos = "throat" ):
        ''' Iteratively calculate at which inlet speed the flow gets supersonic
        :param v0: a low but positive value to start the search from. This should be a subcritical velocity.
        :param maxdev: maximum deviation from the exact solution (stopping tolerance)
        :param chokePos: position of choking one of "throat" or "divergent_part". if throat, the critical solution
        is choking by the throat, if divergent_part, then the choking is allowed in the divergent part of the nozzle also
        :return: the inlet velocity, where the flow gets supersonic
        '''
        assert chokePos in ["throat", "divergent_part"], "chokePos parameter is one of 'throat' or 'divergent_part' "
        t0 = time.time()
        # transformer = lambda vin: self.solve1dNozzle(vin, pin, hin, 0, 0.0)
        if chokePos == "throat":
            transformer = lambda vin: self.solveAdaptive1DBasic( vin, pin, hin, 0.0, self.nozzle.xt, 0.05, 0.1)
            noChoking = lambda solres: solres['x'].iloc[-1] == self.nozzle.xt  # integration succeeded till the nozzle throat
        elif chokePos == "divergent_part":
            transformer = lambda vin: self.solveAdaptive1DBasic( vin, pin, hin, 0.0, self.nozzle.L, 0.05, 0.1)
            noChoking = lambda solres: solres['x'].iloc[-1] == self.nozzle.L  # integration succeeded till the nozzle end
        vin0 = 0.3
        # sol1 = solver.solveAdaptive1D(vin0, pin, hin, 0.0, nozzle.xt)
        vin_crit = numSolvers.findMaxTrue(vin0, transformer, noChoking, tol=maxdev)
        print('critical speed calculation finished in {} sec'.format(round(time.time() - t0, 3)))
        print('critical inlet velocity is {} m/s'.format(vin_crit))
        return vin_crit

    def calcCriticalPressure_out(self, vin_crit, pin, hin, Nint=1000, vstep=0.01):
        ''' Deprecated!  calculate the Pressure by the Nozzle outlet, when the flow gets supersonic by the throat.
         Until this outlet pressure the flow still stays subsonic.
         THIS IS WRONG, DON'T USE IT!

         :param vstep: velocity stepsize in m/s for accuracy
         :param vin_crit: critical inlet speed, where the flow is already supersonic
         :returns: vin where the flow is still fully subsonic, but vin+vstep is already partly supersonic.
          + return a pandas dataframe with pressure, mach, speed, density etc by the divergent nozzle exit in the subsonic case
          + the low exit pressure in the full-supersonic divergent nozzle case (without shock wave in the divergent nozzle)
          + a more precise critical inlet speed, with max error vstep
         '''
        vin = vin_crit + 2 * vstep
        subsonic = False
        ## TODO ! This is not enough yet,
        while not subsonic:
            res = self.solveNplot(vin, pin, hin, doPlot=False)
            if (res['mach'].max() < 1.0):  # TODO:  this condition is not enough!
                subsonic = True
            else:  # supersonic case
                vin_crit_precise = vin
                vin = vin - vstep
                pexit_supersonic = res.iloc[-1]['p']
        print('precise critical inlet speed {}'.format(vin_crit_precise))
        return (vin, res.iloc[-1], pexit_supersonic, vin_crit_precise)

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

    @staticmethod
    def shock_equation(RP, vph: np.array):
        ''' The equation to be solved for the shock wave calculation
                :param vph: a vector with (velocity,pressure,enthalpy)
                :return: the right side of the equation, mass flux, pressure term, energy term'''
        [v, p, h] = vph
        c = refProp.getSpeedSound(RP, h, p)
        D = refProp.getTD(RP, h, p)['D']
        j = v * D
        pterm = p + D * math.pow(v, 2.0) * 1.0e-3  # kPa
        hterm = h + 0.5 * math.pow(v, 2.0) * 1.0e-3  # kJ/kg
        return np.array([j, pterm, hterm])

    def shock_jph(self, vph: np.array):
        ''' The equation to be solved for the shock wave calculation
        :param vph: a vector with (velocity,pressure,enthalpy)
        :return: the right side of the equation, mass flux, pressure term, energy term'''
        return NozzleSolver.shock_equation(self.RP, vph)

    def calcShockWaveFront(self, x_sh, superSolution):
        ''' Starting from the chocked total supersonic solution
         assume a static normal shock wave at x_sh in the divergent nozzle
        this function calculates the (v,p,h) vector on the downstream side of the shock-wave
        :param superSolution: pandas Dataframe with 'x' column
        :return: velocity,pressure,enthalpy,density by the downstream side of the shock-wave.
        '''
        physquants = NozzleSolver.interpolate(superSolution, x_sh, 'x')
        print('shockwave in {}'.format(x_sh))
        print('shockwave upstream v:{}, p:{} h:{} D:{}'.format(*physquants[['v', 'p', 'h', 'd']].values))
        vph_upstream = physquants[['v', 'p', 'h']]
        # we have to solve 0 = fun2solve(vph) system of equations
        leftside = self.shock_jph(vph_upstream)
        print('upstream left side: ' + str(leftside))
        fun2solve = lambda vph: self.shock_jph(vph) - leftside
        # vph2 = numSolvers.NewtonND(fun2solve, x0 = np.array([0.0, vph_upstream[1] * 2.0 , vph_upstream[2] ]),
        #                            maxdev= 0.1, debug = True)
        vphsc = scipy.optimize.root(fun2solve, x0=np.array([0.0, vph_upstream[1] * 2.0, vph_upstream[2]]),
                                    method='hybr')
        v2, p2, h2 = vphsc.x
        D2 = refProp.getTD(self.RP, h2, p2)['D']
        print('shockwave downstream v:{}, p:{} h:{} D:{}'.format(v2, p2, h2, D2))
        return [v2, p2, h2, D2]

    def calcBehindShockWave(self, x_sh, superSolution, nozzle, nint=500):
        assert (x_sh <= nozzle.L)
        assert (x_sh >= nozzle.xt)
        [v2, p2, h2, D2] = self.calcShockWaveFront(x_sh, superSolution)
        nrest = math.ceil(nint / nozzle.L * (nozzle.L - x_sh))
        if nrest >= 2:
            sh_downstream = self.solve1dNozzle(v2, p2, h2, nint=nrest, startx=x_sh)
        else:
            print("shock wave at the very end of the nozzle")
            sh_downstream = np.array([x_sh, v2, p2, h2]).reshape((1, -1))
        if isinstance(sh_downstream, pd.DataFrame):
            flowx = sh_downstream
        else:
            flowx = self.vph2all(sh_downstream)
        return (flowx)

    def calcWithShockinDiv(self, vin_crit, pin, hin, nint=500,  dxsh=None , res_crit: pd.DataFrame = None ):
        """ calculate flow with shock wave at position dxsh far from the throat
        :param vin_crit: the critical speed where the flow is supersonic in the whole divergent nozzle
        :param pin:
        :param hin:
        :param nozzle:
        :param fluid:
        :param nint:
        :param dxsh: distance from throat
        :param res_crit: critical solution
        :return: the flow params through the whole nozzle
        """
        if res_crit is None:
            res_crit = self.solveNplot(vin_crit, pin, hin, doPlot=False)  # this is the full supersonic solution
        if dxsh is None:
            return res_crit
        else:
            assert (dxsh > 0.0), "shock location should be positive"
            assert (dxsh <= (self.nozzle.L - self.nozzle.xt)), "shock should be before the exit ".format(dxsh)
            xsh = self.nozzle.xt + dxsh  # shock x coordinate
            afterShock = self.calcBehindShockWave(xsh, res_crit, self.nozzle, nint=nint)
            out = res_crit[res_crit['x'] < xsh]
            out = out.append(afterShock)
            return out


    def printNozzlePars(self, vvr, pin, Din, hin):
        nozzle = self.nozzle
        print(nozzle)
        [vin, vt, Dt] = vvr
        print("---- NOZZLE inlet : ")
        print("pressure {0}kPa , density {1} g/l , enthalpy {2}".format(pin, Din, hin))
        print("velocity {0} m/s, cross section {1} cm^2".format(vin, nozzle.Ain))
        Qin = Din * vin * nozzle.Ain / 1.e1  # in g/sec
        print("incoming mass flow {0} g/sec".format(Qin))
        c_in = refProp.getSpeedSound(RP, hin, pin)  #
        print("speed of sound {0}".format(c_in))

        print("--- Nozzle throat:")
        pt = pin + (Din * nozzle.Ain * math.pow(vin, 2.0) - Dt * nozzle.At * math.pow(vt, 2.0)) * 1.0e-7
        ht = hin + 0.5 * (math.pow(vin, 2.0) - math.pow(vt, 2.0)) / 1.e3
        print("speed {0} m/s, density {1}, pressure {2}".format(vt, Dt, pt))
        print("spec enthalpy {0} [J/g]".format(ht))
        Qt = Dt * vt * nozzle.At / 1.e1
        print("nozzle throat flow {0} g/sec".format(Qt))
        props = refProp.getTD(self.RP, ht, pt)
        print(props)
        print("quality (vapor mass ratio) = {0}".format(props['q']))

    def solveNplot(self, vin, pin, hin, doPlot=True, title = None):
        ''' assuming an vin inlet speed, solve the initial value problem, and plot all x-dependent quantities
        :param doPlot: show the plot of the quantites, or just return the values
        :return: a pandas dataframe with the x dependent data
        '''
        # TODO : sometimes around the critical inlet speed the solution does not go into
        #  supersonic in the divergent part. To solve this we could introduce a "speed kick" by the throat,
        #  that should be only used for around critical solutions to ensure that the flow gets supersonic after the throat.
        # print("nint =" , self.nint)
        sol = self.solve1dNozzle(vin, pin, hin, self.nint, gridType='invA')
        if(isinstance(sol, pd.DataFrame)):
            if(doPlot ):
                self.plotsol(sol, title)
            return sol
        densfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['D']
        cfunc = lambda row: refProp.getSpeedSound(self.RP, hm=row[3], P=row[2])
        qfunc = lambda row: refProp.getTD(self.RP, hm=row[3], P=row[2])['q']
        density = np.apply_along_axis(densfunc, 1, sol)
        speed_sound = np.apply_along_axis(cfunc, 1, sol)
        quality = np.apply_along_axis(qfunc, 1, sol)

        if doPlot:
            fig = plt.figure(figsize=[11,8])
            gs = fig.add_gridspec(3, 1)
            fig.add_subplot(gs[0, :])
            x = np.linspace(0, self.nozzle.L, 100)
            plt.ylabel("nozzle Radius [cm]")
            plt.plot(x, [self.nozzle.Rprofile(xi) for xi in x])
            plt.ylim(bottom=0.0)
            # plt.subplot(212)
            fig.add_subplot(gs[1:, :])
            plt.plot(sol[:, 0], sol[:, 1])
            plt.plot(sol[:, 0], sol[:, 2])
            plt.plot(sol[:, 0], sol[:, 3])
            plt.plot(sol[:, 0], density)
            plt.plot(sol[:, 0], speed_sound)
            plt.plot(sol[:, 0], quality * 100.0)
            plt.yscale("log")
            plt.legend(['v [m/s]', 'p [kPa]', 'h [J/g]', 'density [kg/m3]', 'c_sound [m/s]', 'vapor ratio(q) [%]'],
                       loc='lower left')
            plt.xlabel('x [cm]')
            plt.axvline(x=self.nozzle.xt, linestyle=':', color='b')
            fig.tight_layout()
            plt.title(title)
        res = pd.DataFrame({'x': sol[:, 0], 'v': sol[:, 1], 'p': sol[:, 2], 'h': sol[:, 3], 'd': density,
                            'c': speed_sound, 'mach': sol[:, 1] / speed_sound, 'quality': quality})
        return res


    def plotsol(self, sol1, title = "", plotCHEM = True):
        """ plotting results dataframe

        :param sol1: the pandas dataframe with the solution table
        :param title: Title of the image
        :param plotCHEM: shall the HEM speed of sound formula be plotted ?
        :return:
        """
        nozzle = self.nozzle
        massflow = sol1['v'] * sol1['d'] * [nozzle.Aprofile(xi) * 1e-4 for xi in sol1['x']]
        plt.figure(figsize = [11,8])
        plt.plot(sol1['x'], sol1['v'])
        plt.plot(sol1['x'], sol1['p'])
        plt.plot(sol1['x'], sol1['quality'])
        plt.plot(sol1['x'], massflow)
        plt.plot(sol1['x'], sol1['d'])
        if plotCHEM:
            plt.plot(sol1['x'], sol1['c'])
        plt.legend(['velocity [m/s]', 'pressure [kPa]', 'quality [0-1]', 'massflow [kg/s]',
                    'density [kg/m^3]', 'cHEM [m/s]'])
        plt.axvline(x=nozzle.xt, linestyle=':', color='b')
        plt.yscale("log")
        plt.xlabel("x [cm]")
        plt.title(title)


if __name__ == '__main__':
    from flows1d.core import EjectorGeom, numSolvers, nozzleFactory, refProp

    nozzle = nozzleFactory.ConicConic(1.0, 2.905, 0.2215, 1.4116, 0.345)
    nozzle.setFriction(1.0e-2)
    ejector = EjectorGeom(nozzle, Dm=1.4)
    print(nozzle)
    # [vin, vt , Dt ] = [10.0, 250., 10.0]
    ## Butane ejector for heat pump see Schlemminger article
    pin = 2100.0  # kPa
    Tin = 387.0  # Kelvin
    fluid = "BUTANE"
    RP = refProp.setup(fluid)

    vin = 3.40
    hin = 506.9
    nint = 100
    v0 = [vin, pin, hin]
    # RP = refProp.setup(fluid)
    solver = NozzleSolver(nozzle, "BUTANE", nint, solver="RK4")
    solRG4 = solver.solveNplot(vin, pin, hin, True)
    print(solRG4[-3:])
    solver2 = NozzleSolver(nozzle, "BUTANE", nint, solver="RK2")
    solRG2 = solver2.solveNplot(vin, pin, hin, True)
    print(solRG2[-3:])

    fun = lambda vph, x: solver.nozzle_dvdpdh(vph, x)
    fty = lambda x, vph: solver.nozzle_dvdpdh(vph, x)
    startx = 0.0
    xvals = np.linspace(startx, nozzle.L, nint)
    if (startx < nozzle.xt) & ((xvals == nozzle.xt).sum() == 0):
        xvals = np.sort(np.append(xvals, nozzle.xt))
    # TODO higher point density by the throat! density \prop 1/R or 1/A
    # TODO: or use an other solver!
    t0 = time.time()

    sol1 = numSolvers.RungeKutta2(fun, v0, xvals)
    t1 = time.time()
    print('solver 1 calculation finished in {} sec'.format(round(t1 - t0, 3)))
    print(sol1)
    # import scipy.integrate
    # t1b = time.time()
    # sol2 = scipy.integrate.solve_ivp(fty, [xvals.min(), xvals.max() ], v0,
    #                                  method= 'LSODA' ) #rtol = 1e-3)
    # print('solver scipy 1 calculation finished in {} sec'.format(round(time.time() - t1b, 3)))
    # tpoints = np.reshape(sol2.t, newshape = (sol2.t.size, 1))
    # solres = np.hstack( ( tpoints,sol2.y.transpose() ))
    # plt.figure()
    # plt.plot(sol1[:,0], sol1[:,2])
    # plt.plot(solres[:, 0], solres[:, 2])
    # print(sol2.y.transpose())
