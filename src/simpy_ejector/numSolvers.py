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

## function Tools for iterative equation solving
# Newton methon for solving nonlinear equations
# original written by Adam Buruzs
import math
import numpy as np
import sys
import time
import logging


def diff(function, x, eps=0.01):
    ''' numerical approximation of the differential quotient:
    :param function: input function to be differentiated
    :param x: the point where the function to be calculated
    :param eps: small epsilon value
    '''
    df = function(x + eps) - function(x - eps)
    dx = 2 * eps
    return (df / dx)


def NewtonRaphson(fun, x0: float, maxdev=0.01, maxit=100, sign=0, eps = None):

    ''' Newton Raphson iterative solution of an equation system
    solves fun(x) = 0.0 equation
    iteratively searches for ONLY 1 solution!!
    stops the iterations, when abs(fun(x)) < maxdev, that is,
    when we are close enough to the correct solution

    :param fun: input function
    :param x0: starting point of the iteration (initial guess)
    :param sign: if 0 no effect. If sign = 1, then only stops if deviation positive \\
     if sign = -1 only stops if deviation negative
    :param eps: the epsilon used to calculate the numeric derviative. If eps = None (not given), then
     it will use  maxdev / 10.0
    :return: one root of fun
    '''
    dev = fun(x0)
    nit = 0
    x = float(x0)
    if eps is None:
        eps = maxdev / 10.0
    while ((abs(dev) > maxdev) | (dev * sign < 0)) & (nit < maxit):
        dfdx = diff(fun, x, eps )
        newx = x - fun(x) / dfdx
        nit += 1
        dev = fun(newx)
        x = newx
        print('iteration {0} dev : {1} x val : {2}'.format(nit, round(dev, 6), round(x, 5)))
    if (nit == maxit): ## TODO! do something if the iteration get stucked (random push-out)
        print('Newton Raphson has not converged in {} iterations. error = {}'.format(nit, dev))
        if abs(dev) < (maxdev * 100) :
            return x
        else :
            return (None)
    else:
        print('Newton Raphson has converged in {} iterations'.format(nit))
        return x


def fixedpoint(fun, x0: float, maxdev=0.01, maxit=100):
    ''' fixed point iteration to solve x = fun(x) fixed point equation'''
    dev = x0 - fun(x0)
    nit = 0
    x = float(x0)
    while (abs(dev) > maxdev) & (nit < maxit):
        x = fun(x)
        dev = x - fun(x)
    if (nit == maxit):
        print('Fixedpoint has not converged in {0} iterations' +
              ' error = {1}'.format(nit, dev))
        return (None)
    else:
        print('Fixedpoint has converged in {0} iterations'.format(nit))
        return x


def fixedpointND(fun, x0: float, maxdev=0.01, maxit=100):
    ''' N dimensional fixed point iteration to solve x = fun(x) fixed point equation
    :param x0: numpy array
    :param fun: function N dim numpy array ->  N dim numpy array
    '''
    dev = np.linalg.norm(x0 - fun(x0))
    nit = 0
    x = np.float64(x0)
    while (abs(dev) > maxdev) & (nit < maxit):
        x = fun(x)
        dev = np.linalg.norm(x - fun(x))
        nit += 1
    if (nit == maxit):
        print('Fixedpoint has not converged in {0} iterations' +
              ' error = {1}'.format(nit, dev))
        return (None)
    else:
        print('Fixedpoint has converged in {0} iterations'.format(nit))
        return x


def Jacobian(fun, x: np.array, eps=0.001):
    ''' calculate the Jacobian matrix
    :param fun: function (n-dim array) -> n-dim array
    :param x: n-dim numpy array
    '''
    N = x.__len__()
    J = np.zeros((N, N))
    for j in range(N):
        dj = np.eye(N)[j]
        J[:, j] = (fun(x) - fun(x - dj * eps)) / eps
    return J


def NewtonND(fun, x0: np.array, alpha=1.0, maxdev=0.01, maxit=100, debug=False):
    ''' Solve fun(x) = 0 equation using Newton Raphson iterations

    :param fun: function whose root is searched
    :param x0: initial value of x, iteratio starts from here
    :param maxdev: maximum deviation
    :param maxit: maximum
    :return: the root, or None, if the iterations don't converge
    '''
    dev = np.linalg.norm(fun(x0))
    nit = 0
    x = np.float64(x0)
    xdim = x0.__len__()
    Xvals = x  ## for debugging collect values for each iteration
    while (abs(dev) > maxdev) & (nit < maxit):
        xold = x
        J = Jacobian(fun, x, eps=0.001)
        fx = fun(x)
        delta = np.linalg.solve(J, -1 * fx)
        newx = x + delta * alpha
        nit += 1
        dev = np.linalg.norm(fun(newx))
        x = newx
        Xvals = np.vstack((Xvals, x))
        if debug:
            print('iteration {0} dev : {1}'.format(nit, dev))
            print("old x " + xold.__str__())
            print("new x:" + x.__str__())
            print("fx" + fx.__str__())
            print("Jacobi " + J.__str__())
            print(delta)
    if (nit == maxit):
        print("{0} dimensional Newton Raphson has not converged in {1} iterations".format(xdim, nit))
        print("error = {0}".format(dev))
        return (None, Xvals)
    else:
        if debug:
            print('Newton Raphson has converged in {0} iterations'.format(nit))
        return (x, Xvals)


def IVEuler(fun, x0, tvec):
    ''' simple solution of the initial value problem with the Euler method
    :param fun: \dot x = fun(x,t) equation to be solved
    :param x0: initial vector
    :param tvec: vector/numpy.array of increasing step points of t. tvec[0] = 0 !
    '''
    x = np.float32(x0)
    solution = np.hstack((tvec[0], x))
    for it in range(tvec.__len__()):
        try:
            t = tvec[it]
            dx = fun(x, t)
            if (it < tvec.__len__() - 1):
                x = x + dx * (tvec[it + 1] - tvec[it])
                solution = np.vstack((solution, np.hstack((tvec[it + 1], x))))
        except OverflowError as err:
            print("ERROR: OverflowError occured in IVEuler at ")
            print("iter: {} t : {}".format(it, t))
            print(x)
    return solution

def IVRetorgrad2(fun, x0, tvec):
    ''' solution of the initial value problem with multistep method = 2 step BDF .
        https://en.wikipedia.org/wiki/Backward_differentiation_formula
    secoond order retrograde

    :param fun: \dot x = fun(x,t) equation to be solved
    :param x0: initial vector
    :param tvec: vector/numpy.array of increasing step points of t. tvec[0] = 0 !
    '''
    x = np.float32(x0)
    solution = np.hstack((tvec[0], x))
    # first step is Euler:
    it= 0
    dx = fun(x, tvec[it])
    x = x + dx * (tvec[it + 1] - tvec[it])
    solution = np.vstack((solution, np.hstack((tvec[it + 1], x))))
    # the rest is retrograde
    for it in range(1,tvec.__len__()-1):
        t = tvec[it]
        fi = fun(x, t)
        h = tvec[it + 1] - tvec[it]
        xim1 = x # x_{i-1}
        xim2 = solution[-2,1:] # x_{i-2}
        if (it < tvec.__len__() - 1):
            x = 4.0/3.0 * xim1 - 1.0/3.0 * xim2 + 2.0/3.0 * h * fi
            solution = np.vstack((solution, np.hstack((tvec[it + 1], x))))
    return solution


def RungeKutta2(fun, x0, tvec):
    ''' Second order Runge Kutta method for 1D initial value problem
    :param fun: \dot x = fun(x,t) equation to be solved
    :param x0: initial vector
    :param tvec: vector/numpy.array of increasing step points of t. tvec[0] = 0 !
    '''
    x = np.float32(x0)
    solution = np.hstack((tvec[0], x))
    for it in range(tvec.__len__()):
        t = tvec[it]
        if (it < tvec.__len__() - 1):
            h = (tvec[it + 1] - tvec[it])
            k1 = h * fun(x, t)
            k2 = h * fun(x + k1 / 2.0, t + h / 2.0)
            x = x + k2
            solution = np.vstack((solution, np.hstack((tvec[it + 1], x))))
    return solution
## TODO : write Runge-Kutta 4!


def RungeKutta4(fun, x0, tvec):
    ''' 4. order Runge Kutta method for 1D initial value problem
    :param fun: \dot x = fun(x,t) equation to be solved
    :param x0: initial vector
    :param tvec: vector/numpy.array of increasing step points of t. tvec[0] = 0 !
    '''
    x = np.float32(x0)
    solution = np.hstack((tvec[0], x))
    for it in range(tvec.__len__()):
        t = tvec[it]
        if (it < tvec.__len__() - 1):
            h = (tvec[it + 1] - tvec[it])
            k1 = h * fun(x, t)
            k2 = h * fun(x + k1 / 2.0, t + h / 2.0)
            k3 = h * fun(x + k2 / 2.0, t + h / 2.0)
            k4 = h * fun(x + k3 , t + h )
            x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            solution = np.vstack((solution, np.hstack((tvec[it + 1], x))))
    return solution


def updateY(updaterFunction, lastY, xlast, dX, xend):
    """ a helper function for adaptive_impicit"""
    if ((xlast + dX) > xend):
        dX = xend - xlast
    try:
        newY, error = updaterFunction(lastY, xlast, dX)
    except:
        print("error occured in updaterFunction by {} dX : {}".format(xlast, dX))
        print(lastY)
        print(sys.exc_info())
        return (None, None)
    return newY, error

def adaptive_implicit(xstart, xend, xStep0, maxStep, updaterFunction,
                      initY, stepCorrection, stopCondition, maxIt = 10000 ):
    """| stepwise solving an implicit differential equation
    | start from initial value initY, and stepsize xStep0,
    :param xStep0: initial stepsize
    :param maxStep: maximum stepsize
    :param updaterFunction: updaterFunction(Ylast, xlast, deltaX) = Ynew the function that calculates the
    new Y values from the old one, and from the x-step
    :param initY: initial Y values
    :param stepCorrection: correction factor for stepsize as function of deltaY
    :param stopCondition: a boolean function, when it's true, the iterations stop
    :param maxIt: maximum number of iterations
    :return: the solution as a numpy array
    """
    xnew = xstart + xStep0
    dX = xStep0
    res = np.append( [xstart] , np.array( initY) )
    xlast = xstart
    lastY = initY
    dY = 0
    it = 1
    error_tolerance = 1.0e-3 # tolerance that we allow by solving the step-equation numerically
    while (xnew < xend) & (it < maxIt):
        it += 1
        newYguess, error = updateY(updaterFunction, lastY, xlast, dX, xend)

        if (newYguess is None):
            logging.debug("update calculation failed")
            logging.debug("xlast {}, last calculation at x={} :".format(xlast, xlast + dX) )
            dX = dX/100.
            logging.debug("iter {}. try smaller stepsize dX = {}".format(it, dX))
            newY, error = updateY(updaterFunction, lastY, xlast, dX, xend)
            if newY is None:
                logging.info("update failed by {} also with dX={}. stop solver".format(xlast, dX))
                logging.info(lastY)
                return res
        else: # check the Y changes
            dY = newYguess - lastY
            stepFac = stepCorrection(np.array(dY))
            # if the result changes too much or too little, then change the stepsize:
            if (stepFac is None):
                logging.info("step correction failed, stopping solver")
                logging.info("xlast {}, dX {} dY :".format(xlast, dX) )
                logging.info(dY)
                return res
            if ((stepFac != 1.0) & ((xlast + dX) < xend)):
                logging.debug("x: {} stepsize corrected by {}".format( xlast, stepFac) )
                dX = min( dX * stepFac, maxStep, xend-xlast)
                logging.debug("new stepsize = {}".format(dX))
                newY, error = updateY(updaterFunction, lastY, xlast, dX, xend)
                if(newY is not None):
                    dY = newY - lastY
                else:
                    logging.info("update failed by {} with dX={}. stop solver".format(xlast, dX))
                    return res
            else: # previous stepsize is OK
                newY = newYguess
            if ( error > error_tolerance ): # the update equation was inproperly solved
                dX = dX * error_tolerance / error
                logging.debug("x: {} new stepsize {} because error ({}) is too big ".format(xlast, dX, error))
                ## try to solve the equation with a smaller step-size:
                newY, error = updaterFunction(lastY, xlast, dX)
                dY = newY - lastY
                logging.debug("new error : {}".format(error))
                if (error > 5 * error_tolerance):
                    logging.info("can't control error, error is still {}. stop iterations ".format(error) )
                    return res
        if( (xlast + dX) > xend):
            logging.info(" x interval end reached xlast {} +  dX {} vs xend {}".format(xlast, dX, xend))
            dX = xend - xlast
            newY, error = updaterFunction(lastY, xlast, dX)
            dY = newY - lastY
        ### update the values. take 1 step:
        xnew = min(xlast + dX, xend)
        lastY = newY
        res = np.vstack((res, np.append( [xnew] , np.array( newY) )))
        xlast = xnew
        dYnorm = np.linalg.norm(dY)
        if ((dYnorm == 0.0) & (dX < 1.e-12) ):
            logging.info("no changes detected, and the stepsize got too small. we stop the iterations at")
            logging.info(" it {} xlast {} dX {} stepFac {}  dYnorm {} ".format(it, xlast, dX, stepFac, dYnorm))
            return res
        if( stopCondition(newY)):
            logging.info("x: {} condition for stopping hit".format(xlast))
            logging.info("y: " + str(newY))
            logging.info("stop iterations ")
            return res
    logging.info("solver finished in {} iterations. final point xlast ={}".format(it, xlast))
    return res
        # vph_update(self, vphold, xlast, deltaX, constants)


# def OLD_adaptive_implicit(xstart, xend, xStep0, maxStep, updaterFunction,
#                       initY, stepCorrection, stopCondition, maxIt = 10000 ):
#     """| stepwise solving an implicit differential equation
#     | start from initial value initY, and stepsize xStep0,
#     :param xStep0: initial stepsize
#     :param maxStep: maximum stepsize
#     :param updaterFunction: updaterFunction(Ylast, xlast, deltaX) = Ynew the function that calculates the
#     new Y values from the old one, and from the x-step
#     :param initY: initial Y values
#     :param stepCorrection: correction factor for stepsize as function of deltaY
#     :param stopCondition: a boolean function, when it's true, the iterations stop
#     :param maxIt: maximum number of iterations
#     :return: the solution as a numpy array
#     """
#     xnew = xstart + xStep0
#     dX = xStep0
#     res = np.append( [xstart] , np.array( initY) )
#     xlast = xstart
#     lastY = initY
#     it = 1
#     while (xnew < xend) & (it < maxIt):
#         it += 1
#         if ((xlast + dX) > xend):
#             dX = xend - xlast
#         try:
#             newYguess, error = updaterFunction( lastY, xlast, dX )
#         except :
#             print("error occured in updaterFunction by {} dX : {}".format(xlast, dX) )
#             print(lastY)
#             print(sys.exc_info())
#         if (newYguess is None):
#             print("update calculation failed, stopping solver")
#             print("xlast {}, last calculation at x={} :".format(xlast, xlast + dX) )
#             return res
#         dY = newYguess - lastY
#         stepFac = stepCorrection(np.array(dY))
#         if (stepFac is None):
#             print("step correction failed, stopping solver")
#             print("xlast {}, dX {} dY :".format(xlast, dX) )
#             print(dY)
#             return res
#         if ((stepFac != 1.0) & ((xlast + dX) < xend)):
#             print("x: {} stepsize corrected by {}".format( xlast, stepFac) )
#             dX = min( dX * stepFac, maxStep, xend-xlast)
#             print("new stepsize = {}".format(dX))
#             newY, error = updaterFunction(lastY, xlast, dX)
#             dY = newY - lastY
#         else: # previous stepsize is OK
#             newY = newYguess
#         if(error > 1.e-3): # the update equation was inproperly solved
#             dX = dX * 1.e-3 / error
#             print("x: {} new stepsize {} because error ({}) is too big ".format(xlast, dX, error))
#             newY, error = updaterFunction(lastY, xlast, dX)
#             dY = newY - lastY
#             print("new error : {}".format(error))
#             if (error > 5e-3):
#                 print("can't control error, stop iterations ")
#                 return res
#         if( (xlast + dX) > xend):
#             print(" x interval end reached xlast {} +  dX {} vs xend {}".format(xlast, dX, xend))
#             dX = xend - xlast
#             newY, error = updaterFunction(lastY, xlast, dX)
#             dY = newY - lastY
#         ### update the values. take 1 step:
#         xnew = xlast + dX
#         lastY = newY
#         res = np.vstack((res, np.append( [xnew] , np.array( newY) )))
#         xlast = xnew
#         if( stopCondition(newY)):
#             print("x: {} condition for stopping hit".format(xlast))
#             print("y: " + str(newY))
#             print("stop iterations ")
#             return res
#     return res
#         # vph_update(self, vphold, xlast, deltaX, constants)


def findMaxTrue(x0, transformer, condition, tol):
        """Bisection method. find maximum value of x for that
        out = transformer(x) satisfies condition(out) == True

        :param x0: mimum x value, where we know condition(x0) = True
        :param transformer: a function that transforms x to y
        :param condition: a boolean function that we are checking to be true
        :param tol: tolerance for x
        :return:
        """
        # assert condition(transformer(x0)) , "wrong x0 startvalue, condition not satisfied"
        assert x0 > 0 , " positive initial value is needed"
        t0 = time.time()
        ## finding a value where condi is false:
        if (  condition(transformer(x0)) ): # is the condition satisfied for x0 ?
            conditrue = True
            upper = 2 * x0
            condit = condition(transformer(upper))
            while condit:
                upper = 2.0 * upper
                condit = condition(transformer(upper))
                print(upper)
            lower = upper / 2.0
        else:
            conditrue = False
            upper = x0
            lower = x0 / 2.0
            condit = condition(transformer(lower))
            while not condit:
                lower = lower / 2.0
                condit = condition(transformer(lower))
                print(lower)
            upper = lower * 2.0
        print("result is between {} and {}".format(lower, upper))
        ## halfing method to find the root
        span = upper - lower
        t1 = time.time()
        it = 0
        while span > tol:
            mid = (upper + lower ) / 2.0
            condit = condition(transformer(mid))
            if condit :
                lower = mid
            else :
                upper = mid
            span = upper - lower
            print("solution between {} - {}".format(lower, upper) )
            it+=1
        print("bisection method {} iterations in {} + {} sec".format(
            it, round(t1- t0, 3), round(time.time()- t1, 3)))
        return lower





if __name__ == '__main__':
    # polynomial function for test
    p1 = lambda x: math.pow(x, 3) - 2 * pow(x, 2) + 3.2 * x + 7.0

    dp1Odx = diff(p1, 1.0)

    cross = NewtonRaphson(p1, 145, 0.03, 35)

    fun = lambda x: np.power(x, 2.0) + np.sin(x)
    fun2 = lambda x: x[0] * x[1] * np.eye(3)[1] + np.power(x, 2.0) + np.sin(x) + np.max(x)

    Jmat = Jacobian(fun2, [1.0, 2.4, -4.2])

    fun3a = lambda x: x[0] * x[1] * np.eye(3)[1] + np.power(x, 2.0) - x + np.array([-1.0, 2.3, 0.78])
    fun3b = lambda x: x[0] * x[1] * np.eye(3)[1] + np.power(x, 2.0) - x - np.array([1.0, 2.3, 0.78])
    root = NewtonND(fun3b, np.ones(3))
    fun3b(root[0])

    x0 = -1
    fy = lambda x, t: 10 - t
    sol = IVEuler(fy, x0, np.linspace(0, 5, 100))
    sol2 = RungeKutta2(fy, x0, np.linspace(0, 5, 100))
    sol4 = RungeKutta4(fy, x0, np.linspace(0, 5, 100))

    print(sol[-5:])
    print(sol2[-5:])
    print(sol4[-5:])

    print("testing findMaxTrue algorithm:")
    transformer = lambda x : 0.5 * x + 10.5
    condition = lambda y : (y*0.33 < 130 )
    lower = findMaxTrue(0.1, transformer, condition, tol= 1e-3)
