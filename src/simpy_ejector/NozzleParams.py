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
import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

class NozzleParams(object):
    """The class cotaining the Nozzle geometric parameters, and shape
        """

    def __init__(self, Ain, At, Ao, L, xt, Aprofile = None, Rprofile = None):
        """
       :param Ain: inlet cross section area (! in cm^2)
       :param At: throat cross section area (! in cm^2)
       :param Ao: outlet (! in cm^2)
       :param L: length [cm]
       :param xt: position of the throat [cm]
       :param Aprofile: A function of x dependent cross section area of the axis symmetric nozzle A(x) in cm^2
       :param Rprofile: A function of x dependent radius of the axis symmetric nozzle R(x) in cm
        """
        self.Ain = Ain
        self.At = At
        self.Ao = Ao
        self.L = L
        self.xt = xt
        self.Aprofile = Aprofile
        self.Rprofile = Rprofile

    def set_Aprofile(self, Aprofile):
        self.Aprofile = Aprofile

    def setFriction(self, frictionCoef):
        self.frictionCoef = frictionCoef

    @classmethod
    def fromRProfile(cls, RProfile : Callable, L:float, xt:float ):
        '''| define a Nozzle from the R(x) function (that is the Radius(x -coordinate))
        :param RProfile: profile of cross sectional Radius: R(x) in cm
        :param L: length of Nozzle
        :param xt: the position of the throat
        :return:
        '''
        Ain = math.pow(RProfile(0), 2.0) * math.pi
        Ao = math.pow(RProfile(L), 2.0) * math.pi
        xsample = np.linspace(0,L, 10000)
        rsample = np.array( [RProfile(xi) for xi in xsample] )
        Rmin = RProfile(xt)
        ## the throat should be specified totally accurately, it's critical!!!
        At = math.pow(Rmin, 2.0) * math.pi
        Aprofile = lambda x : math.pow(RProfile(x), 2.0) * math.pi
        return cls(Ain,At, Ao,L,xt, Aprofile, RProfile)

    def dAdxNum(self,x):
        """ numerical calculation of the dA/dx derivative
        :param x : x position
        :return: dA/dx in cm^2/cm = cm
        """
        eps = 0.0001
        assert (x >= 0 )
        if (x + eps < self.xt):
            dAdx = (self.Aprofile(x + eps) - self.Aprofile(x)) / eps
        elif( (x + eps >= self.xt) & (x < self.xt) ) :
            dAdx = - (self.Aprofile(x - eps) - self.Aprofile(x)) / eps
        elif( x == self.xt):
            dAdx = 0.0
        elif( (x > self.xt) & (x + eps < self.L )):
            dAdx = (self.Aprofile(x + eps) - self.Aprofile(x)) / eps
        elif( (x + eps > self.L ) & (x  <= self.L )) :
            dAdx = - (self.Aprofile(x - eps) - self.Aprofile(x)) / eps
        else :
            dAdx = None
        return dAdx

    def __str__(self):
        outstr = "Axis symmetric Nozzle with :\n"
        outstr +=  "inlet cross section {0} cm^2 \n".format( self.Ain)
        outstr += ("throat cross section {0} cm^2 \n".format( self.At ))
        outstr += ("throat diameter {0} cm \n".format( math.sqrt( self.At /math.pi) * 2.0 ))
        outstr += ("outlet cross section {0} cm^2 \n".format( self.Ao ))
        outstr += ("outlet diameter {0} cm \n".format( math.sqrt( self.Ao /math.pi) * 2.0 ))
        outstr += ("Length {} cm \n".format(self.L))
        outstr += ("The throat is at {} cm from the inlet \n".format(self.xt))
        if hasattr(self, 'frictionCoef'):
            outstr += ("Friction coefficient {} \n".format(self.frictionCoef))
        return outstr

    def show(self):
        xsample = np.linspace(0, self.L, 10000)
        rsample = [self.Rprofile(xi) for xi in xsample]
        fig = plt.figure()
        plt.plot(xsample, rsample)
        plt.title("Nozzle Radius profile")
        plt.xlabel("x [cm]")
        plt.ylabel("Radius [cm]")

    def gridLinvA(self, startx=0.0, Ni=100):
        """ create 1 dimensional mesh/grid for the nozzle, where the dx interval length is proportional to 1/A.
        That means, more points by small cross-section.

        :param startx: in the x point where the mesh starts
        :param Ni: parameter to control the number of points created
        :return: The meshpoints
        """
        xsample = np.linspace(0, self.L, Ni)
        Asample = np.array([self.Aprofile(xi) for xi in xsample])
        Amax = Asample.max()
        At = self.Aprofile(self.xt)
        dx0 = self.L / Ni
        xpoints = [startx]
        while xpoints[-1] < self.L:
            dx = dx0 * self.Aprofile(xpoints[-1]) / Amax
            xnext = xpoints[-1] + dx
            if (xnext > self.xt) & (xpoints[-1] < self.xt):
                if ( (self.xt - xpoints[-1]) < dx / 3.0 ): # xt is very close to the previous mesh point:
                    xpoints.pop()
                xpoints.append(self.xt)  # the throat point should be in the grid!
            xpoints.append(xnext)
        xpoints[-1] = self.L # last value is the total length
        npxp = np.array(xpoints)
        #print('generated {} grid 1D points'.format(npxp.size))
        return npxp