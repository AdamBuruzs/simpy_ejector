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
# class to store Ejector Geometry
from simpy_ejector import NozzleParams, nozzleFactory
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

class EjectorGeom(object):
    """class to store Ejector Geometry """

    def __init__(self, nozzle: NozzleParams, Dm):
        """
        :param nozzle: The nozzle object storing the geometry of the Nozzle
        :param Dm: the mixer diameter in cm (The mixer has constant diameter!)
        """
        self.nozzle = nozzle
        self.Dm = Dm
        self.Am = math.pow( Dm/2, 2.0) * math.pi # mixer cross section area in cm^2

    def setMixer(self, mixerstart, mixerLen, diffuserLen, diffuserHeight, mixerAngle = 0.0):
        """ specify a constant area mixer with conical diffuser

        :param mixerstart: x position of the beginning of the mixer region
        :param mixerLen: length of the mixer in cm
        :param diffuserLen: length of the diffuser
        :param diffuserHeight: Radius of the diffuser
        :param mixerAngle: angle of the mixer, zero for constant pressure diffusers
        :return:
        """
        self.mixerStart = mixerstart
        self.mixerLen = mixerLen
        self.diffuserLen = diffuserLen
        self.diffuserHeight= diffuserHeight
        self.mixerAngle = mixerAngle
        self.mixerRad = self.Dm/2
        self.ejectorlength = self.mixerStart + self.mixerLen + self.diffuserLen


    def mixerR(self, x):
        """ radius as function of x -coordinate
        for the constant area mixer and the conical diffuser"""
        if (x < self.mixerStart):
            out = 0.0
        elif (x <= self.mixerStart + self.mixerLen):
            out = self.mixerRad
        elif (x <= self.ejectorlength):
            xd = x - self.mixerStart - self.mixerLen
            out = xd/ self.diffuserLen * (self.diffuserHeight - self.mixerRad) + self.mixerRad
        else:
            print("ejector length reached")
            out = None
        return out

    def mixerdAdx(self, x):
        """ numerical differential of the cross section with respect to the x coordinate

        :param x: x ccordinate in cm
        :return: dA/dx in cm
        """
        eps = 0.0001
        if (x < (self.mixerStart + self.mixerLen)) :
            out = 0.0
        elif ( x <= self.ejectorlength):
            out = (self.mixerArea(x) - self.mixerArea(x-eps) )/ eps
        else:
            print("ejector length reached")
            out = None
        return out


    def mixerArea(self, x):
        return math.pow(self.mixerR(x), 2.0) * math.pi

    def __str__(self):
        out = ' Nozzle : \n' + str(self.nozzle) + "\n"
        out += out + 'mixing diameter {} cm and area {} cm^2'.format(round(self.Dm,4),round( self.Am, 4) )
        return out

    def draw(self, title = ""):
        """ plot the ejector

        :return:
        """
        xno = np.linspace(0.0, self.nozzle.L, 100) # nozzle x axis
        rno = np.array( [self.nozzle.Rprofile(xi) for xi in xno ] )
        xmix = np.linspace(self.mixerStart, self.ejectorlength, 100) # mixer x axis
        rmix = np.array( [self.mixerR(xi) for xi in xmix ])
        small = self.nozzle.Rprofile(0.0) / 20.0
        fig = plt.figure(figsize=(14,6))
        plt.plot(xno, rno, color = 'b')
        plt.plot(xno, -rno, color = 'b')
        plt.arrow(0.0, 0.0, self.nozzle.L/3.0, 0.0, head_width = small*2, head_length= small *3, color='b')
        # plt.arrow(0.0, 0.0, self.nozzle.L/3.0, 0.0, arrowstyle = '->') # arrowprops = {'arrowstyle': '->', 'lw': 2, 'color': 'b'} )
        plt.plot(xmix, rmix, color= 'g')
        plt.plot(xmix, -1* rmix, color='g')
        plt.grid(True)
        plt.text( 0.0, self.nozzle.Rprofile(0.0)*1.2, 'motive nozzle', {'color': 'b', 'fontsize': 14})
        plt.text(small*5, self.nozzle.Rprofile(0.0) * 1.4, 'suction nozzle', {'color': 'r', 'fontsize': 14})
        plt.text(self.mixerStart + self.mixerLen/2.0, self.mixerRad * 1.2, 'mixer', {'color': 'g', 'fontsize': 14})
        plt.text(self.ejectorlength - self.diffuserLen / 2.0, self.mixerRad * 2.0, 'diffuser', {'color': 'g', 'fontsize': 14})
        plt.title('Ejector geometry')
        plt.xlabel('cm')
        plt.ylabel('cm')
        ## suction entry:
        x1,y1 = [self.nozzle.L , self.nozzle.Rprofile(self.nozzle.L)]
        x2,y2 = [self.mixerStart , self.mixerRad]
        suctg = 0.3
        suc1x = np.linspace(x1- self.nozzle.L/ 3.0, x1, 20)
        suc1y = suctg * (x1 - suc1x ) + y1
        suc2x = np.linspace(x2 - self.nozzle.L / 3.0, x2, 20)
        suc2y =  suctg * (x2 - suc2x ) + y2
        plt.arrow(x1 - self.nozzle.L/ 3.0 + small *10, suc1y[0] + (y2-y1)/2.0 , small*10, - small*10*suctg ,
                  head_width = small*2, color ='r' )
        plt.plot(suc1x, suc1y, color = 'r')
        plt.plot(suc2x, suc2y, color='r')
        plt.plot(suc1x, - suc1y, color = 'r')
        plt.plot(suc2x, - suc2y, color='r')
        plt.title(title)
        return fig



if __name__ == '__main__':
    nozzle = nozzleFactory.ConicConic(1.0, 2.905, 0.2215, 1.4116, 0.345)
    # Dm = 1.4, Lm = 11.2
    ejector = EjectorGeom(nozzle, Dm = 1.4 )
    # Ld = 25.2
    # DH = 18
    ejector.setMixer(mixerstart = 5.514, mixerLen= 11.2, diffuserLen= 25.2, diffuserHeight= 1.80)

    ejector.draw("my first ejector")
