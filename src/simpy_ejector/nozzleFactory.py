
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
from simpy_ejector import NozzleParams

## Create Convergent Divergent Nozzles

def ConicConic(Rin, Lcon, Rt, Ldiv,Rout):
    """create a CD Nozzle with Conic convergent and Conic divergent part.
    Each paramters are in cm!

    :param Rin: inlet Radius
    :param Lcon: length of convergent part
    :param Rt: throat radius
    :param Ldiv: length of divergent part
    :param Rout: radius of outlet exit
    :return: the nozzle
    """
    def nozzleRProfile(x):
        if (x < 0):
            return 0.0
        elif ((x >= 0) & (x <= Lcon)):
            return Rin - x / Lcon * (Rin - Rt)
        elif ((x >= Lcon) & (x <= Lcon + Ldiv)):
            return Rt + (x - Lcon) / Ldiv * (Rout - Rt)
        else:
            return 0

    nozzle = NozzleParams.NozzleParams.fromRProfile(nozzleRProfile, Lcon + Ldiv, Lcon)
    return nozzle