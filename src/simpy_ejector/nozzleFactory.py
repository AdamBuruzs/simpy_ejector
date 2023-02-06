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