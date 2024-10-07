## factory class for material properties
## it is creating material properties instance with either refprop, or coolprop


class MaterialPropertiesFactory:
    @staticmethod
    def create( material, library='auto'):
        refprop_available = False
        coolprop_available = False

        try:
            import ctREFPROP.ctREFPROP as RP
            refprop_available = True
        except ImportError:
            pass

        try:
            import CoolProp.CoolProp as CP
            coolprop_available = True
        except ImportError:
            pass

        if library == 'refprop' and refprop_available:
            from simpy_ejector.refprop_material import RefpropProperties
            return RefpropProperties(material)
        elif library == 'coolprop' and coolprop_available:
            from simpy_ejector.coolprop_material import CoolpropProperties
            return CoolpropProperties(material)
        elif library == 'auto':
            if refprop_available:
                from simpy_ejector.refprop_material import RefpropProperties
                return RefpropProperties(material)
            elif coolprop_available:
                from simpy_ejector.coolprop_material import CoolpropProperties
                return CoolpropProperties(material)
            else:
                raise ImportError("Neither REFPROP nor CoolProp is available.")
        else:
            raise ValueError("Invalid library choice or library not available.")


if __name__ == "__main__":
    # Example usage
    ## WATER,
    material = MaterialPropertiesFactory.create(library='refprop', material = 'butane')
    # density = material.get_property('Water', 'density', 'T', 300, 'P', 101325)
    resprops = material.getTD( hm=270.0, P=350.0, debug=False)
    print(f"Density: {resprops}")
