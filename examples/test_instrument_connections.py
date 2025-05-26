import json
from instruments.Oscilloscopes import *
from instruments.Polarimeters import *
from instruments.PolarizationControllers import *

CONFIG_FILE = "../CONFIG.json"
INSTALLED_INSTRUMENTS = {"polarization_controller": "OzOptics",
               "polarimeter": "ThorLabs",
               "oscilloscope": "RIGOL"}

with open (CONFIG_FILE, 'r') as j_file:
    CONFIGS = json.load(j_file)


def p_controller_configuration(brand= "OzOptics"):
    p_controller = OzOptics(CONFIGS)
    p_controller.connect()
    return p_controller

def scope_configuration(brand= "OWON"):
    if brand == "RIGOL":
        scope = RIGOL(CONFIGS)
    elif brand == "OWON":
        scope = OWON(CONFIGS)
    scope.connect()
    return scope


def polarimeter_configuration(brand= "ThorLabs"):
    polarimeter = Thorlabs(CONFIGS)
    polarimeter.connect()
    return polarimeter




instrument_configuration_functions = {
    "polarization_controller": p_controller_configuration,
    "oscilloscope": scope_configuration,
    "polarimeter": polarimeter_configuration}

connected_instruments = {}
for instrument_type, brand in INSTALLED_INSTRUMENTS.items():
    config_func = instrument_configuration_functions.get(instrument_type)
    if config_func:
        try:
            connected_instruments[instrument_type] = config_func(brand)
        except Exception as e:
            print(e)
    else:
        print(f"No configuration function defined for {instrument_type}")