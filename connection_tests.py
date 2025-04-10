import time
import json
from Oscilloscopes import *
from Polarimeters import *
from PolarizationControllers import *

CONFIG_FILE = "CONFIG.json"

POLARIZATION_CONTROLLER = "OzOptics"
OSCILLOSCOPE = "RIGOL"
POLARIMETER = "ThorLabs"

with open (CONFIG_FILE, 'r') as j_file:
    CONFIGS = json.load(j_file)


def p_controller_configuration(brand= "OzOptics"):
    p_controller = OzOptics(CONFIGS)
    p_controller.connect()
    return p_controller

def scope_configurations(brand= "OWON"):
    scope = RIGOL(CONFIGS)
    print("Scope Address: ", scope.resource_address)
    print(scope.query("*IDN?"))
    return scope


controller = p_controller_configuration()
controller.send_voltages([0, 0, 0, 0])
scope = scope_configurations()
