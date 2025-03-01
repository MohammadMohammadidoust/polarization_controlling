import time
import json
from Oscilloscopes import OWON
from Polarimeters import Thorlabs
from PolarizationControllers import OzOptics
from Optimizers import PSO

CONFIG_FILE = "CONFIG.json"

with open (CONFIG_FILE, 'r') as j_file:
    CONFIGS = json.load(j_file)


def p_controller_configuration(brand= "OzOptics"):
    p_controller = OzOptics(CONFIGS)
    p_controller.connect()
    return p_controller

def scope_configurations(brand= "OWON"):
    scope = OWON(CONFIGS)
    print("Scope Address: ", scope.resource_address)
    print(scope.query("*IDN?"))
    return scope


controller = p_controller_configuration()
controller.send_voltages([0, 0, 0, 0])
scope = scope_configurations()
