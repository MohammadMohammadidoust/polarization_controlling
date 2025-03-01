import time
import json
from Oscilloscopes import OWON
from Polarimeters import Thorlabs
from PolarizationControllers import OzOptics
from Optimizers import PSO

CONFIG_FILE = "CONFIG.json"

with open (CONFIG_FILE, 'r') as j_file:
    CONFIGS = json.load(j_file)


def scope_configuration(brand= "OWON", channel_mode= "quad", auto_set= False,
                         source_channel= 1):
    scope = OWON(CONFIGS)
    print("Scope Address: ", scope.resource_address)
    if auto_set:
        scope.auto_set_device()
        print("auto set is done!")
    print("wait for initialisation!")
    scope.initialise()
    time.sleep(5)
    sample_rate = scope.get_sample_rate(channel_mode)
    print("sample_rate: ", sample_rate)
    print("time base ", scope.time_base)
    print("horizontal offset: ", scope.horizontal_offset)
    print("vertical offset: ", scope.vertical_offset)
    print("voltage_scale: ", scope.voltage_scale)
    return scope


def p_controller_configuration(brand= "OzOptics"):
    p_controller = OzOptics(CONFIGS)
    p_controller.connect()
    return p_controller



controller = p_controller_configuration()
acquirer = scope_configuration()
optimizer = PSO(CONFIGS, acquirer, controller)


counter = 0
while True:
    acquirer.get_data()
    print("qber: ",acquirer.qber)
    if acquirer.qber > 0.02:
        optimizer.run()
    time.sleep(0.2)
    counter += 1
    if counter % 10 == 1:
        acquirer.extract_results("output.csv")

