import time
import json
import LoggingConfiguration
import logging
from instruments.Oscilloscopes import RIGOL
from instruments.PolarizationControllers import OzOptics
from optimizers.Optimizers import PSO

CONFIG_FILE = "../CONFIG.json"

with open (CONFIG_FILE, 'r') as j_file:
    CONFIGS = json.load(j_file)

LoggingConfiguration.configure_logging(CONFIGS["logging"])
logger = logging.getLogger("main")


def scope_configuration(brand= "RIGOL", auto_set= False, source_channel= 1):
    scope = RIGOL(CONFIGS)
    logger.debug("scope with address {} start configuring".format(scope.resource_address))
    print("Scope Address: ", scope.resource_address)
    if auto_set:
        scope.auto_set_device()
        print("auto set is done!")
    print("wait for initialisation!")
    scope.initialise()
    time.sleep(5)
    return scope


def p_controller_configuration(brand= "OzOptics"):
    p_controller = OzOptics(CONFIGS)
    p_controller.connect()
    return p_controller


acquirer = scope_configuration()


for i in range(3):
    try:
        acquirer.update_data(additional_data= [None, None, None, None], source_channel= 1)
        acquirer.isualise(acquirer.scaled_data)
        acquirer.visualise(acquirer.smoothed_data)
        acquirer.visualise(acquirer.cleaned_data)
        print("type one wave: ", acquirer.discriminator(s_channel= 1))
        time.sleep(10)
    except:
        print("something went Wrong!")
