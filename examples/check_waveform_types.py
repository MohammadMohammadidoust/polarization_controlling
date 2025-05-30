import time
import json
from logs import LoggingConfiguration
import logging
from instruments.Oscilloscopes import RIGOL

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
    time.sleep(4)
    return scope


acquirer = scope_configuration()


acquirer.update_data(additional_data= [None, None, None, None], source_channel= 1)
acquirer.visualise(acquirer.scaled_data)
acquirer.visualise(acquirer.smoothed_data)
acquirer.visualise(acquirer.cleaned_data)
print("type one wave: ", acquirer.discriminator(s_channel= 1))
