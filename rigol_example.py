import time
import json
from Oscilloscopes import RIGOL
from PolarizationControllers import OzOptics
from Optimizers import PSO

CONFIG_FILE = "CONFIG.json"

with open (CONFIG_FILE, 'r') as j_file:
    CONFIGS = json.load(j_file)


def scope_configuration(brand= "RIGOL", auto_set= False, source_channel= 1):
    scope = RIGOL(CONFIGS)
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



controller = p_controller_configuration()
acquirer = scope_configuration()
optimizer = PSO(CONFIGS, acquirer, controller)

acquirer.capture()
acquirer.visualise(acquirer.scaled_data)

acquirer.smoother()
acquirer.visualise(acquirer.smoothed_data)

i, f = acquirer.extract_period_index_v4(s_channel= 1)
acquirer.clean_wave_form_data(i, f)
#acquirer.visualise(acquirer.cleaned_data)
type_one = acquirer.discriminator(s_channel= 1)
print("type one wave: ", type_one)
acquirer.qber_calculator(type_one)
print("hv_qber: ",acquirer.hv_qber)
print("pm_qber: ",acquirer.pm_qber)
print("qber: ",acquirer.qber)




counter = 0
while True:
    acquirer.get_data(source_channel= 1)
    #print("live pm_qber: ",acquirer.pm_qber)
    print("live hv_qber: ",acquirer.hv_qber)
    print("QBER: ", acquirer.qber)
    if acquirer.qber < 0.1:
        acquirer.visualise(acquirer.cleaned_data)
        print("type one wave: ", acquirer.discriminator(s_channel= 1))
        acquirer.visualise(acquirer.smoothed_data)
    if 0.12 < acquirer.qber:
        time.sleep(0.3)
        optimizer.run()
    time.sleep(0.2)
    counter += 1
    if counter % 10 == 0:
        acquirer.visualise(acquirer.smoothed_data)
        acquirer.extract_results("new_output.csv")
