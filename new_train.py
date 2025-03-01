import time
from OWON import Scope


scope = Scope("CONFIG.json")

print(scope.resource_address)
#scope.auto_set_device()
print("auto set is done!")
print("wait for initialisation!")
time.sleep(5)
scope.initialise()
time.sleep(5)

sample_rate = scope.get_sample_rate('quad')
print("sample_rate", sample_rate)
#scope.update_device_parameters(channel= 1)
print("time base ", scope.time_base)
print("horizontal offset: ", scope.horizontal_offset)
print("vertical offset: ", scope.vertical_offset)
print("voltage_scale: ", scope.voltage_scale)

while True:
    scope.capture()
    scope.calculate_voltage_and_time()
    test_wave_form = scope.scaled_data[2]
    v4 = scope.extract_period_index_v4(test_wave_form)
    i_index = v4[0]
    f_index = v4[1]
    scope.clean_wave_form_data(i_index, f_index)
    scope.qber_calculator()
    print("hv_qber: ", scope.hv_qber)
    print("pm_qber: ", scope.pm_qber)
    time.sleep(0.5)
#scope.capture()
#scope.calculate_voltage_and_time()
#test_wave_form = scope.scaled_data[2]
#v4 = scope.extract_period_index_v4(test_wave_form)
#i_index = v4[0]
#f_index = v4[1]
#scope.clean_wave_form_data(i_index, f_index)
#scope.qber_calculator()
#print("hv_qber: ", scope.hv_qber)
#print("pm_qber: ", scope.pm_qber)
#time.sleep(0.5)