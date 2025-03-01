import pyvisa

address = "USB0::0x5345::0x1235::2052231::INSTR"
resource = pyvisa.ResourceManager()
instrument = resource.open_resource(address)
print(instrument.query("*IDN?"))
