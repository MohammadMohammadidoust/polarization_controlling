import serial

class OzOptics(object):
    def __init__(self, conf_dict):
        self.configs = conf_dict
        self.port = self.configs['p_controller']['ozoptics']['port']
        self.baudrate = self.configs['p_controller']['ozoptics']['baudrate']
        self.timeout = self.configs['p_controller']['ozoptics']['timeout']

    def connect(self):
        try:
            self.device = serial.Serial(self.port, self.baudrate, timeout= self.timeout)
            print("OzOptics connected successfully!")
        except:
            print("OzOptics connection failed")

    def send_voltages(self, volts):
        if self.device.isOpen():
            self.device.write(("V1,"+str(int(volts[0]))+"\r\n").encode('ascii'))
            self.device.write(("V2,"+str(int(volts[1]))+"\r\n").encode('ascii'))
            self.device.write(("V3,"+str(int(volts[2]))+"\r\n").encode('ascii'))
            self.device.write(("V4,"+str(int(volts[3]))+"\r\n").encode('ascii'))
        else:
            print("Polarization Controller is not open!")

