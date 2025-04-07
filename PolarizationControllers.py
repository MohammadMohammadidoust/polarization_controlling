import serial
import logging

logger = logging.getLogger(__name__)

class OzOptics(object):
    def __init__(self, conf_dict: dict[str, object]) -> None:
        self.configs = conf_dict
        self.start_voltage = np.array(self.configs['p_controller']['ozoptics']['initial_state'])
        self.current_voltages = np.array(self.configs['p_controller']['ozoptics']['initial_state'])
        self.step = self.configs['p_controller']['ozoptics']['step_size']
        self.port = self.configs['p_controller']['ozoptics']['port']
        self.baudrate = self.configs['p_controller']['ozoptics']['baudrate']
        self.timeout = self.configs['p_controller']['ozoptics']['timeout']
        self.min_voltage = self.configs['p_controller']['ozoptics']['min_voltage']
        self.max_voltage = self.configs['p_controller']['ozoptics']['max_voltage']

    def connect(self) -> None:
        try:
            self.device = serial.Serial(self.port, self.baudrate, timeout= self.timeout)
            logger.info("OzOptics connected successfully!")
        except:
            logger.critical("OzOptics connection failed")
            exit()

    def update_voltages(self, volts: list[int] | np.ndarray) -> None:
        self.current_voltages = volts

    def send_voltages(self, volts: list[int] | np.ndarray) -> None:
        self.update_voltages(volts)
        if self.device.isOpen():
            self.device.write(("V1,"+str(int(volts[0]))+"\r\n").encode('ascii'))
            self.device.write(("V2,"+str(int(volts[1]))+"\r\n").encode('ascii'))
            self.device.write(("V3,"+str(int(volts[2]))+"\r\n").encode('ascii'))
            self.device.write(("V4,"+str(int(volts[3]))+"\r\n").encode('ascii'))
        else:
            logger.error("Polarization Controller is not open!")
            raise RuntimeError("Polarization Controller is not connected!")

    def reset_voltages(self) -> None:
        self.send_voltages([0, 0, 0, 0])

    def action_to_voltages(self, actions: str) -> list[int] | np.ndarray:
        new_voltages = self.current_voltages.copy()  
        mapping = {"U": self.step, "D": -self.step}
        for i, action in enumerate(actions):
            new_voltages[i] += mapping.get(action, 0)
        return new_voltages
