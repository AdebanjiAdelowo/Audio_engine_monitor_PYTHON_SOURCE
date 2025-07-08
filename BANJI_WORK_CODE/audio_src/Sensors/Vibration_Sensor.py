import smbus
import time

class VibrationSensor:
    def __init__(self, address=0x53):
        self.bus = smbus.SMBus(1)
        self.address = address

    def read_accelerometer(self):
        """
        Reads accelerometer data and returns X, Y, Z axis values.
        """
        try:
            # Wake up MPU6050
            self.bus.write_byte_data(self.address, 0x6B, 0)
            time.sleep(0.1)

            accel_x = self.read_word_2c(0x3B)
            accel_y = self.read_word_2c(0x3D)
            accel_z = self.read_word_2c(0x3F)

            return accel_x, accel_y, accel_z
        except Exception as e:
            print(f"Error reading accelerometer: {e}")
            return None

    def read_word_2c(self, reg):
        """
        Reads a 16-bit value from the sensor.
        """
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val

    def get_vibration_status(self):
        """
        Returns an easy-to-understand vibration status.
        """
        accel_data = self.read_accelerometer()
        if accel_data:
            x, y, z = accel_data
            vibration_intensity = abs(x) + abs(y) + abs(z)

            # Define status messages
            if vibration_intensity < 5000:
                status = "Normal Vibration"
            elif vibration_intensity < 10000:
                status = "Moderate Vibration - Check Engine"
            else:
                status = "High Vibration - Possible Engine Issue!"

            return status, vibration_intensity
        return "Vibration Sensor Error", 0

if __name__ == "__main__":
    sensor = VibrationSensor()
    while True:
        status, intensity = sensor.get_vibration_status()
        print(f"[Vibration] Intensity: {intensity} | Status: {status}")
        time.sleep(1)
