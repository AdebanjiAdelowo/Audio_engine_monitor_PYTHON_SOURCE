import smbus 
import time

# ADXL345 I2C address
ADXL345_ADDR = 0x53

# ADXL345 Register addresses
DATA_FORMAT = 0x31  # Data format control register
POWER_CTL = 0x2D    # Power control register
DATAX0 = 0x32       # X-axis data register (low byte)
DATAY0 = 0x34       # Y-axis data register (low byte)
DATAZ0 = 0x36       # Z-axis data register (low byte)

# Initialize the I2C bus
bus = smbus.SMBus(1)

# Function to initialize the ADXL345
def init_adxl345():
    # Power on the ADXL345
    bus.write_byte_data(ADXL345_ADDR, POWER_CTL, 0x08)
    time.sleep(0.1)
    
    # Set data format to full resolution (±16g, 4mg/LSB)
    bus.write_byte_data(ADXL345_ADDR, DATA_FORMAT, 0x08)
    time.sleep(0.1)

# Function to read 16-bit signed data from the sensor (2 bytes for each axis)
def read_axis(axis):
    low = bus.read_byte_data(ADXL345_ADDR, axis)
    high = bus.read_byte_data(ADXL345_ADDR, axis + 1)
    value = (high << 8) | low  # Combine high and low bytes
    if value > 32767:  # Convert from unsigned to signed
        value -= 65536
    return value

# Function to get acceleration values for X, Y, Z axes
def get_acceleration():
    x = read_axis(DATAX0)
    y = read_axis(DATAY0)
    z = read_axis(DATAZ0)
    
    # Convert the raw data to g values (assuming 4mg/LSB, and full resolution mode)
    x_g = x * 0.004  # 4mg per LSB
    y_g = y * 0.004
    z_g = z * 0.004
    
    return x_g, y_g, z_g

# Function to calibrate and find the zero offset
def calibrate_sensor(samples=100):
    x_offset, y_offset, z_offset = 0, 0, 0
    for _ in range(samples):
        x, y, z = get_acceleration()
        x_offset += x
        y_offset += y
        z_offset += z
        time.sleep(0.01)  # Small delay between samples
    
    # Average the offsets over the number of samples
    x_offset /= samples
    y_offset /= samples
    z_offset /= samples
    
    return x_offset, y_offset, z_offset

# Main function to run the program
def main():
    init_adxl345()  # Initialize ADXL345 sensor
    print("ADXL345 Vibration Sensor Initialized\n")
    print("Calibrating sensor... Please wait...\n")
    
    # Calibrate the sensor by taking a few initial readings
    x_offset, y_offset, z_offset = calibrate_sensor(samples=100)
    print(f"Calibration complete. Offsets - X: {x_offset:.3f} g, Y: {y_offset:.3f} g, Z: {z_offset:.3f} g")
    print("\nReading vibration data from the sensor...\n")
    
    try:
        while True:
            x, y, z = get_acceleration()

            # Subtract the calibration offsets
            x -= x_offset
            y -= y_offset
            z -= z_offset

            # Provide clear and easy-to-understand output
            print("Vibration Levels (in g - acceleration due to gravity):")
            print(f"  X-axis (Left-Right): {x:.3f} g")
            print(f"  Y-axis (Up-Down): {y:.3f} g")
            print(f"  Z-axis (Forward-Backward): {z:.3f} g")
            print("\n---------------------------------------------\n")

            # Check for excessive vibration (above 0.2 g)
            if abs(x) > 0.2 or abs(y) > 0.2 or abs(z) > 0.2:
                print("WARNING: Excessive Vibration Detected!")
                print("Potential mechanical or structural issue.")
                print("Vibration readings exceed the safe threshold of 0.2 g.")
                print("---------------------------------------------\n")
            
            time.sleep(0.5)  # Wait for half a second before reading again
    except KeyboardInterrupt:
        print("Program terminated by user.")
        pass

if __name__ == "__main__":
    main()
