from pynput.keyboard import Key, Controller
import serial
import time

# Set up serial connection
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Replace '/dev/ttyUSB0' with your correct port
time.sleep(2)  # Give some time to establish the connection

keyboard = Controller()

def listen_serial():
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()  # Read the serial input
            if line == "footSwitchPressed":
                print("Foot switch pressed! Triggering key press...")
                # Simulate a key combination (e.g., Ctrl+C)
                with keyboard.pressed(Key.ctrl):
                    with keyboard.pressed(Key.alt):
                        keyboard.press('p')
                        keyboard.release('p')
        time.sleep(0.1)

# Start listening to the serial input
listen_serial()
