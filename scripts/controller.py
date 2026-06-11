#!/usr/bin/env python3
import sys
import time
import gc

# I installed the Gamepad library in a custom location, so I need to add it to the path
sys.path.insert(0, '/home/robotcar/Gamepad')

import Gamepad # gamepad is for logitech F710 connected in usb mode with dungle
from pyvesc import VESC # pyvesc is for cruise control connected in usb mode

GAMEPAD_TYPE = Gamepad.Xbox360 # Logitech F710 has two modes, and the mode X simulate Xbox360 controller
AXIS_FORWARD = "RIGHT-TRIGGER"
AXIS_BACKWARD = "LEFT-TRIGGER"
AXIS_STEERING  = 'RIGHT-X'

DEADZONE = 0.08 # to prevent of micro movements of the joystick, can be changed

VESC_PORT     = '/dev/ttyACM0'
VESC_BAUDRATE = 115200
VESC_TIMEOUT  = 1.0

# To prevent when device is in use or locked
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0 

MAX_DUTY_CYCLE = 0.3 # can be changed
SERVO_CENTER   = 0.5
SERVO_RANGE    = 0.3
POLL_INTERVAL = 0.05

def my_vesc_connect():
    last_exception = None
    for attempt in range(VESC_CONNECT_RETRIES):
        try:
            vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
            print(f'[INFO] Successfully connected to VESC on attempt {attempt + 1}')
            return vesc
        except Exception as e:
            last_exception = e
            print(f'[WARNING] Attempt {attempt + 1} failed: {e}')
            time.sleep(VESC_CONNECT_SETTLE)
    raise Exception(f'Failed to connect to VESC after {VESC_CONNECT_RETRIES} attempts: {last_exception}')

def main():
    print("[INFO] Robot Car Controller Starting...")
    if not Gamepad.available():
        print("[INFO] Waiting for gamepad to be connected...")
        while not Gamepad.available():
            time.sleep(1)
    print("[INFO] Gamepad connected.")

    gamepad = GAMEPAD_TYPE()
    gamepad.startBackgroundUpdates()

    print(f'[INFO] Connecting to VESC on port {VESC_PORT} with baudrate {VESC_BAUDRATE}...')
    try:
        vesc = my_vesc_connect()
        with vesc:
            time.sleep(0.5)
            print(f'[INFO] RT = avancer  |  LT = reculer  |  Joystick droit = direction')
            print(f'[WARNING] Duty cycle is limited to {MAX_DUTY_CYCLE * 100:.1f}% for safety.')
            vesc.set_servo(SERVO_CENTER) # recenter servo on startup
            try: 
                while Gamepad.isConnected(): # prevent of gamepad disconnection
                    forward_raw  = gamepad.axis(AXIS_FORWARD)
                    backward_raw = gamepad.axis(AXIS_BACKWARD)
                    steering_raw = gamepad.axis(AXIS_STEERING)

            except Exception as e:
                print(f'[ERROR] Error reading gamepad axes: {e}')
                return

    except Exception as e:
        print(f'[ERROR] Error occurred while connecting to VESC: {e}')
        return
    except Exception as e:
        print(f'[ERROR] Erreur lors de la connexion au VESC: {e}')
        return

if __name__ == '__main__':
    main()