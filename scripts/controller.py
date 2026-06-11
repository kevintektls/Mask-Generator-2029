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

# To prevent when device is in use or locked
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0 

MAX_DUTY_CYCLE = 0.3 # can be changed
SERVO_CENTER   = 0.5
SERVO_RANGE    = 0.3
POLL_INTERVAL = 0.05

def main():
    print("[INFO] Robot Car Controller Starting...")
    if not Gamepad.available():
        print("[INFO] Waiting for gamepad to be connected...")
        while not Gamepad.available():
            time.sleep(1)
    print("[INFO] Gamepad connected.")

    gamepad = GAMEPAD_TYPE()
    gamepad.startBackgroundUpdates()

if __name__ == '__main__':
    main()