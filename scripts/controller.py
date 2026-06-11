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
