#!/usr/bin/env python3
import sys
import time
import gc

# I installed the Gamepad library in a custom location, so I need to add it to the path
sys.path.insert(0, '/home/robotcar/Gamepad')

import Gamepad # gamepad is for logitech F710 connected in usb mode with dungle
from pyvesc import VESC # pyvesc is for cruise control connected in usb mode

