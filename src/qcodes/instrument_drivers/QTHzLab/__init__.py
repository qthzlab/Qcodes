"""
QTHzLab Instrument Drivers
==========================

Custom QCoDeS instrument drivers for the Quantum THz Lab
at Georgia State University, Department of Physics and Astronomy.

Available Instruments:
----------------------
- DHT11: Arduino-based temperature/humidity sensor with SCPI-like interface
- SR850: Stanford Research Systems SR850 Lock-in Amplifier

Usage:
------
    from qcodes.instrument_drivers.QTHzLab import DHT11, SR850
    
    # Or import individually
    from qcodes.instrument_drivers.QTHzLab.DHT11 import ArduinoDHT11
    from qcodes.instrument_drivers.QTHzLab.SR850 import SR850

Author: QTHzLab @ GSU
"""

# Import instruments for convenient access
from .DHT11 import ArduinoDHT11

# If you have the SR850 driver in this folder, uncomment:
# from .SR850 import SR850

# Define what's available when someone does "from QTHzLab import *"
__all__ = [
    'ArduinoDHT11',
    # 'SR850',  # Uncomment when SR850.py is added
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'QTHz Lab, Georgia State University'