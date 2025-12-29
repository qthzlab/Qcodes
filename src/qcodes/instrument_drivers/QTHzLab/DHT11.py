"""
QCoDeS Driver for Arduino DHT11 Sensor
======================================

Simple and robust driver.

Author: Abhay's Lab @ GSU
Version: 2.1.0
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any

from qcodes import VisaInstrument
from qcodes.parameters import Parameter
from qcodes.validators import Ints

logger = logging.getLogger(__name__)


class ArduinoDHT11(VisaInstrument):
    """
    QCoDeS driver for Arduino DHT11 sensor.
    
    Example:
        dht = ArduinoDHT11('dht', 'ASRL3::INSTR', visalib='@py')
        print(dht.temperature())
        print(dht.humidity())
    """
    
    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 5,
        **kwargs: Any
    ) -> None:
        
        super().__init__(name, address, terminator='\n', **kwargs)
        
        # Serial settings
        self.visa_handle.baud_rate = 115200
        self.visa_handle.timeout = timeout * 1000
        
        # Wait for Arduino reset
        logger.info("Waiting for Arduino...")
        time.sleep(4)
        
        # Flush Python serial buffer
        self._flush()
        
        # Send dummy command to clear Arduino's garbage bytes
        # (Arduino bootloader leaves 0xf0 bytes in its input buffer)
        logger.info("Clearing Arduino input buffer...")
        self.visa_handle.write('X')
        time.sleep(0.3)
        self._flush()
        
        # Test connection
        time.sleep(0.3)
        idn = self._query('*IDN?')
        logger.info(f"Connected: {idn}")
        
        # Parameters
        self.add_parameter(
            'temperature',
            unit='°C',
            label='Temperature',
            get_cmd=self._get_temperature
        )
        
        self.add_parameter(
            'humidity', 
            unit='%',
            label='Humidity',
            get_cmd=self._get_humidity
        )
        
        self.add_parameter(
            'averaging',
            vals=Ints(1, 16),
            label='Averaging',
            get_cmd=self._get_averaging,
            set_cmd=self._set_averaging
        )
    
    def _flush(self) -> None:
        """Flush serial buffer."""
        old_timeout = self.visa_handle.timeout
        self.visa_handle.timeout = 200
        try:
            for _ in range(50):
                try:
                    self.visa_handle.read_raw()
                except:
                    break
        finally:
            self.visa_handle.timeout = old_timeout
    
    def _query(self, cmd: str) -> str:
        """Send command and get response."""
        time.sleep(0.3)
        self.visa_handle.write(cmd)
        time.sleep(0.3)
        raw = self.visa_handle.read_raw()
        return raw.decode('ascii', errors='ignore').strip()
    
    def _get_temperature(self) -> float:
        """Get temperature."""
        response = self._query('MEAS:TEMP?')
        if response.startswith('ERR'):
            raise RuntimeError(response)
        return float(response)
    
    def _get_humidity(self) -> float:
        """Get humidity."""
        response = self._query('MEAS:HUM?')
        if response.startswith('ERR'):
            raise RuntimeError(response)
        return float(response)
    
    def _get_averaging(self) -> int:
        """Get averaging count."""
        response = self._query('CONF:AVG?')
        return int(response)
    
    def _set_averaging(self, value: int) -> None:
        """Set averaging count."""
        response = self._query(f'CONF:AVG {value}')
        if response != 'OK':
            raise RuntimeError(response)
    
    def get_idn(self) -> Dict[str, Optional[str]]:
        """Get instrument ID."""
        response = self._query('*IDN?')
        parts = response.split(',')
        return {
            'vendor': parts[0] if len(parts) > 0 else None,
            'model': parts[1] if len(parts) > 1 else None,
            'serial': parts[2] if len(parts) > 2 else None,
            'firmware': parts[3] if len(parts) > 3 else None
        }
    
    def get_all(self) -> Tuple[float, float]:
        """Get temperature and humidity together."""
        response = self._query('MEAS:ALL?')
        # Parse "TEMP:24.00,HUM:31.00"
        parts = response.replace('TEMP:', '').replace('HUM:', '').split(',')
        return float(parts[0]), float(parts[1])
    
    def reset(self) -> None:
        """Reset to defaults."""
        self._query('*RST')