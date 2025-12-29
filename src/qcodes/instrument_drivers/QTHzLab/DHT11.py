"""
QCoDeS Driver for Arduino DHT11 Sensor
======================================

Production-grade driver for 24/7 operation.
Handles USB-serial garbage bytes issue.

Author: Abhay's Lab @ GSU
Version: 3.1.0
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
    Robust QCoDeS driver for Arduino DHT11 sensor.
    """
    
    MAX_RETRIES = 5
    COMMAND_DELAY = 0.8
    
    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 5,
        **kwargs: Any
    ) -> None:
        
        super().__init__(name, address, terminator='\n', **kwargs)
        
        self.visa_handle.baud_rate = 115200
        self.visa_handle.timeout = timeout * 1000
        
        # Statistics
        self._total_queries = 0
        self._failed_queries = 0
        self._retries = 0
        
        # Initialize
        self._initialize_connection()
        
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
    
    def _initialize_connection(self) -> None:
        """Initialize connection."""
        logger.info("Initializing Arduino connection...")
        
        # Wait for Arduino reset
        time.sleep(4)
        
        # Flush READY message
        self._flush()
        
        # CRITICAL: Send sacrificial command to absorb 0xf0 garbage bytes
        # These bytes are injected by USB-serial chip during connection
        logger.info("Sending sacrificial command...")
        time.sleep(0.3)
        self.visa_handle.write('SACRIFICE')
        time.sleep(0.3)
        self._flush()  # Discard error response
        
        # Now connection is clean - verify with IDN
        time.sleep(0.3)
        try:
            response = self._query_once('*IDN?')
            logger.info(f"Connected: {response}")
        except Exception as e:
            logger.warning(f"IDN check failed: {e}")
    
    def _flush(self) -> None:
        """Flush serial buffer."""
        old_timeout = self.visa_handle.timeout
        self.visa_handle.timeout = 200
        try:
            for _ in range(20):
                try:
                    self.visa_handle.read_raw()
                except:
                    break
        finally:
            self.visa_handle.timeout = old_timeout
    
    def _query_once(self, cmd: str) -> str:
        """Single query without retry."""
        time.sleep(self.COMMAND_DELAY)
        self._flush()  # Always flush before sending
        self.visa_handle.write(cmd)
        time.sleep(0.5)  # Increased delay
        raw = self.visa_handle.read_raw()
        return raw.decode('ascii', errors='ignore').strip()
    
    def _query(self, cmd: str) -> str:
        """Query with automatic retry."""
        self._total_queries += 1
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._query_once(cmd)
                
                # Check for corrupted command
                if 'ERR:100:Unknown command' in response:
                    self._retries += 1
                    logger.warning(f"Retry {attempt + 1}: {response}")
                    
                    # Clear Arduino's command buffer by sending newline
                    self.visa_handle.write('')  # Sends just \n
                    time.sleep(0.2)
                    self._flush()
                    time.sleep(0.5)
                    continue
                
                return response
                
            except Exception as e:
                self._retries += 1
                logger.warning(f"Retry {attempt + 1}: {e}")
                
                # Clear Arduino's command buffer
                self.visa_handle.write('')
                time.sleep(0.2)
                self._flush()
                time.sleep(0.5)
        
        self._failed_queries += 1
        raise RuntimeError(f"Failed after {self.MAX_RETRIES} attempts: {cmd}")
    
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
        """Get averaging."""
        return int(self._query('CONF:AVG?'))
    
    def _set_averaging(self, value: int) -> None:
        """Set averaging."""
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
        """Get temperature and humidity."""
        for attempt in range(self.MAX_RETRIES):
            response = self._query('MEAS:ALL?')
            
            # Validate format
            if response.startswith('TEMP:') and ',HUM:' in response:
                try:
                    parts = response.replace('TEMP:', '').replace('HUM:', '').split(',')
                    return float(parts[0]), float(parts[1])
                except (ValueError, IndexError):
                    pass
            
            logger.warning(f"Invalid response: {response}")
            time.sleep(0.5)
        
        raise RuntimeError("get_all failed")
    
    def reset(self) -> None:
        """Reset to defaults."""
        self._query('*RST')
    
    def get_statistics(self) -> Dict:
        """Get communication statistics."""
        rate = 0
        if self._total_queries > 0:
            rate = (self._total_queries - self._failed_queries) / self._total_queries * 100
        return {
            'total': self._total_queries,
            'failed': self._failed_queries,
            'retries': self._retries,
            'success_rate': f"{rate:.1f}%"
        }
    
    def print_statistics(self) -> None:
        """Print statistics."""
        s = self.get_statistics()
        print(f"Queries: {s['total']}, Failed: {s['failed']}, Retries: {s['retries']}, Success: {s['success_rate']}")