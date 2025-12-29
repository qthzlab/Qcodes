"""
QCoDeS Driver for Arduino DHT11 SCPI-like Instrument
=====================================================

Version: 1.4.0 - Overrides ask() to handle Arduino garbage bytes

Author: Abhay's Lab @ GSU
"""

import re
import time
import logging
from typing import Optional, Tuple, Dict, Generator, Any

from qcodes import VisaInstrument
from qcodes.parameters import Parameter
from qcodes.validators import Enum, Numbers, Ints

logger = logging.getLogger(__name__)


class ArduinoDHT11(VisaInstrument):
    """
    QCoDeS driver for Arduino Mega + DHT11 temperature/humidity sensor.
    
    This driver overrides the standard ask() method to handle
    garbage bytes from Arduino bootloader reset.
    """
    
    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 5,
        reset_delay: float = 5.0,
        **kwargs: Any
    ) -> None:
        
        super().__init__(name, address, terminator='\n', **kwargs)
        
        # Configure serial communication
        self.visa_handle.baud_rate = 115200
        self.visa_handle.timeout = timeout * 1000
        
        # =====================================================================
        # ARDUINO RESET HANDLING
        # =====================================================================
        logger.info(f"Waiting {reset_delay}s for Arduino to initialize...")
        
        # Just wait for Arduino to fully boot
        time.sleep(reset_delay)
        
        # Flush everything using raw reads
        self._flush_buffer_aggressive()
        
        # Extra wait
        time.sleep(0.5)
        
        # Test connection using our custom ask
        try:
            idn_response = self.ask('*IDN?')
            logger.info(f"Connected: {idn_response}")
        except Exception as e:
            logger.warning(f"Initial IDN failed: {e}, but continuing...")
        
        # =====================================================================
        # PARAMETERS
        # =====================================================================
        
        self.temperature: Parameter = self.add_parameter(
            'temperature',
            get_cmd='MEAS:TEMP?',
            get_parser=self._parse_float,
            unit='°C',
            label='Temperature'
        )
        
        self.humidity: Parameter = self.add_parameter(
            'humidity',
            get_cmd='MEAS:HUM?',
            get_parser=self._parse_float,
            unit='%',
            label='Relative Humidity'
        )
        
        self.unit: Parameter = self.add_parameter(
            'unit',
            get_cmd='CONF:UNIT?',
            set_cmd='CONF:UNIT {}',
            vals=Enum('C', 'F', 'K'),
            label='Temperature Unit',
            set_parser=str.upper
        )
        
        self.averaging: Parameter = self.add_parameter(
            'averaging',
            get_cmd='CONF:AVG?',
            set_cmd='CONF:AVG {}',
            get_parser=int,
            vals=Ints(1, 16),
            label='Averaging Count'
        )
        
        self.mode: Parameter = self.add_parameter(
            'mode',
            get_cmd='SYST:MODE?',
            set_cmd='SYST:MODE {}',
            vals=Enum('QUERY', 'STREAM'),
            label='Operating Mode'
        )
        
        self.stream_interval: Parameter = self.add_parameter(
            'stream_interval',
            get_cmd='SYST:INTV?',
            set_cmd='SYST:INTV {}',
            get_parser=int,
            vals=Numbers(min_value=2000),
            unit='ms',
            label='Stream Interval'
        )
        
        self.streaming: Parameter = self.add_parameter(
            'streaming',
            get_cmd='DATA:STREAM?',
            get_parser=lambda x: x.strip() == 'ON',
            label='Streaming Status'
        )
        
        # Skip connect_message - it calls get_idn which might fail
        # self.connect_message()
        logger.info(f"ArduinoDHT11 '{name}' initialized")
    
    # =========================================================================
    # OVERRIDE ask() TO HANDLE GARBAGE BYTES
    # =========================================================================
    
    def ask(self, cmd: str) -> str:
        """
        Override ask() to use raw reads and handle garbage bytes.
        """
        # Write command
        self.visa_handle.write(cmd)
        time.sleep(0.1)  # Give Arduino time to respond
        
        # Read response using raw method
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_data = self.visa_handle.read_raw()
                
                # Decode ignoring bad bytes
                response = raw_data.decode('ascii', errors='ignore').strip()
                
                # Remove any null characters or other garbage
                response = ''.join(c for c in response if c.isprintable() or c in '\n\r')
                response = response.strip()
                
                if response:
                    return response
                    
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.debug(f"Read attempt {attempt + 1} failed: {e}")
                    time.sleep(0.2)
                else:
                    raise
        
        raise RuntimeError(f"No valid response for command: {cmd}")
    
    def ask_raw(self, cmd: str) -> str:
        """Override ask_raw to use our custom ask."""
        return self.ask(cmd)
    
    def write(self, cmd: str) -> None:
        """Write a command without expecting a response."""
        self.visa_handle.write(cmd)
    
    # =========================================================================
    # BUFFER MANAGEMENT
    # =========================================================================
    
    def _flush_buffer_aggressive(self) -> None:
        """Aggressively flush all data from serial buffer."""
        old_timeout = self.visa_handle.timeout
        self.visa_handle.timeout = 100
        
        total_flushed = 0
        try:
            for _ in range(100):  # Up to 100 attempts
                try:
                    data = self.visa_handle.read_raw()
                    total_flushed += len(data)
                    
                    # Log what we're flushing (decoded)
                    text = data.decode('ascii', errors='replace')
                    logger.debug(f"Flushed: {repr(text)}")
                except:
                    break
        finally:
            self.visa_handle.timeout = old_timeout
        
        if total_flushed > 0:
            logger.info(f"Flushed {total_flushed} bytes from buffer")
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def get_idn(self) -> Dict[str, Optional[str]]:
        """Query instrument identification."""
        response = self.ask('*IDN?')
        parts = response.strip().split(',')
        
        return {
            'vendor': parts[0] if len(parts) > 0 else None,
            'model': parts[1] if len(parts) > 1 else None,
            'serial': parts[2] if len(parts) > 2 else None,
            'firmware': parts[3] if len(parts) > 3 else None
        }
    
    def reset(self) -> None:
        """Reset instrument to default settings."""
        response = self.ask('*RST')
        if response.strip() != 'OK':
            raise RuntimeError(f"Reset failed: {response}")
    
    def get_all(self) -> Tuple[float, float]:
        """Query both temperature and humidity."""
        response = self.ask('MEAS:ALL?')
        
        match = re.match(r'TEMP:([\d.-]+),HUM:([\d.-]+)', response.strip())
        if match:
            return float(match.group(1)), float(match.group(2))
        
        raise ValueError(f"Unexpected response: {response}")
    
    def get_error(self) -> Tuple[int, str]:
        """Query and clear the last error."""
        response = self.ask('SYST:ERR?')
        parts = response.strip().split(':', 1)
        code = int(parts[0])
        message = parts[1] if len(parts) > 1 else ''
        return code, message
    
    def start_streaming(self) -> None:
        """Start continuous data streaming."""
        if self.mode() != 'STREAM':
            raise RuntimeError("Set mode to STREAM first")
        response = self.ask('DATA:STREAM:START')
        if response.strip() != 'OK':
            raise RuntimeError(f"Failed: {response}")
    
    def stop_streaming(self) -> None:
        """Stop continuous data streaming."""
        response = self.ask('DATA:STREAM:STOP')
        if response.strip() != 'OK':
            raise RuntimeError(f"Failed: {response}")
    
    def wait_for_valid_reading(self, timeout: float = 10) -> bool:
        """Wait until a valid sensor reading is available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                _ = self.temperature()
                return True
            except:
                time.sleep(0.5)
        return False
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _parse_float(self, response: str) -> float:
        """Parse a float response."""
        response = response.strip()
        if response.startswith('ERR:'):
            raise RuntimeError(f"Instrument error: {response}")
        return float(response)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def find_arduino_ports() -> list:
    """Find available serial ports."""
    import pyvisa
    rm = pyvisa.ResourceManager('@py')
    return [r for r in rm.list_resources() if r.startswith('ASRL')]


def create_dht11_instrument(name: str = 'dht11', port: Optional[str] = None) -> ArduinoDHT11:
    """Create a DHT11 instrument."""
    if port is None:
        ports = find_arduino_ports()
        if not ports:
            raise RuntimeError("No serial ports found.")
        port = ports[0]
    
    if not port.startswith('ASRL'):
        if port.upper().startswith('COM'):
            num = port.upper().replace('COM', '')
            port = f'ASRL{num}::INSTR'
    
    return ArduinoDHT11(name, port, visalib='@py')