"""
QCoDeS Driver for Arduino DHT11 SCPI-like Instrument
=====================================================

This driver provides a professional interface to the Arduino-based
temperature and humidity sensor using SCPI-like commands.

Installation:
    pip install qcodes pyvisa pyvisa-py pyserial

Usage:
    from DHT11 import ArduinoDHT11
    
    # Connect to the instrument
    dht = ArduinoDHT11('dht11', 'ASRL3::INSTR', visalib='@py')
    
    # Query measurements
    temp = dht.temperature()
    hum = dht.humidity()

Author: Abhay's Lab @ GSU
Version: 1.1.0
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
    
    This instrument uses a SCPI-like command set over USB serial communication.
    
    Args:
        name: Instrument name for QCoDeS
        address: VISA resource string (e.g., 'ASRL3::INSTR')
        timeout: Communication timeout in seconds (default: 5)
        reset_delay: Time to wait for Arduino reset in seconds (default: 3.0)
        **kwargs: Additional arguments passed to VisaInstrument
    
    Example:
        >>> dht = ArduinoDHT11('dht11', 'ASRL3::INSTR', visalib='@py')
        >>> print(f"Temperature: {dht.temperature()} °C")
        >>> print(f"Humidity: {dht.humidity()} %")
    """
    
    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 5,
        reset_delay: float = 3.0,
        **kwargs: Any
    ) -> None:
        
        super().__init__(name, address, terminator='\n', **kwargs)
        
        # Configure serial communication
        self.visa_handle.baud_rate = 115200
        self.visa_handle.timeout = timeout * 1000  # Convert to ms
        
        # =====================================================================
        # ARDUINO RESET HANDLING
        # =====================================================================
        # Arduino Mega resets when serial port opens. The firmware now sends
        # "READY" when it's fully initialized. We wait for this message.
        
        logger.info("Waiting for Arduino to initialize...")
        
        # Wait for READY message from Arduino
        self._wait_for_ready(timeout=reset_delay + 5)
        
        # Verify connection by checking IDN
        idn = self.get_idn()
        if 'DHT11-SCPI' not in idn.get('model', ''):
            logger.warning(
                f"Unexpected instrument response: {idn}. "
                "Expected DHT11-SCPI firmware."
            )
        
        # =====================================================================
        # PARAMETERS
        # =====================================================================
        
        self.temperature: Parameter = self.add_parameter(
            'temperature',
            get_cmd='MEAS:TEMP?',
            get_parser=self._parse_float,
            unit='°C',
            label='Temperature',
            docstring='Current temperature reading in configured units'
        )
        
        self.humidity: Parameter = self.add_parameter(
            'humidity',
            get_cmd='MEAS:HUM?',
            get_parser=self._parse_float,
            unit='%',
            label='Relative Humidity',
            docstring='Current relative humidity reading'
        )
        
        self.unit: Parameter = self.add_parameter(
            'unit',
            get_cmd='CONF:UNIT?',
            set_cmd='CONF:UNIT {}',
            vals=Enum('C', 'F', 'K'),
            label='Temperature Unit',
            docstring='Temperature unit: C (Celsius), F (Fahrenheit), K (Kelvin)',
            set_parser=str.upper,
            post_set=self._update_temp_unit
        )
        
        self.averaging: Parameter = self.add_parameter(
            'averaging',
            get_cmd='CONF:AVG?',
            set_cmd='CONF:AVG {}',
            get_parser=int,
            vals=Ints(1, 16),
            label='Averaging Count',
            docstring='Number of samples to average (1-16)'
        )
        
        self.mode: Parameter = self.add_parameter(
            'mode',
            get_cmd='SYST:MODE?',
            set_cmd='SYST:MODE {}',
            vals=Enum('QUERY', 'STREAM'),
            label='Operating Mode',
            docstring='Operating mode: QUERY (on-demand) or STREAM (continuous)'
        )
        
        self.stream_interval: Parameter = self.add_parameter(
            'stream_interval',
            get_cmd='SYST:INTV?',
            set_cmd='SYST:INTV {}',
            get_parser=int,
            vals=Numbers(min_value=2000),
            unit='ms',
            label='Stream Interval',
            docstring='Streaming interval in milliseconds (minimum 2000)'
        )
        
        self.streaming: Parameter = self.add_parameter(
            'streaming',
            get_cmd='DATA:STREAM?',
            get_parser=lambda x: x.strip() == 'ON',
            label='Streaming Status',
            docstring='Whether streaming is currently active'
        )
        
        self.connect_message()
    
    # =========================================================================
    # BUFFER CLEARING - CRITICAL FOR ARDUINO RESET
    # =========================================================================
    
    def _clear_buffer_raw(self) -> None:
        """
        Clear any pending data in the serial buffer using RAW reads.
        
        This avoids UnicodeDecodeError on garbage bytes from Arduino reset.
        """
        old_timeout = self.visa_handle.timeout
        self.visa_handle.timeout = 200  # Short timeout for clearing
        
        cleared_bytes = 0
        try:
            for _ in range(20):  # Max 20 attempts
                try:
                    # Use read_raw() to avoid decode errors
                    chunk = self.visa_handle.read_raw()
                    cleared_bytes += len(chunk)
                    time.sleep(0.05)
                except Exception:
                    # Timeout = buffer empty
                    break
        finally:
            self.visa_handle.timeout = old_timeout
        
        if cleared_bytes > 0:
            logger.debug(f"Cleared {cleared_bytes} bytes from buffer")
    
    def _wait_for_ready(self, timeout: float = 10) -> None:
        """
        Wait for Arduino to send READY message after reset.
        
        This ensures all bootloader garbage is done and firmware is running.
        """
        old_timeout = self.visa_handle.timeout
        self.visa_handle.timeout = int(timeout * 1000)
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                try:
                    # Read raw to avoid decode errors on garbage
                    data = self.visa_handle.read_raw()
                    
                    # Try to decode - if it works, check for READY
                    try:
                        text = data.decode('ascii', errors='ignore').strip()
                        if 'READY' in text:
                            logger.info("Arduino ready")
                            return
                    except:
                        pass
                    
                except Exception:
                    # Timeout on read, keep waiting
                    time.sleep(0.1)
            
            # If we get here, no READY received - try anyway
            logger.warning("No READY message received, attempting connection anyway")
            self._clear_buffer_raw()
            
        finally:
            self.visa_handle.timeout = old_timeout
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        Query instrument identification.
        
        Returns:
            Dictionary with keys: vendor, model, serial, firmware
        """
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
        self.unit.get()
        self.averaging.get()
        self.mode.get()
    
    def get_all(self) -> Tuple[float, float]:
        """
        Query both temperature and humidity in a single command.
        
        Returns:
            Tuple of (temperature, humidity)
        """
        response = self.ask('MEAS:ALL?')
        
        match = re.match(r'TEMP:([\d.-]+),HUM:([\d.-]+)', response.strip())
        if match:
            return float(match.group(1)), float(match.group(2))
        
        if response.startswith('ERR:'):
            raise RuntimeError(f"Measurement error: {response}")
        
        raise ValueError(f"Unexpected response format: {response}")
    
    def get_error(self) -> Tuple[int, str]:
        """
        Query and clear the last error.
        
        Returns:
            Tuple of (error_code, error_message)
        """
        response = self.ask('SYST:ERR?')
        parts = response.strip().split(':', 1)
        
        code = int(parts[0])
        message = parts[1] if len(parts) > 1 else ''
        
        return code, message
    
    def start_streaming(self) -> None:
        """Start continuous data streaming."""
        if self.mode() != 'STREAM':
            raise RuntimeError("Set mode to STREAM first: dht.mode('STREAM')")
        
        response = self.ask('DATA:STREAM:START')
        if response.strip() != 'OK':
            raise RuntimeError(f"Failed to start streaming: {response}")
    
    def stop_streaming(self) -> None:
        """Stop continuous data streaming."""
        response = self.ask('DATA:STREAM:STOP')
        if response.strip() != 'OK':
            raise RuntimeError(f"Failed to stop streaming: {response}")
    
    def read_stream(
        self,
        timeout: float = 60,
        max_samples: Optional[int] = None
    ) -> Generator[Dict[str, float], None, None]:
        """
        Generator that yields streaming data.
        
        Args:
            timeout: Maximum time to wait for data in seconds
            max_samples: Maximum number of samples to yield (None for unlimited)
        
        Yields:
            Dictionary with keys: temperature, humidity, timestamp
        """
        samples = 0
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                break
            
            if max_samples is not None and samples >= max_samples:
                break
            
            try:
                line = self.visa_handle.read()
                
                if line.startswith('DATA:'):
                    match = re.match(
                        r'DATA:TEMP:([\d.-]+),HUM:([\d.-]+),TIME:(\d+)',
                        line.strip()
                    )
                    if match:
                        samples += 1
                        yield {
                            'temperature': float(match.group(1)),
                            'humidity': float(match.group(2)),
                            'timestamp': int(match.group(3))
                        }
                
            except Exception as e:
                if 'timeout' not in str(e).lower():
                    logger.warning(f"Stream read error: {e}")
    
    def wait_for_valid_reading(self, timeout: float = 10) -> bool:
        """
        Wait until a valid sensor reading is available.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if valid reading is available, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                _ = self.temperature()
                return True
            except RuntimeError:
                time.sleep(0.5)
        
        return False
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _parse_float(self, response: str) -> float:
        """Parse a float response, handling errors."""
        response = response.strip()
        
        if response.startswith('ERR:'):
            raise RuntimeError(f"Instrument error: {response}")
        
        try:
            return float(response)
        except ValueError:
            raise ValueError(f"Could not parse response as float: {response}")
    
    def _update_temp_unit(self, value: str) -> None:
        """Update the temperature parameter unit after changing unit setting."""
        unit_map = {'C': '°C', 'F': '°F', 'K': 'K'}
        self.temperature.unit = unit_map.get(value, '°C')


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def find_arduino_ports() -> list:
    """Find available Arduino serial ports."""
    import pyvisa
    rm = pyvisa.ResourceManager('@py')
    resources = rm.list_resources()
    return [r for r in resources if r.startswith('ASRL')]


def create_dht11_instrument(
    name: str = 'dht11',
    port: Optional[str] = None
) -> ArduinoDHT11:
    """
    Convenience function to create a DHT11 instrument.
    
    Args:
        name: Instrument name for QCoDeS
        port: Serial port (e.g., 'COM3' or 'ASRL3::INSTR')
    
    Returns:
        Configured ArduinoDHT11 instance
    """
    if port is None:
        ports = find_arduino_ports()
        if not ports:
            raise RuntimeError("No serial ports found.")
        port = ports[0]
        logger.info(f"Auto-detected port: {port}")
    
    # Convert simple port name to VISA resource string
    if not port.startswith('ASRL'):
        if port.upper().startswith('COM'):
            # COM3 -> ASRL3::INSTR
            num = port.upper().replace('COM', '')
            port = f'ASRL{num}::INSTR'
    
    return ArduinoDHT11(name, port, visalib='@py')