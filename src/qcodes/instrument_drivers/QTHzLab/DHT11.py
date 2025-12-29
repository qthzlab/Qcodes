"""
QCoDeS Driver for Arduino DHT11 SCPI-like Instrument
=====================================================

This driver provides a professional interface to the Arduino-based
temperature and humidity sensor using SCPI-like commands.

Installation:
    pip install qcodes pyvisa pyvisa-py pyserial

Usage:
    from dht11_qcodes_driver import ArduinoDHT11
    
    # Connect to the instrument
    dht = ArduinoDHT11('dht11', 'ASRL/dev/ttyACM0::INSTR')  # Linux
    # dht = ArduinoDHT11('dht11', 'ASRLCOM3::INSTR')        # Windows
    
    # Query measurements
    temp = dht.temperature()
    hum = dht.humidity()
    
    # Or get both at once
    temp, hum = dht.get_all()
    
    # Configure
    dht.unit('F')           # Fahrenheit
    dht.averaging(4)        # 4-point averaging
    
    # Streaming mode for continuous monitoring
    dht.mode('STREAM')
    dht.stream_interval(5000)  # 5 seconds
    dht.start_streaming()
    
    # Read stream data
    for data in dht.read_stream(timeout=30):
        print(f"T={data['temperature']}, H={data['humidity']}")

Author: Abhay's Lab @ GSU
Version: 1.0.0
"""

import re
import time
import logging
from typing import Optional, Tuple, Dict, Generator, Any

from qcodes import Instrument, VisaInstrument
from qcodes.parameters import Parameter
from qcodes.validators import Enum, Numbers, Ints

logger = logging.getLogger(__name__)


class ArduinoDHT11(VisaInstrument):
    """
    QCoDeS driver for Arduino Mega + DHT11 temperature/humidity sensor.
    
    This instrument uses a SCPI-like command set over USB serial communication.
    
    Args:
        name: Instrument name for QCoDeS
        address: VISA resource string (e.g., 'ASRL/dev/ttyACM0::INSTR')
        timeout: Communication timeout in seconds (default: 5)
        **kwargs: Additional arguments passed to VisaInstrument
    
    Attributes:
        temperature: Current temperature reading (in configured units)
        humidity: Current relative humidity reading (%)
        unit: Temperature unit ('C', 'F', or 'K')
        averaging: Number of samples to average (1-16)
        mode: Operating mode ('QUERY' or 'STREAM')
        stream_interval: Streaming interval in milliseconds
        streaming: Whether streaming is currently active
    
    Example:
        >>> dht = ArduinoDHT11('dht11', 'ASRL/dev/ttyACM0::INSTR')
        >>> print(f"Temperature: {dht.temperature()} °C")
        >>> print(f"Humidity: {dht.humidity()} %")
    """
    
    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 5,
        **kwargs: Any
    ) -> None:
        
        super().__init__(name, address, terminator='\n', **kwargs)
        
        # Configure serial communication
        self.visa_handle.baud_rate = 115200
        self.visa_handle.timeout = timeout * 1000  # Convert to ms
        
        # Clear any pending data
        self._clear_buffer()
        
        # small delay before first command
        time.sleep(0.1)
        
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
            unit='°C',  # Updated dynamically based on unit setting
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
        
        # Connect to the instrument and initialize
        self.connect_message()
    
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
        # Update cached parameter values
        self.unit.get()
        self.averaging.get()
        self.mode.get()
    
    def get_all(self) -> Tuple[float, float]:
        """
        Query both temperature and humidity in a single command.
        
        Returns:
            Tuple of (temperature, humidity)
        
        Example:
            >>> temp, hum = dht.get_all()
            >>> print(f"T={temp}, H={hum}")
        """
        response = self.ask('MEAS:ALL?')
        
        # Parse "TEMP:23.50,HUM:45.00"
        match = re.match(r'TEMP:([\d.-]+),HUM:([\d.-]+)', response.strip())
        if match:
            return float(match.group(1)), float(match.group(2))
        
        # Check for error
        if response.startswith('ERR:'):
            raise RuntimeError(f"Measurement error: {response}")
        
        raise ValueError(f"Unexpected response format: {response}")
    
    def get_error(self) -> Tuple[int, str]:
        """
        Query and clear the last error.
        
        Returns:
            Tuple of (error_code, error_message)
            Error code 0 means no error.
        """
        response = self.ask('SYST:ERR?')
        parts = response.strip().split(':', 1)
        
        code = int(parts[0])
        message = parts[1] if len(parts) > 1 else ''
        
        return code, message
    
    def start_streaming(self) -> None:
        """
        Start continuous data streaming.
        
        The instrument must be in STREAM mode first.
        Use read_stream() to receive the data.
        """
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
        
        Example:
            >>> dht.mode('STREAM')
            >>> dht.start_streaming()
            >>> for data in dht.read_stream(timeout=30, max_samples=10):
            ...     print(f"T={data['temperature']}, H={data['humidity']}")
            >>> dht.stop_streaming()
        """
        samples = 0
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                logger.info("Stream read timeout reached")
                break
            
            # Check sample limit
            if max_samples is not None and samples >= max_samples:
                logger.info(f"Reached maximum samples: {max_samples}")
                break
            
            try:
                # Read with short timeout to stay responsive
                line = self.visa_handle.read()
                
                # Parse stream data: "DATA:TEMP:23.50,HUM:45.00,TIME:123456"
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
                # Timeout or other error - continue trying
                if 'timeout' not in str(e).lower():
                    logger.warning(f"Stream read error: {e}")
    
    def wait_for_valid_reading(self, timeout: float = 10) -> bool:
        """
        Wait until a valid sensor reading is available.
        
        The DHT11 needs time after power-on to provide valid readings.
        
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
    
    def _clear_buffer(self) -> None:
        """Clear any pending data in the serial buffer."""
        try:
            # Set short timeout for clearing
            old_timeout = self.visa_handle.timeout
            self.visa_handle.timeout = 100  # 100 ms
            
            while True:
                try:
                    self.visa_handle.read()
                except:
                    break
            
            self.visa_handle.timeout = old_timeout
        except:
            pass
    
    def _parse_float(self, response: str) -> float:
        """
        Parse a float response, handling errors.
        
        Args:
            response: Raw response string from instrument
        
        Returns:
            Parsed float value
        
        Raises:
            RuntimeError: If response indicates an error
            ValueError: If response cannot be parsed
        """
        response = response.strip()
        
        # Check for error response
        if response.startswith('ERR:'):
            raise RuntimeError(f"Instrument error: {response}")
        
        try:
            return float(response)
        except ValueError:
            raise ValueError(f"Could not parse response as float: {response}")
    
    def _update_temp_unit(self, value: str) -> None:
        """Update the temperature parameter unit after changing unit setting."""
        unit_map = {
            'C': '°C',
            'F': '°F',
            'K': 'K'
        }
        self.temperature.unit = unit_map.get(value, '°C')


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def find_arduino_ports() -> list:
    """
    Find available Arduino serial ports.
    
    Returns:
        List of VISA resource strings for potential Arduino devices
    """
    import pyvisa
    rm = pyvisa.ResourceManager('@py')
    resources = rm.list_resources()
    
    # Filter for serial ports
    serial_ports = [r for r in resources if r.startswith('ASRL')]
    
    return serial_ports


def create_dht11_instrument(
    name: str = 'dht11',
    port: Optional[str] = None
) -> ArduinoDHT11:
    """
    Convenience function to create a DHT11 instrument.
    
    If port is not specified, attempts to find an Arduino automatically.
    
    Args:
        name: Instrument name for QCoDeS
        port: Serial port (e.g., '/dev/ttyACM0' or 'COM3')
               If None, attempts auto-detection
    
    Returns:
        Configured ArduinoDHT11 instance
    
    Example:
        >>> dht = create_dht11_instrument()
        >>> print(dht.temperature())
    """
    if port is None:
        ports = find_arduino_ports()
        if not ports:
            raise RuntimeError(
                "No serial ports found. Please specify port manually."
            )
        # Try each port
        for p in ports:
            try:
                dht = ArduinoDHT11(name, p)
                logger.info(f"Connected to DHT11 on {p}")
                return dht
            except Exception as e:
                logger.debug(f"Failed to connect to {p}: {e}")
                continue
        
        raise RuntimeError(
            f"Could not connect to DHT11 on any port: {ports}"
        )
    
    # Convert simple port name to VISA resource string
    if not port.startswith('ASRL'):
        if port.startswith('/dev/'):
            # Linux
            port = f'ASRL{port}::INSTR'
        elif port.upper().startswith('COM'):
            # Windows
            port = f'ASRL{port}::INSTR'
    
    return ArduinoDHT11(name, port)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    """Example demonstrating basic usage of the DHT11 driver."""
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Arduino DHT11 QCoDeS Driver - Example")
    print("=" * 60)
    
    # Try to find and connect to the instrument
    try:
        dht = create_dht11_instrument('dht11')
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nTo connect manually, use:")
        print("  dht = ArduinoDHT11('dht11', 'ASRL/dev/ttyACM0::INSTR')  # Linux")
        print("  dht = ArduinoDHT11('dht11', 'ASRLCOM3::INSTR')          # Windows")
        exit(1)
    
    # Display identification
    idn = dht.get_idn()
    print(f"\nConnected to: {idn['vendor']} {idn['model']}")
    print(f"Serial: {idn['serial']}, Firmware: {idn['firmware']}")
    
    # Wait for valid reading
    print("\nWaiting for valid sensor reading...")
    if not dht.wait_for_valid_reading(timeout=10):
        print("Warning: Could not get valid reading within timeout")
    
    # Read measurements
    print("\n--- Current Readings ---")
    try:
        temp, hum = dht.get_all()
        print(f"Temperature: {temp:.1f} °C")
        print(f"Humidity: {hum:.1f} %")
    except RuntimeError as e:
        print(f"Measurement error: {e}")
    
    # Show configuration
    print("\n--- Configuration ---")
    print(f"Unit: {dht.unit()}")
    print(f"Averaging: {dht.averaging()}")
    print(f"Mode: {dht.mode()}")
    print(f"Stream Interval: {dht.stream_interval()} ms")
    
    # Example: Change unit to Fahrenheit
    print("\n--- Changing to Fahrenheit ---")
    dht.unit('F')
    temp = dht.temperature()
    print(f"Temperature: {temp:.1f} °F")
    
    # Reset to defaults
    dht.reset()
    
    # Close connection
    dht.close()
    print("\nConnection closed.")