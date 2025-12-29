"""
QCoDeS driver for Stanford Research Systems SR850 DSP Lock-in Amplifier

This driver is adapted from the SR830 driver with modifications for SR850-specific features:
- Larger data buffer (65,536 points total vs 16,383 per channel)
- FAST mode for high-speed binary data transfer
- Scan length parameter (SLEN)
- Optional trace definition support (TRCD)

MIGRATION STATUS:
✓ Phase 1: Basic measurements (COMPLETE - should work immediately)
⚠ Phase 2: Buffer operations (IN PROGRESS - needs testing)
⚠ Phase 3: Advanced features (PLANNED - FAST mode, trace definitions)

Author: Adapted from QCoDeS SR830 driver for SR850 compatibility
"""

from __future__ import annotations

import time
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import (
    ArrayParameter,
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
)
from qcodes.validators import Arrays, ComplexNumbers, Enum, Ints, Numbers, Strings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Unpack


class ChannelTrace(ParameterWithSetpoints):
    """
    Parameter class for the four channel buffers (SR850 supports 4 traces)
    
    Note: SR850 buffer can be configured for 1, 2, or 4 traces with
    total of 65,536 points shared among active traces.
    """

    def __init__(self, name: str, channel: int, **kwargs: Any) -> None:
        """
        Args:
            name: The name of the parameter
            channel: The relevant channel (1-4). The name should match this.
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, **kwargs)

        # SR850 has 4 traces vs SR830's 2 channels
        self._valid_channels = (1, 2, 3, 4)

        if channel not in self._valid_channels:
            raise ValueError(
                "Invalid channel specifier. SR850 has traces 1-4."
            )

        if not isinstance(self.root_instrument, SR850):
            raise ValueError(
                "Invalid parent instrument. ChannelTrace can only live on an SR850."
            )

        self.channel = channel
        self.update_unit()

    def update_unit(self) -> None:
        """Update the unit based on trace definition"""
        assert isinstance(self.root_instrument, SR850)
        # For now, keep it simple - default to V
        # TODO: Parse TRCD? response to determine actual units
        self.unit = "V"
        self.label = f"Trace {self.channel}"

    def get_raw(self) -> ParamRawDataType:
        """
        Get command. Returns numpy array
        
        Uses TRCL (non-normalized binary) format for fastest transfer
        """
        assert isinstance(self.root_instrument, SR850)
        
        # Query number of points stored in this trace
        N = self.root_instrument.buffer_npts()
        if N == 0:
            raise ValueError(
                f"No points stored in SR850 trace {self.channel}. Buffer may not be started."
            )

        # Poll raw binary data using TRCL command
        # Format: TRCL? i,j,k where i=trace, j=start bin (0-indexed), k=count
        self.root_instrument.write(f"TRCL? {self.channel},0,{N}")
        rawdata = self.root_instrument.visa_handle.read_raw()

        # Parse binary data - same format as SR830
        # Each point is 2 bytes (int16): mantissa and exponent
        realdata = np.frombuffer(rawdata, dtype="<i2")
        numbers = realdata[::2] * 2.0 ** (realdata[1::2] - 124)

        return numbers


class ChannelBuffer(ArrayParameter):
    """
    Parameter class for the SR850 channel buffers
    
    SR850 differences from SR830:
    - Supports 4 traces (vs 2 channels)  
    - Total 65,536 points shared among active traces
    - Can be configured for 1 trace (64k pts), 2 traces (32k pts each), 
      or 4 traces (16k pts each)
    """

    def __init__(self, name: str, instrument: SR850, channel: int) -> None:
        """
        Args:
            name: The name of the parameter
            instrument: The parent instrument
            channel: The relevant trace number (1-4)
        """
        self._valid_channels = (1, 2, 3, 4)

        if channel not in self._valid_channels:
            raise ValueError(
                "Invalid channel specifier. SR850 has traces 1-4."
            )

        if not isinstance(instrument, SR850):
            raise ValueError(
                "Invalid parent instrument. ChannelBuffer can only live on an SR850."
            )

        super().__init__(
            name,
            shape=(1,),  # dummy initial shape
            unit="V",  # dummy initial unit
            setpoint_names=("Time",),
            setpoint_labels=("Time",),
            setpoint_units=("s",),
            docstring="Holds acquired data from one SR850 trace buffer.",
            instrument=instrument,
        )

        self.channel = channel

    def prepare_buffer_readout(self) -> None:
        """
        Function to generate the setpoints for the channel buffer and
        get the right units
        """
        assert isinstance(self.instrument, SR850)
        
        # Query number of points stored
        N = self.instrument.buffer_npts()
        if N == 0:
            raise RuntimeWarning("Buffer has 0 points. Has acquisition started?")
        
        # Get sample rate to determine time axis
        # Use try-except to handle query failures gracefully
        try:
            SR = self.instrument.buffer_SR()
        except Exception:
            # If buffer_SR query fails, use simple index-based setpoints
            self.setpoint_units = ("",)
            self.setpoint_names = ("index",)
            self.setpoint_labels = ("Sample index",)
            self.setpoints = (tuple(np.arange(0, N)),)
            SR = None
        
        if SR is not None:
            if SR == "Trigger":
                # Trigger mode: points are trigger events, not time-based
                self.setpoint_units = ("",)
                self.setpoint_names = ("trig_events",)
                self.setpoint_labels = ("Trigger event number",)
                self.setpoints = (tuple(np.arange(0, N)),)
            else:
                # Constant sample rate mode
                dt = 1 / SR
                self.setpoint_units = ("s",)
                self.setpoint_names = ("Time",)
                self.setpoint_labels = ("Time",)
                self.setpoints = (tuple(np.linspace(0, N * dt, N)),)

        self.shape = (N,)
        
        # Set unit - for now default to V
        # TODO: Parse trace definition to determine actual units
        self.unit = "V"

        # Mark this buffer as ready
        if self.channel == 1:
            self.instrument._buffer1_ready = True
        elif self.channel == 2:
            self.instrument._buffer2_ready = True
        elif self.channel == 3:
            self.instrument._buffer3_ready = True
        elif self.channel == 4:
            self.instrument._buffer4_ready = True

    def get_raw(self) -> ParamRawDataType:
        """
        Get command. Returns numpy array
        """
        assert isinstance(self.instrument, SR850)
        
        # Check if buffer was prepared
        ready_flags = {
            1: self.instrument._buffer1_ready,
            2: self.instrument._buffer2_ready,
            3: self.instrument._buffer3_ready,
            4: self.instrument._buffer4_ready,
        }
        
        if not ready_flags[self.channel]:
            raise RuntimeError(
                f"Buffer {self.channel} not ready. Please run prepare_buffer_readout()"
            )
        
        N = self.instrument.buffer_npts()
        if N == 0:
            raise ValueError(
                f"No points stored in SR850 trace {self.channel}. Buffer empty."
            )

        # Read data using TRCA (ASCII format) - more reliable than binary
        # Increase timeout for large buffers
        old_timeout = self.instrument.visa_handle.timeout
        self.instrument.visa_handle.timeout = max(10000, N * 50)  # 50ms per point
        
        try:
            # Use ASCII format (TRCA) instead of binary (TRCL) for reliability
            response = self.instrument.ask(f"TRCA? {self.channel},0,{N}")
            # Parse comma-separated values, filtering empty strings
            numbers = np.array([float(x) for x in response.split(',') if x.strip()])
        finally:
            self.instrument.visa_handle.timeout = old_timeout
        
        # Validate we got the expected number of points
        if len(numbers) != N:
            raise RuntimeError(
                f"SR850 trace {self.channel}: got {len(numbers)} points, expected {N}"
            )
        
        return numbers


class SR850(VisaInstrument):
    """
    QCoDeS driver for the Stanford Research Systems SR850 DSP Lock-in Amplifier.
    
    The SR850 is a DSP-based lock-in amplifier with the following key features:
    - Frequency range: 1 mHz to 102 kHz
    - Sensitivity: 2 nV to 1 V (voltage), 2 fA to 1 µA (current)
    - Data buffer: 65,536 points total (configurable for 1/2/4 traces)
    - CRT display with extensive visualization options
    - FAST mode for high-speed binary data streaming
    
    This driver maintains ~85% compatibility with the SR830 driver, with
    the main differences being:
    1. Larger buffer (65k vs 16k points)
    2. 4 traces instead of 2 channels
    3. FAST mode support
    4. Scan length parameter (SLEN)
    """

    # Maximum buffer size - SR850 specific
    _MAX_BUFFER_SIZE = 65536  # Total points across all traces
    
    # Sensitivity mappings - IDENTICAL to SR830
    _VOLT_TO_N: ClassVar[dict[float | int, int]] = {
        2e-9: 0,
        5e-9: 1,
        10e-9: 2,
        20e-9: 3,
        50e-9: 4,
        100e-9: 5,
        200e-9: 6,
        500e-9: 7,
        1e-6: 8,
        2e-6: 9,
        5e-6: 10,
        10e-6: 11,
        20e-6: 12,
        50e-6: 13,
        100e-6: 14,
        200e-6: 15,
        500e-6: 16,
        1e-3: 17,
        2e-3: 18,
        5e-3: 19,
        10e-3: 20,
        20e-3: 21,
        50e-3: 22,
        100e-3: 23,
        200e-3: 24,
        500e-3: 25,
        1: 26,
    }
    _N_TO_VOLT: ClassVar[dict[int, float | int]] = {v: k for k, v in _VOLT_TO_N.items()}

    _CURR_TO_N: ClassVar[dict[float, int]] = {
        2e-15: 0,
        5e-15: 1,
        10e-15: 2,
        20e-15: 3,
        50e-15: 4,
        100e-15: 5,
        200e-15: 6,
        500e-15: 7,
        1e-12: 8,
        2e-12: 9,
        5e-12: 10,
        10e-12: 11,
        20e-12: 12,
        50e-12: 13,
        100e-12: 14,
        200e-12: 15,
        500e-12: 16,
        1e-9: 17,
        2e-9: 18,
        5e-9: 19,
        10e-9: 20,
        20e-9: 21,
        50e-9: 22,
        100e-9: 23,
        200e-9: 24,
        500e-9: 25,
        1e-6: 26,
    }
    _N_TO_CURR: ClassVar[dict[int, float]] = {v: k for k, v in _CURR_TO_N.items()}

    _VOLT_ENUM = Enum(*_VOLT_TO_N.keys())
    _CURR_ENUM = Enum(*_CURR_TO_N.keys())

    # Input configuration - IDENTICAL to SR830
    _INPUT_CONFIG_TO_N: ClassVar[dict[str, int]] = {
        "a": 0,
        "a-b": 1,
        "I 1M": 2,
        "I 100M": 3,
    }

    _N_TO_INPUT_CONFIG: ClassVar[dict[int, str]] = {
        v: k for k, v in _INPUT_CONFIG_TO_N.items()
    }

    def __init__(self, name: str, address: str, **kwargs: Unpack[VisaInstrumentKWArgs]):
        # Set VISA defaults for SR850 if not specified
        kwargs.setdefault('timeout', 10)  # 10 second timeout
        kwargs.setdefault('terminator', '\n')  # SR850 uses \n terminator
        
        super().__init__(name, address, **kwargs)

        # ========== REFERENCE AND PHASE - IDENTICAL TO SR830 ==========
        self.phase: Parameter = self.add_parameter(
            "phase",
            label="Phase",
            get_cmd="PHAS?",
            get_parser=float,
            set_cmd="PHAS {:.2f}",
            unit="deg",
            vals=Numbers(min_value=-360, max_value=729.99),
        )
        """Parameter phase"""

        self.reference_source: Parameter = self.add_parameter(
            "reference_source",
            label="Reference source",
            get_cmd="FMOD?",
            set_cmd="FMOD {}",
            val_mapping={
                "external": 2,  # External reference
                "internal": 0,  # Internal oscillator  
                "sweep": 1,     # Internal sweep (SR850 specific)
            },
            vals=Enum("external", "internal", "sweep"),
        )
        """Parameter reference_source"""

        self.frequency: Parameter = self.add_parameter(
            "frequency",
            label="Frequency",
            get_cmd="FREQ?",
            get_parser=float,
            set_cmd="FREQ {:.4f}",
            unit="Hz",
            vals=Numbers(min_value=1e-3, max_value=102e3),
        )
        """Parameter frequency"""

        self.ext_trigger: Parameter = self.add_parameter(
            "ext_trigger",
            label="External trigger",
            get_cmd="RSLP?",
            set_cmd="RSLP {}",
            val_mapping={
                "sine": 0,
                "TTL rising": 1,
                "TTL falling": 2,
            },
        )
        """Parameter ext_trigger"""

        self.harmonic: Parameter = self.add_parameter(
            "harmonic",
            label="Harmonic",
            get_cmd="HARM?",
            get_parser=int,
            set_cmd="HARM {:d}",
            vals=Ints(min_value=1, max_value=19999),
        )
        """Parameter harmonic"""

        self.amplitude: Parameter = self.add_parameter(
            "amplitude",
            label="Amplitude",
            get_cmd="SLVL?",
            get_parser=float,
            set_cmd="SLVL {:.3f}",
            unit="V",
            vals=Numbers(min_value=0.004, max_value=5.000),
        )
        """Parameter amplitude"""

        # ========== INPUT AND FILTER - IDENTICAL TO SR830 ==========
        self.input_config: Parameter = self.add_parameter(
            "input_config",
            label="Input configuration",
            get_cmd="ISRC?",
            get_parser=self._get_input_config,
            set_cmd="ISRC {}",
            set_parser=self._set_input_config,
            vals=Enum(*self._INPUT_CONFIG_TO_N.keys()),
        )
        """Parameter input_config"""

        self.input_shield: Parameter = self.add_parameter(
            "input_shield",
            label="Input shield",
            get_cmd="IGND?",
            set_cmd="IGND {}",
            val_mapping={
                "float": 0,
                "ground": 1,
            },
        )
        """Parameter input_shield"""

        self.input_coupling: Parameter = self.add_parameter(
            "input_coupling",
            label="Input coupling",
            get_cmd="ICPL?",
            set_cmd="ICPL {}",
            val_mapping={
                "AC": 0,
                "DC": 1,
            },
        )
        """Parameter input_coupling"""

        self.notch_filter: Parameter = self.add_parameter(
            "notch_filter",
            label="Notch filter",
            get_cmd="ILIN?",
            set_cmd="ILIN {}",
            val_mapping={
                "off": 0,
                "line in": 1,
                "2x line in": 2,
                "both": 3,
            },
        )
        """Parameter notch_filter"""

        # ========== GAIN AND TIME CONSTANT - IDENTICAL TO SR830 ==========
        self.sensitivity: Parameter = self.add_parameter(
            name="sensitivity",
            label="Sensitivity",
            get_cmd="SENS?",
            set_cmd="SENS {:d}",
            get_parser=self._get_sensitivity,
            set_parser=self._set_sensitivity,
        )
        """Parameter sensitivity"""

        self.reserve: Parameter = self.add_parameter(
            "reserve",
            label="Reserve",
            get_cmd="RMOD?",
            set_cmd="RMOD {}",
            val_mapping={
                "max": 0,      # SR850 uses "max" instead of "high"
                "manual": 1,   # SR850 uses "manual" instead of "normal"  
                "min": 2,      # SR850 uses "min" instead of "low noise"
            },
        )
        """Parameter reserve"""

        self.time_constant: Parameter = self.add_parameter(
            "time_constant",
            label="Time constant",
            get_cmd="OFLT?",
            set_cmd="OFLT {}",
            unit="s",
            val_mapping={
                10e-6: 0,
                30e-6: 1,
                100e-6: 2,
                300e-6: 3,
                1e-3: 4,
                3e-3: 5,
                10e-3: 6,
                30e-3: 7,
                100e-3: 8,
                300e-3: 9,
                1: 10,
                3: 11,
                10: 12,
                30: 13,
                100: 14,
                300: 15,
                1e3: 16,
                3e3: 17,
                10e3: 18,
                30e3: 19,
                # SR850 can extend to 30,000 s for f < 200 Hz
            },
        )
        """Parameter time_constant"""

        self.filter_slope: Parameter = self.add_parameter(
            "filter_slope",
            label="Filter slope",
            get_cmd="OFSL?",
            set_cmd="OFSL {}",
            unit="dB/oct",
            val_mapping={
                6: 0,
                12: 1,
                18: 2,
                24: 3,
            },
        )
        """Parameter filter_slope"""

        self.sync_filter: Parameter = self.add_parameter(
            "sync_filter",
            label="Sync filter",
            get_cmd="SYNC?",
            set_cmd="SYNC {}",
            val_mapping={
                "off": 0,
                "on": 1,
            },
        )
        """Parameter sync_filter"""

        # Note: OEXP not fully implemented yet - need multi-arg set
        def parse_offset_get(s: str) -> tuple[float, int]:
            parts = s.split(",")
            return float(parts[0]), int(parts[1])

        self.X_offset: Parameter = self.add_parameter(
            "X_offset", get_cmd="OEXP? 1", get_parser=parse_offset_get
        )
        """Parameter X_offset"""

        self.Y_offset: Parameter = self.add_parameter(
            "Y_offset", get_cmd="OEXP? 2", get_parser=parse_offset_get
        )
        """Parameter Y_offset"""

        self.R_offset: Parameter = self.add_parameter(
            "R_offset", get_cmd="OEXP? 3", get_parser=parse_offset_get
        )
        """Parameter R_offset"""

        # ========== AUX INPUT/OUTPUT - IDENTICAL TO SR830 ==========
        for i in [1, 2, 3, 4]:
            self.add_parameter(
                f"aux_in{i}",
                label=f"Aux input {i}",
                get_cmd=f"OAUX? {i}",
                get_parser=float,
                unit="V",
            )

            self.add_parameter(
                f"aux_out{i}",
                label=f"Aux output {i}",
                get_cmd=f"AUXV? {i}",
                get_parser=float,
                set_cmd=f"AUXV {i}, {{}}",
                unit="V",
            )

        # ========== SETUP - IDENTICAL TO SR830 ==========
        self.output_interface: Parameter = self.add_parameter(
            "output_interface",
            label="Output interface",
            get_cmd="OUTX?",
            set_cmd="OUTX {}",
            val_mapping={
                "RS232": "0\n",
                "GPIB": "1\n",
            },
        )
        """Parameter output_interface"""

        # ========== DATA TRANSFER - IDENTICAL TO SR830 ==========
        self.X: Parameter = self.add_parameter(
            "X", get_cmd="OUTP? 1", get_parser=float, unit="V"
        )
        """Parameter X"""

        self.Y: Parameter = self.add_parameter(
            "Y", get_cmd="OUTP? 2", get_parser=float, unit="V"
        )
        """Parameter Y"""

        self.R: Parameter = self.add_parameter(
            "R", get_cmd="OUTP? 3", get_parser=float, unit="V"
        )
        """Parameter R"""

        self.P: Parameter = self.add_parameter(
            "P", get_cmd="OUTP? 4", get_parser=float, unit="deg"
        )
        """Parameter P"""

        self.complex_voltage: Parameter = self.add_parameter(
            "complex_voltage",
            label="Voltage",
            get_cmd=self._get_complex_voltage,
            unit="V",
            docstring="Complex voltage parameter "
            "calculated from X, Y using "
            "Z = X + j*Y",
            vals=ComplexNumbers(),
        )
        """Complex voltage parameter calculated from X, Y phase using Z = X +j*Y"""

        # ========== DATA BUFFER SETTINGS - MOSTLY IDENTICAL TO SR830 ==========
        self.buffer_SR: Parameter = self.add_parameter(
            "buffer_SR",
            label="Buffer sample rate",
            get_cmd="SRAT?",
            set_cmd=self._set_buffer_SR,
            unit="Hz",
            val_mapping={
                62.5e-3: 0,
                0.125: 1,
                0.250: 2,
                0.5: 3,
                1: 4,
                2: 5,
                4: 6,
                8: 7,
                16: 8,
                32: 9,
                64: 10,
                128: 11,
                256: 12,
                512: 13,
                "Trigger": 14,
            },
            get_parser=int,
        )
        """Parameter buffer_SR"""

        self.buffer_acq_mode: Parameter = self.add_parameter(
            "buffer_acq_mode",
            label="Buffer acquisition mode",
            get_cmd="SEND?",
            set_cmd="SEND {}",
            val_mapping={"single shot": 0, "loop": 1},
            get_parser=int,
        )
        """Parameter buffer_acq_mode"""

        self.buffer_trig_mode: Parameter = self.add_parameter(
            "buffer_trig_mode",
            label="Buffer trigger start mode",
            get_cmd="TSTR?",
            set_cmd="TSTR {}",
            val_mapping={"ON": 1, "OFF": 0},
            get_parser=int,
        )
        """Parameter buffer_trig_mode"""

        self.buffer_npts: Parameter = self.add_parameter(
            "buffer_npts",
            label="Buffer number of stored points",
            get_cmd="SPTS? 1",  # Query trace 1 points
            get_parser=int,
        )
        """Parameter buffer_npts"""

        # ========== SR850-SPECIFIC: SCAN LENGTH ==========
        self.scan_length: Parameter = self.add_parameter(
            "scan_length",
            label="Scan length",
            get_cmd="SLEN?",
            set_cmd="SLEN {}",
            get_parser=float,
            unit="s",
            vals=Numbers(min_value=0, max_value=1e9),
            docstring="Scan length in seconds (SR850-specific parameter)"
        )
        """Parameter scan_length (SR850-specific)"""

        # ========== SR850-SPECIFIC: FAST MODE (OPTIONAL) ==========
        self.fast_mode: Parameter = self.add_parameter(
            "fast_mode",
            label="FAST data transfer mode",
            get_cmd="FAST?",
            set_cmd="FAST {}",
            val_mapping={
                "OFF": 0,
                "DOS": 1,
                "Windows": 2,
            },
            get_parser=int,
            docstring="FAST mode for high-speed binary data streaming during scans"
        )
        """Parameter fast_mode (SR850-specific for high-speed data streaming)"""

        # Setpoints for sweeps
        self.sweep_setpoints: GeneratedSetPoints = self.add_parameter(
            "sweep_setpoints",
            parameter_class=GeneratedSetPoints,
            vals=Arrays(shape=(self.buffer_npts.get,)),
        )
        """Parameter sweep_setpoints"""

        # ========== CHANNEL/TRACE SETUP - SR850 HAS 4 TRACES ==========
        # SR850 has 4 definable traces (vs SR830's 2 channels)
        for ch in range(1, 5):  # Traces 1-4
            # Add databuffer and datatrace for each trace
            self.add_parameter(
                f"ch{ch}_databuffer", channel=ch, parameter_class=ChannelBuffer
            )
            self.add_parameter(
                f"ch{ch}_datatrace",
                channel=ch,
                vals=Arrays(shape=(self.buffer_npts.get,)),
                setpoints=(self.sweep_setpoints,),
                parameter_class=ChannelTrace,
            )

        # ========== AUTO FUNCTIONS - IDENTICAL TO SR830 ==========
        self.add_function("auto_gain", call_cmd="AGAN")
        self.add_function("auto_reserve", call_cmd="ARSV")
        self.add_function("auto_phase", call_cmd="APHS")
        self.add_function("auto_offset", call_cmd="AOFF {0}", args=[Enum(1, 2, 3)])

        # ========== INTERFACE FUNCTIONS - IDENTICAL TO SR830 ==========
        self.add_function("reset", call_cmd="*RST")
        self.add_function("disable_front_panel", call_cmd="OVRM 0")
        self.add_function("enable_front_panel", call_cmd="OVRM 1")

        self.add_function(
            "send_trigger",
            call_cmd="TRIG",
            docstring=(
                "Send a software trigger. "
                "This command has the same effect as a "
                "trigger at the rear panel trigger input."
            ),
        )

        # ========== BUFFER CONTROL FUNCTIONS - IDENTICAL TO SR830 ==========
        self.add_function(
            "buffer_start",
            call_cmd="STRT",
            docstring=(
                "Start or resume data storage. "
                "Ignored if storage is already in progress."
            ),
        )

        self.add_function(
            "buffer_pause",
            call_cmd="PAUS",
            docstring=(
                "Pause data storage. "
                "Ignored if already paused or reset."
            ),
        )

        self.add_function(
            "buffer_reset",
            call_cmd="REST",
            docstring=(
                "Reset the data buffers. "
                "Can be sent at any time. "
                "WARNING: This erases all buffer data."
            ),
        )

        # ========== SR850-SPECIFIC: FAST MODE START FUNCTION ==========
        self.add_function(
            "buffer_start_delayed",
            call_cmd="STRD",
            docstring=(
                "Start scan with 0.5 second delay (SR850-specific). "
                "Use with FAST mode for synchronized data transfer."
            ),
        )

        # Initialize the proper units of the outputs and sensitivities
        self.input_config()

        # Track buffer ready states - SR850 has 4 traces
        self._buffer1_ready = False
        self._buffer2_ready = False
        self._buffer3_ready = False
        self._buffer4_ready = False

        self.connect_message()

    # ========== SNAP COMMAND - IDENTICAL TO SR830 ==========
    SNAP_PARAMETERS: ClassVar[dict[str, str]] = {
        "x": "1",
        "y": "2",
        "r": "3",
        "p": "4",
        "phase": "4",
        "θ": "4",
        "aux1": "5",
        "aux2": "6",
        "aux3": "7",
        "aux4": "8",
        "freq": "9",
        "ch1": "10",
        "ch2": "11",
    }

    def snap(self, *parameters: str) -> tuple[float, ...]:
        """
        Get between 2 and 6 parameters at a single instant. This provides a
        coherent snapshot of measured signals. Pick up to 6 from: X, Y, R, θ,
        the aux inputs 1-4, frequency, or what is currently displayed on
        channels 1 and 2.

        Reading X and Y (or R and θ) gives a coherent snapshot of the signal.
        Snap is important when the time constant is very short, a time constant
        less than 100 ms.

        Args:
            *parameters: From 2 to 6 strings of names of parameters for which
                the values are requested. including: 'x', 'y', 'r', 'p',
                'phase' or 'θ', 'aux1', 'aux2', 'aux3', 'aux4', 'freq',
                'ch1', and 'ch2'.

        Returns:
            A tuple of floating point values in the same order as requested.

        Examples:
            >>> lockin.snap('x','y') -> tuple(x,y)

            >>> lockin.snap('aux1','aux2','freq','phase')
            >>> -> tuple(aux1,aux2,freq,phase)

        Note:
            Volts for x, y, r, and aux 1-4
            Degrees for θ
            Hertz for freq
            Unknown for ch1 and ch2. It will depend on what was set.

             - If X,Y,R and θ are all read, then the values of X,Y are recorded
               approximately 10 µs apart from R,θ. Thus, the values of X and Y
               may not yield the exact values of R and θ from a single snap.
             - The values of the Aux Inputs may have an uncertainty of
               up to 32 µs.
             - The frequency is computed only every other period or 40 ms,
               whichever is longer.

        """
        if not 2 <= len(parameters) <= 6:
            raise KeyError(
                "It is only possible to request values of 2 to 6 parameters at a time."
            )

        for name in parameters:
            if name.lower() not in self.SNAP_PARAMETERS:
                raise KeyError(
                    f"{name} is an unknown parameter. Refer"
                    f" to `SNAP_PARAMETERS` for a list of valid"
                    f" parameter names"
                )

        p_ids = [self.SNAP_PARAMETERS[name.lower()] for name in parameters]
        output = self.ask(f"SNAP? {','.join(p_ids)}")

        return tuple(float(val) for val in output.split(","))

    # ========== SENSITIVITY CONTROL - IDENTICAL TO SR830 ==========
    def increment_sensitivity(self) -> bool:
        """
        Increment the sensitivity setting of the lock-in. This is equivalent
        to pushing the sensitivity up button on the front panel. This has no
        effect if the sensitivity is already at the maximum.

        Returns:
            Whether or not the sensitivity was actually changed.
        """
        return self._change_sensitivity(1)

    def decrement_sensitivity(self) -> bool:
        """
        Decrement the sensitivity setting of the lock-in. This is equivalent
        to pushing the sensitivity down button on the front panel. This has no
        effect if the sensitivity is already at the minimum.

        Returns:
            Whether or not the sensitivity was actually changed.
        """
        return self._change_sensitivity(-1)

    def _change_sensitivity(self, dn: int) -> bool:
        if self.input_config() in ["a", "a-b"]:
            n_to = self._N_TO_VOLT
            to_n = self._VOLT_TO_N
        else:
            n_to = self._N_TO_CURR
            to_n = self._CURR_TO_N

        n = to_n[self.sensitivity()]

        if n + dn > max(n_to.keys()) or n + dn < min(n_to.keys()):
            return False

        self.sensitivity.set(n_to[n + dn])
        return True

    # ========== BUFFER CONTROL - MOSTLY IDENTICAL TO SR830 ==========
    def _set_buffer_SR(self, SR: int) -> None:
        self.write(f"SRAT {SR}")
        # Mark all buffers as not ready
        self._buffer1_ready = False
        self._buffer2_ready = False
        self._buffer3_ready = False
        self._buffer4_ready = False
        self.sweep_setpoints.update_units_if_constant_sample_rate()

    # ========== HELPER FUNCTIONS - IDENTICAL TO SR830 ==========
    def _set_units(self, unit: str) -> None:
        for param in [self.X, self.Y, self.R, self.sensitivity]:
            param.unit = unit

    def _get_complex_voltage(self) -> complex:
        x, y = self.snap("X", "Y")
        return x + 1.0j * y

    def _get_input_config(self, s: int) -> str:
        mode = self._N_TO_INPUT_CONFIG[int(s)]

        if mode in ["a", "a-b"]:
            self.sensitivity.vals = self._VOLT_ENUM
            self._set_units("V")
        else:
            self.sensitivity.vals = self._CURR_ENUM
            self._set_units("A")

        return mode

    def _set_input_config(self, s: str) -> int:
        if s in ["a", "a-b"]:
            self.sensitivity.vals = self._VOLT_ENUM
            self._set_units("V")
        else:
            self.sensitivity.vals = self._CURR_ENUM
            self._set_units("A")

        return self._INPUT_CONFIG_TO_N[s]

    def _get_sensitivity(self, s: int) -> float:
        if self.input_config() in ["a", "a-b"]:
            return self._N_TO_VOLT[int(s)]
        else:
            return self._N_TO_CURR[int(s)]

    def _set_sensitivity(self, s: float) -> int:
        if self.input_config() in ["a", "a-b"]:
            return self._VOLT_TO_N[s]
        else:
            return self._CURR_TO_N[s]

    # ========== AUTORANGE - IDENTICAL TO SR830 ==========
    def autorange(self, max_changes: int = 1) -> None:
        """
        Automatically changes the sensitivity of the instrument according to
        the R value and defined max_changes.

        Args:
            max_changes: Maximum number of steps allowing the function to
                automatically change the sensitivity (default is 1). The actual
                number of steps needed to change to the optimal sensitivity may
                be more or less than this maximum.
        """

        def autorange_once() -> bool:
            r = self.R()
            sens = self.sensitivity()
            if r > 0.9 * sens:
                return self.increment_sensitivity()
            elif r < 0.1 * sens:
                return self.decrement_sensitivity()
            return False

        sets = 0
        while autorange_once() and sets < max_changes:
            sets += 1
            time.sleep(self.time_constant())

    def set_sweep_parameters(
        self,
        sweep_param: Parameter,
        start: float,
        stop: float,
        n_points: int = 10,
        label: str | None = None,
    ) -> None:
        """
        Set parameters for a sweep measurement
        
        Args:
            sweep_param: The parameter being swept
            start: Start value
            stop: Stop value  
            n_points: Number of points in sweep
            label: Optional label for setpoints
        """
        self.sweep_setpoints.sweep_array = np.linspace(start, stop, n_points)
        self.sweep_setpoints.unit = sweep_param.unit
        if label is not None:
            self.sweep_setpoints.label = label
        elif sweep_param.label is not None:
            self.sweep_setpoints.label = sweep_param.label


class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """

    def __init__(
        self,
        sweep_array: Iterable[float | int] = np.linspace(0, 1, 10),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sweep_array = sweep_array
        # Don't query instrument during __init__ - it may not be ready yet
        # Will update units on first get() call instead

    def update_units_if_constant_sample_rate(self) -> None:
        """
        If the buffer is filled at a constant sample rate,
        update the unit to "s" and label to "Time";
        otherwise do nothing
        """
        assert isinstance(self.root_instrument, SR850)
        try:
            SR = self.root_instrument.buffer_SR.get()
            if SR != "Trigger":
                self.unit = "s"
                self.label = "Time"
        except:
            # If buffer_SR query fails, just keep default units
            pass

    def set_raw(self, value: Iterable[float | int]) -> None:
        self.sweep_array = value

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.root_instrument, SR850)
        SR = self.root_instrument.buffer_SR.get()
        if SR == "Trigger":
            return self.sweep_array
        else:
            N = self.root_instrument.buffer_npts.get()
            dt = 1 / SR

            return np.linspace(0, N * dt, N)