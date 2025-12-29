import time
from qcodes.instrument import VisaInstrument, InstrumentChannel
from qcodes.validators import Numbers


class ESP302Axis(InstrumentChannel):
    """
    Single axis of ESP302 controller
    """

    def __init__(self, parent, name: str, axis: int):
        super().__init__(parent, name)
        self.axis = axis

        self.add_parameter(
            "velocity",
            get_cmd=f"{axis}VA?",
            set_cmd=f"{axis}VA {{}}",
            get_parser=float,
            vals=Numbers(min_value=0),
            unit="units/s",
        )

        self.add_parameter(
            "acceleration",
            get_cmd=f"{axis}AC?",
            set_cmd=f"{axis}AC {{}}",
            get_parser=float,
            vals=Numbers(min_value=0),
            unit="units/s^2",
        )

        self.add_parameter(
            "deceleration",
            get_cmd=f"{axis}AG?",
            set_cmd=f"{axis}AG {{}}",
            get_parser=float,
            vals=Numbers(min_value=0),
            unit="units/s^2",
        )

        self.add_parameter(
            "position",
            get_cmd=f"{axis}TP?",
            set_cmd=self._set_position_abs,
            get_parser=float,
            vals=Numbers(),
            unit="units",
        )

    def _set_position_abs(self, value):
        self.write(f"{self.axis}PA {value}")

    def move_relative(self, delta):
        self.write(f"{self.axis}PR {delta}")

    def wait_for_stop(self):
        self.write(f"{self.axis}WS")
        time.sleep(0.05)

    def wait_until_done(self, poll_interval=0.1, timeout=30):
        t0 = time.time()
        while True:
            done = self.ask(f"{self.axis}MD?")
            if done.strip() == "1":
                return
            if time.time() - t0 > timeout:
                raise TimeoutError("Motion did not complete in time")
            time.sleep(poll_interval)

class ESP302(VisaInstrument):
    """
    Newport ESP302 Motion Controller (Ethernet, ESP command set)
    """

    DEFAULT_IP = "10.51.100.200"
    DEFAULT_PORT = 5001

    def __init__(
        self,
        name: str,
        address: str | None = None,
        timeout: float = 5,
        axes=(1,),
        **kwargs,
    ):
        if address is None:
            address = f"TCPIP::{self.DEFAULT_IP}::{self.DEFAULT_PORT}::SOCKET"

        super().__init__(
            name,
            address,
            timeout=timeout,
            terminator="\r\n",
            **kwargs,
        )

        print(f"Connected to ESP302 at {address}")

        self.add_parameter(
            "version",
            get_cmd="VE?",
            label="Controller Firmware Version",
        )

        self.axes = {}
        for ax in axes:
            axis = ESP302Axis(self, f"axis{ax}", ax)
            self.add_submodule(f"axis{ax}", axis)
            self.axes[ax] = axis

    def get_idn(self):
        return {}
