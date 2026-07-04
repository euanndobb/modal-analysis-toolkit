"""National Instruments DAQ acquisition helpers.

Thin wrappers around :mod:`nidaqmx` to drive an output signal (e.g. a shaker)
and simultaneously record one or more input channels (accelerometers, force
sensors). Importing this module requires the ``nidaqmx`` package and, at run
time, connected NI DAQ hardware.
"""

from __future__ import annotations

from typing import Sequence

import nidaqmx


class DAQInput:
    """Configuration and captured data for a single DAQ input channel.

    Parameters
    ----------
    name : str
        Human-readable label for the sensor/channel.
    module : str
        Suffix identifying the DAQ module (matched against device names).
    channel : str
        Physical analog-input channel identifier (e.g. ``"ai1"``).
    ch_type : {"voltage", "iepe"}
        Sensor coupling type. ``"iepe"`` is used for IEPE accelerometers/force
        sensors; ``"voltage"`` for plain voltage inputs.

    Attributes
    ----------
    signal : numpy.ndarray or None
        The recorded samples, populated by :func:`run_daq`.
    fs : int or None
        The sampling rate the data was captured at.
    bit_resolution : int or None
        The ADC bit resolution reported for the capture.
    """

    def __init__(self, name: str, module: str, channel: str, ch_type: str):
        self.name = name
        self.module = module
        self.channel = channel
        self.ch_type = ch_type
        self.signal = None
        self.fs = None
        self.bit_resolution = None


class DAQOutput:
    """Configuration and waveform for a single DAQ analog-output channel.

    Parameters
    ----------
    module : str
        Suffix identifying the DAQ output module.
    channel : str
        Physical analog-output channel identifier (e.g. ``"ao0"``).
    signal : numpy.ndarray
        The waveform samples to write to the output (drives, e.g., a shaker).
    """

    def __init__(self, module: str, channel: str, signal):
        self.module = module
        self.channel = channel
        self.signal = signal


def run_daq(
    inputs: Sequence[DAQInput],
    output: DAQOutput | None,
    fs: int,
    duration: float | None,
) -> Sequence[DAQInput]:
    """Run a synchronised output/input acquisition on an NI DAQ device.

    If ``output`` is supplied, its waveform is written to the output channel
    and the recording length is taken from the length of that waveform; in this
    case ``duration`` must be ``None``. If ``output`` is ``None``, the
    recording length is set by ``duration`` seconds.

    Parameters
    ----------
    inputs : sequence of DAQInput
        Input channels to record. Each is populated in place with the captured
        ``signal``, ``fs`` and ``bit_resolution``.
    output : DAQOutput or None
        Output waveform to play during acquisition, or ``None`` for
        input-only recording.
    fs : int
        Requested sampling rate in Hz. A :class:`ValueError` is raised if the
        hardware cannot match it.
    duration : float or None
        Recording duration in seconds. Must be ``None`` when ``output`` is
        given, and set when it is not.

    Returns
    -------
    sequence of DAQInput
        The same ``inputs`` objects, now populated with captured data.

    Raises
    ------
    ValueError
        If both ``output`` and ``duration`` are given, if no matching input
        module is found, if an unknown ``ch_type`` is requested, if the
        hardware sample rate does not match ``fs``, or if the full output
        waveform could not be written.
    """
    system = nidaqmx.system.System.local()
    input_task = nidaqmx.Task()

    if output is not None:
        if duration is not None:
            raise ValueError("Don't specify duration when an output signal is specified")

        num_samples = len(output.signal)

        output_device = [d for d in system.devices if d.name.endswith(output.module)][0]
        output_chan = output_device.ao_physical_chans[output.channel]

        output_task = nidaqmx.Task()
        output_task.ao_channels.add_ao_voltage_chan(output_chan.name)

        output_task.timing.cfg_samp_clk_timing(
            fs,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=num_samples,
        )

        daq_fs = output_task.timing.samp_clk_rate
        if daq_fs != fs:
            raise ValueError(f"Specified sample rate {fs} not matched by output DAQ {daq_fs}")
    else:
        # No output signal, so we need to set duration of recording explicitly
        num_samples = duration * fs
        print("No output signal, recording for the specified duration")

    print("Adding input channels")
    for input in inputs:
        print("Finding device..")
        input_device = [d for d in system.devices if d.name.endswith(input.module)]
        if len(input_device) == 0:
            raise ValueError("No input DAQ module found. Is it plugged in?")
        else:
            input_device = input_device[0]
            input_chan_sensor = input_device.ai_physical_chans[input.channel]

        if input.ch_type == "voltage":
            print("Voltage channel added")
            input_task.ai_channels.add_ai_voltage_chan(input_chan_sensor.name)
        elif input.ch_type == "iepe":
            print("Iepe channel added")
            input_task.ai_channels.add_ai_force_iepe_chan(input_chan_sensor.name, sensitivity=1)
        else:
            raise ValueError("Unknown sensor type. Support 'voltage' or 'iepe'.")

    input_task.timing.cfg_samp_clk_timing(
        fs,
        sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
        samps_per_chan=num_samples,
    )

    daq_fs = input_task.timing.samp_clk_rate
    if daq_fs != fs:
        raise ValueError(f"Specified sample rate {fs} not matched by input DAQ {daq_fs}")

    # Write output signal to DAQ box to drive phantom
    if output is not None:
        num_samples_written = output_task.write(output.signal, auto_start=True)
        if num_samples_written != len(output.signal):
            raise ValueError("Wrote less samples")

    # Start recording from DAQ box (phantom sensors, external sensors, or both)
    data = input_task.read(num_samples, timeout=(num_samples / fs) + 1)
    input_task.close()

    if output is not None:
        output_task.close()

    if len(inputs) == 1:
        # if just one input, then task.read() returns just the array instead of a list of arrays
        data = [data]

    for signal, input in zip(data, inputs):
        input.signal = signal
        input.fs = fs
        input.bit_resolution = 24

    return inputs
