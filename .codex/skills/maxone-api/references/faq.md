# FAQ Notes

Use this file for practical caveats that are easy to miss when writing code from memory.

## Routing And Stimulation Pitfalls

- Selecting a stimulation electrode does not guarantee successful routing.
- Connecting stimulation electrodes does not guarantee one unique stimulation unit per electrode.
- If two critical stimulation electrodes land on one unit, try neighboring electrodes or slightly change the routing configuration.
- Electrodes connected to stimulation units cannot record electrophysiological signals while driven; use adjacent electrodes for recording.
- The docs note that one stimulation unit can usually drive up to 30 electrodes without significantly affecting the pulse for common use cases.

## DAC Limits

- There are 3 DACs in total.
- You can have at most 3 independently configured simultaneous stimulation patterns.
- Simultaneous control is limited: DAC0 and DAC1 can be updated at the same time using `mx.DAC(dac_no=3, ...)`.
- Sequential DAC changes are already fast, on the order of a few microseconds according to the FAQ, so true simultaneity is often unnecessary.

## Initialization, Offsets, And Server Assumptions

- Keep `mxwserver` running for the API to work.
- Do not rerun `mx.offset()` repeatedly; run it after `mx.initialize()` and again only after reinitialization.
- Avoid unsynchronized concurrent access to the same system from multiple clients.

## Multiwell And Device Differences

- Use `mx.set_primary_well()` and `mx.activate()` for multiwell workflows.
- The docs state the command sending rate is 20 kHz on both systems.
- The docs state MaxTwo data streaming is 10 kHz, while MaxOne data streaming is 20 kHz.
- `mx.GPIODirection`, `mx.GPIOOutput`, and `mx.StatusLED` are not relevant for MaxTwo.

## File Formats And Config Files

- `group_define(...)` is relevant for the new file format when saving signal traces.
- `.cfg` files contain channel number, electrode number, X/Y position, and a binary blob describing routing state.
- Use `mx.Config` in Python to parse `.cfg` files when possible.
- In C++, either parse `.cfg` manually or pass the parsed mapping from Python.

## LSB And Units

- The H5 file LSB refers to the readout amplifier and ADC path.
- `mx.query_DAC_lsb_mV()` refers to stimulation DAC LSB.
- The units also differ: H5 uses volts, while the API call returns millivolts.

## C++ Streaming Notes

- Toolchain errors are usually compilation-environment issues; the FAQ points to GCC 11+.
- A slow closed-loop path is often caused by too much per-frame work, not by the device itself.
- Use multiple threads if receiving and processing cannot keep up in one loop.
- FIR and IIR outputs are not directly amplitude-comparable.
