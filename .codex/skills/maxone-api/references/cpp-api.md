# C/C++ API

Use this file for real-time data streaming and closed-loop triggers.

## When To Switch To C++

- Switch from Python to C/C++ when the code must react to the incoming stream in real time.
- Keep Python responsible for preparation work: initialization, routing, stimulation-unit setup, and named sequence creation.
- Keep C/C++ responsible for opening the stream, watching frames or spike events, deciding when to trigger, and calling `sendSequence(...)`.

## Prerequisites

- Use the `maxlab_lib` package from `~/MaxLab/share/libmaxlab-*.zip`.
- Expect a GCC 11+ toolchain; the docs were compiled with GCC 11.2.1 and `-std=gnu++20`.
- Call `maxlab::checkVersions()` before opening streams.

## Important Types And Functions

- `maxlab::FilterType::{FIR,IIR}`: filter mode for filtered streams.
- `maxlab::Status`: API and stream status codes.
- `maxlab::verifyStatus(status)`: print an error and exit on non-OK status.
- `maxlab::sendSequence("name")`: trigger a named sequence that Python already prepared.
- `maxlab::DataStreamerRaw_open()` / `_close()` / `_receiveNextFrame(...)`: raw voltage streaming.
- `maxlab::DataStreamerFiltered_open(filter)` / `_close()` / `_receiveNextFrame(...)`: filtered spike-event streaming.
- `maxlab::FrameInfo`: frame number, well id, and corruption flag.
- `maxlab::RawFrameData`: raw sample amplitudes for 1024 channels.
- `maxlab::FilteredFrameData`: filtered spike events plus frame metadata.
- `maxlab::SpikeEvent`: per-spike channel, frame, and amplitude fields.

## Follow This Runtime Pattern

1. Prepare everything in Python first.
- Initialize the system.
- Route electrodes.
- Create named sequences such as `trigger` or `closed_loop`.
- Save the Python setup as a script that can run before or alongside the C++ process.

2. Open the stream in C++.
- Call `maxlab::checkVersions()`.
- Call the relevant `DataStreamer* _open(...)`.
- Sleep briefly after opening; the tutorial examples wait about 2 seconds.

3. Run a tight loop.
- Allocate one frame struct outside or near the loop.
- Call `receiveNextFrame(...)`.
- Ignore `MAXLAB_NO_FRAME`.
- Ignore corrupted raw frames.
- Apply a blanking interval after each trigger to avoid immediate retriggering.

4. Trigger prepared sequences.
- Inspect either raw amplitudes or filtered `SpikeEvent` entries.
- Call `maxlab::sendSequence("closed_loop")` when the trigger condition is met.
- Keep the sequence name aligned with the Python setup script.

5. Close the stream.
- Always call the matching `_close()` function.
- Treat stream-close cleanup as mandatory, otherwise the tutorial warns the server can remain in an undefined state.

## Raw Vs Filtered Stream

- Use raw streaming when the trigger depends on waveform amplitude or custom signal processing.
- Use filtered streaming when the trigger can operate on detected spike events.
- The docs state FIR latency is roughly `2048 / sampling_rate` ms.
- The docs state IIR latency is roughly `10 / sampling_rate` ms.
- Do not compare FIR and IIR spike amplitudes directly; the FAQ says they are different detection methods.

## Units, Timing, And Performance

- Raw amplitudes and spike amplitudes are in bits, not volts.
- Convert bits to volts or microvolts using the LSB for the chosen gain.
- The FAQ gives example LSB values in microvolts: gain `1 -> 3222.66`, `7 -> 460.38`, `112 -> 28.77`, `512 -> 6.29`, `1024 -> 3.15`, `2048 -> 1.57`.
- If per-frame computation becomes heavy, use separate threads for receiving and processing instead of blaming the hardware first.

## Handle Missing Metadata Deliberately

- Electrode positions and channel mappings are not magically available in C++.
- Parse `.cfg` files manually or pass information from Python via `mx.Config` or another handoff channel.
- Do not assume frame numbers are a perfect wall-clock; the FAQ notes they can arrive with jitter.
