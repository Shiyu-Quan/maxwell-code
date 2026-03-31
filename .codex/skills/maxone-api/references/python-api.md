# Python API

Use this file for day-to-day Python `maxlab` control code.

## Core Mental Model

- Import `maxlab` as `mx`.
- Treat most hardware commands as objects derived from `maxlab.api_object.ApiObject`.
- Send those objects with `mx.send(...)`; constructing the object alone does nothing.
- Let the API format messages for `mxwserver`; do not assemble protocol payloads by hand.

## Typical Python Building Blocks

- `mx.initialize(...)`: initialize all activated wells or the provided well list.
- `mx.offset()`: run offset compensation after initialization.
- `mx.activate([...])`: select wells on multiwell systems.
- `mx.set_primary_well(well)`: choose the single well used for real-time streaming.
- `mx.query_DAC_lsb_mV()`: return DAC LSB in millivolts for voltage stimulation conversions.
- `mx.electrode_neighbors(index, radius)`: find nearby electrodes for rerouting or fallback stimulation sites.
- `mx.Array(token, persistent=False)`: maintain an electrode-array model on the server.
- `mx.StimulationUnit(unit_no)`: configure one of 32 stimulation units.
- `mx.DAC(dac_no, dac_code, dac_code_second)`: program on-chip DACs.
- `mx.Sequence(name, persistent=False)`: build reusable stimulation sequences.
- `mx.Saving()`: configure directories, files, groups, and recording lifecycle.

## Use This Default Python Flow

1. Import modules and initialize the system.
- Start with `import maxlab as mx`.
- Call `mx.initialize()`.
- If the experiment includes stimulation, enable power with `mx.send(mx.Core().enable_stimulation_power(True))`.

2. Handle well selection when needed.
- On multiwell devices, call `mx.activate([...])` first.
- Use `mx.set_primary_well(...)` when later C++ or live streaming depends on one specific well.

3. Prepare the array.
- Create an `mx.Array(...)` object.
- Reset and clear selections before constructing a new routing plan.
- Call `select_electrodes(...)` for recording electrodes.
- Call `select_stimulation_electrodes(...)` for candidate stimulation electrodes.
- Call `route()`.

4. Verify stimulation routing.
- Connect electrodes with `connect_electrode_to_stimulation(...)`.
- Query assignments with `query_stimulation_at_electrode(...)`.
- Treat missing assignments as real hardware/routing failures, not as a cosmetic issue.

5. Configure stimulation hardware.
- Configure each `mx.StimulationUnit(...)` explicitly.
- Assign DAC sources with `dac_source(...)`.
- Remember that DAC codes are 10-bit values in the range `0..1023`.
- Remember that `512` corresponds to `0 V` and voltage mode is inverted: larger codes mean more negative output.

6. Build and send sequences.
- Use `mx.Sequence(...)` to assemble pulse trains or reusable named sequences.
- Keep sequence names stable if C++ later triggers them via `sendSequence(...)`.
- Send or download the sequence only after routing and stimulation-unit setup are coherent.

7. Configure saving before recording.
- Create `s = mx.Saving()`.
- Call `open_directory(...)`, then `start_file(...)`.
- Call `group_delete_all()` before redefining channel groups.
- Call `group_define(well, name, channels)` after `start_file(...)` and before `start_recording(...)`.

## Respect These Constraints

- Run `mx.offset()` only after `mx.initialize()`. The FAQ states there is no need to rerun offset compensation unless you reinitialize the system.
- Limit stimulation to the 32 available stimulation units.
- Expect routing conflicts. A selected stimulation electrode may fail to route, or multiple stimulation electrodes may land on the same unit.
- Remember there are 3 DACs total. Simultaneous hardware control is limited: the FAQ says only DAC0 and DAC1 can be updated simultaneously, using `dac_no=3`.
- Treat `mx.query_DAC_lsb_mV()` as stimulation-DAC units in millivolts; this is not the same LSB stored in H5 readout data.
- Do not rely on GPIO or `StatusLED` classes on MaxTwo; the FAQ marks them as irrelevant there.

## Use These Practical Heuristics

- Prefer neighboring electrodes when rerouting around stimulation conflicts.
- Keep array setup, unit configuration, and sequence construction in separate helper functions so routing mistakes are inspectable.
- When sharing code, name placeholders explicitly: `recording_electrodes`, `stim_electrodes`, `target_well`, `sequence_name`.
- When the user asks for exact signatures or examples, stop summarizing and run `python scripts/search_docs.py --query "..."`.
