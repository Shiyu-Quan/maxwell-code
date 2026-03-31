# Workflows

Use this file for end-to-end structure rather than low-level API details.

## Python Open-Loop Experiment

Use this order unless the request clearly needs a variant:

1. Import modules.
- Import `time` and `maxlab as mx`.

2. Initialize.
- Call `mx.initialize()`.
- Enable stimulation power if stimulation is part of the experiment.

3. Choose electrodes.
- Separate recording electrodes from stimulation electrodes.
- Keep the stimulation set small enough to survive routing and the 32-unit limit.

4. Route the array.
- Build an `mx.Array(...)`.
- Reset it, clear prior selections, select electrodes, prioritize stimulation electrodes, and call `route()`.

5. Bind stimulation units.
- Connect each stimulation electrode to a stimulation unit.
- Query the actual mapping back from the array object.
- Rework the electrode list if two critical electrodes land on one unit or fail to route.

6. Prepare DACs and sequences.
- Configure `mx.StimulationUnit(...)`.
- Build a named `mx.Sequence(...)`.
- Keep names stable if later reused from C++.

7. Prepare saving.
- Use `mx.Saving()` to open a directory and file.
- Call `group_delete_all()` and then `group_define(...)` if raw traces are needed.

8. Run and clean up.
- Start recording.
- Send or trigger sequences.
- Stop recording and let post-recording waits happen where appropriate.

## Python + C++ Closed-Loop Split

Use this split when the user asks for real-time reaction:

### Python Setup Side

- Initialize wells and enable stimulation.
- Clear old named sequences if the tutorial pattern requires reusing names like `trigger` or `closed_loop`.
- Set detection thresholds when required.
- Route the array and create stimulation sequences.
- Prepare saving the same way as in the Python tutorial.

### C++ Runtime Side

- Open the raw or filtered stream.
- Wait for the stream to become ready.
- Inspect frames or spike events in a tight loop.
- Trigger `sendSequence("closed_loop")` when the condition matches.
- Apply a blanking window.
- Close the stream on every exit path.

## Example File Routing

- `docs/examples/python/stimulate.html`: general stimulation script patterns.
- `docs/examples/python/sim_stim.html`: simultaneous stimulation on two electrodes with independent stimulation units.
- `docs/examples/python/interleave.html`: interleaving two stimulation sequences.
- `docs/examples/python/saving.html`: recording and `mx.Saving()` flow.
- `docs/examples/cpp/raw.html`: raw-data closed-loop trigger loop.
- `docs/examples/cpp/filtered.html`: filtered spike-event trigger loop.

## Ask For Missing Inputs Early

When code would otherwise be fake, surface placeholders for:

- `target_well`
- `recording_electrodes`
- `stim_electrodes`
- `detection_channel`
- `sequence_name`
- stimulation amplitudes or DAC codes
- desired saved channels or recording groups
