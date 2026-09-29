# Working on RoomGraph

The user prioritizes training efficiency, robustness, training speed, clean
Python, meaningful unit tests, and useful visualizations.

- Optimize measured time to validation quality. Report full-run wall time,
  throughput, peak tensor memory, and boundary F1; include compilation/startup.
- Compare systems changes with the same data, model objective, batch/update
  budget, and schedule. Validate numerical quality before adopting a speedup.
- Keep all views of a room/building in one split. Tune on validation only and
  reserve test evaluation for frozen milestone experiments.
- Reuse generated training caches. Do not render inside the training loop or
  run competing GPU jobs during a benchmark. Keep an eager fallback for short
  diagnostics and hardware where compilation has not been measured.
- Preserve finite-gradient checks, experiment provenance, atomic checkpoints,
  and resumable sample order. Never silently overwrite an existing run.
- Keep generated datasets/checkpoints/logs and the full `vis/` tree local and
  ignored by Git. Only selected public website media belong in the website repo.
- Run Ruff and meaningful unit tests with the learning environment when changing
  the trainer. The lightweight `.venv` may skip tests without PyTorch; use the
  host CUDA-enabled Python for the full learning suite.
- For active-view experiments, expose only acquired RGB predictions and calibration
  to the policy. Keep future frames and renderer ground truth behind the evaluator
  boundary; record motion budgets separately from compute latency. Archive frozen
  predictions and verify replay consistency before comparing camera policies.
- Report synthetic-pilot limitations honestly. A hidden predicted wall is a
  structural hypothesis, not verified free space for robot navigation.

See `docs/training-efficiency.md` and `docs/perception.md` for the protocol.
