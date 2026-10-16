# Simulation evaluation

Roll out a checkpoint in the benchmark simulators. [SETUP.md](SETUP.md) installs the simulators;
[docs/evaluation.md](../../docs/evaluation.md) gives the benchmarks, the protocols and the
metrics. `launchers/eval_protocols.py` prints the `rollout.py` commands of each protocol, and the
[launchers](../../launchers/README.md) submit them to Slurm.

| File | Contents |
|---|---|
| `rollout.py` | the CLI of `reki.eval.rollout`: one checkpoint on one benchmark task |
| `robotwin_sim_server.py` | RoboTwin 2.0 server (`robotwin`, `robotwin_clean2rand`) |
| `robocasa365_sim_server.py` | RoboCasa365 server (`robocasa365`) |
| `install.sh` | installs the simulators of the given benchmarks under `$REKI_SIM_ROOT` |
| `install_robotwin.sh` | the RoboTwin install that `install.sh` runs; applies the three `robotwin_*_fix.patch` files |
| `libero_config.sh` | writes the LIBERO and LIBERO-Plus configs and robosuite's macros file |
| `results_table.py` | success-rate tables of one or more eval output directories |
| `viewer.py` | serves the episode videos and results of an eval directory in a browser |
| `sim_relay.py` | TCP relay from the submit node's cluster address to the loopback end of an SSH tunnel |

LIBERO and LIBERO-Plus run in the `rollout.py` process (pixi envs `sim` and `sim-plus`). The
other benchmarks run a server in its own micromamba env, bound to `127.0.0.1` by default.
`rollout.py` connects to it directly or through an SSH tunnel
([Sim servers](../../docs/evaluation.md#sim-servers)).

[docs/evaluation.md](../../docs/evaluation.md#evaluate-your-checkpoints) gives the commands, with
and without Slurm.
