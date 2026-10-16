"""Simulation rollouts of a REKI checkpoint (CLI: ``scripts/eval/rollout.py``).

``loop`` runs the episodes of every benchmark and writes the results; ``libero`` (in-process) and
``remote`` (sim servers over a socket) adapt each benchmark; ``workers`` runs several rollout
processes on one GPU; ``traj`` records the optional per-episode trace.
"""
