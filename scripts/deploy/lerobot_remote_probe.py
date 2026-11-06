#!/usr/bin/env python3
"""Dummy robot client for ``lerobot-policy-server``: no robot, no cameras.

It connects with LeRobot's remote client, takes the served feature schema from the server and
requests one action chunk per ``history_stride_s`` with constant states and black frames, so the
actions have no meaning. It checks the handshake, the chunk shape, the cadence and the session reset.

Usage (the server runs ``examples/lerobot_policy_server/server.yaml``):
    python scripts/deploy/lerobot_remote_probe.py --endpoint=tcp/127.0.0.1:7447 \\
        --deployment=reki-so101 --semantics=so101-degrees-gripper-percent-v1 --seconds=60
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
from lerobot.inference import (
    ChunkRequest,
    ExecutionMode,
    ObservationSnapshot,
    RemoteInferenceConfig,
)
from lerobot.policies.rtc.action_queue import QueueSnapshot
from lerobot.remote_inference import RemoteClient


@dataclass
class ProbeResult:
    requests: int = 0
    shape_ok: int = 0
    shape: tuple[int, ...] = ()
    expected_shape: tuple[int, ...] = ()
    gaps_s: list[float] = field(default_factory=list)
    reset_acknowledged: bool = False


def run_probe(
    config: RemoteInferenceConfig, task: str, seconds: float, period_s: float
) -> ProbeResult:
    """Send ``seconds / period_s`` requests at ``period_s`` spacing, then reset the session."""
    client = RemoteClient.connect(config)
    caps = client.capabilities
    result = ProbeResult(expected_shape=(caps.execution_steps, caps.action_feature.shape[0]))
    try:
        client.admit(
            features=caps.features,
            action_feature=caps.action_feature,
            semantics=config.semantics,
            action_interval=caps.action_interval,
            mode=config.mode,
        )
        empty = QueueSnapshot(
            generation=0,
            cursor=0,
            index=0,
            model_actions=None,
            canonical_actions=None,
            provenance=(),
        )
        started = last = time.monotonic()
        for i in range(max(1, round(seconds / period_s))):
            time.sleep(max(0.0, started + i * period_s - time.monotonic()))
            now = time.monotonic()
            if i:
                result.gaps_s.append(now - last)
            last = now
            observation = ObservationSnapshot(
                {
                    f.name: np.zeros(f.shape, dtype=f.dtype)
                    if f.kind == "rgb"
                    else np.full(f.shape, 0.1 * i, dtype=f.dtype)
                    for f in caps.features
                },
                now,
                task,
                observation_id=f"probe{i}",
            )
            chunk = client.infer(
                ChunkRequest(
                    observation=observation,
                    continuation=empty,
                    request_id=uuid4().hex,
                    generation=client.generation,
                    mode=ExecutionMode.CHUNK,
                    delay=0,
                    submitted_at=now,
                )
            )
            result.requests += 1
            result.shape = tuple(chunk.canonical_actions.shape)
            result.shape_ok += result.shape == result.expected_shape
        client.control("reset", client.generation + 1)
        result.reset_acknowledged = True
    finally:
        client.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    p.add_argument("--deployment", required=True, help="`deployment` of the server YAML")
    p.add_argument("--semantics", required=True, help="`semantics` of the server YAML")
    p.add_argument("--task", default="put the bowl on the plate")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--period_s", type=float, default=1.0, help="history_stride_s of the checkpoint")
    a = p.parse_args()
    config = RemoteInferenceConfig(
        endpoint=a.endpoint, deployment=a.deployment, semantics=a.semantics, chunk_merge="append"
    )
    r = run_probe(config, a.task, a.seconds, a.period_s)
    gaps = np.array(r.gaps_s)
    print(
        f"requests={r.requests} shape_ok={r.shape_ok}/{r.requests} shape={r.shape} "
        f"expected={r.expected_shape} gap_s min/mean/max="
        f"{gaps.min():.3f}/{gaps.mean():.3f}/{gaps.max():.3f} reset_acknowledged={r.reset_acknowledged}"
        if len(gaps)
        else f"requests={r.requests} shape_ok={r.shape_ok}/{r.requests}"
    )
    raise SystemExit(0 if r.shape_ok == r.requests and r.reset_acknowledged else 1)


if __name__ == "__main__":
    main()
