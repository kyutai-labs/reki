"""RoboTwinEnv and RoboTwinSeedCache against a scripted fake sim server (no simulator)."""

from __future__ import annotations

import pickle
import re
import socket
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from reki.eval import sim_protocol
from reki.eval.rollout.remote import (
    RoboTwinAdapter,
    RoboTwinEnv,
    RoboTwinSeedCache,
)

pytestmark = pytest.mark.cpu

CODE = {
    "code_md5": "c" * 32,
    "task_config": "demo_clean",
    "task_config_md5": "t" * 32,
    "asset_list_md5": "a" * 32,
}


def _scene(seed: int, shift: float = 0.0) -> dict:
    return {
        "instruction": f"instr {seed}",
        "images": {},
        "state": [0.1] * 14,
        "seed_used": seed,
        "sim_state": {"actors": {"box": {"pose": [seed + shift, 0, 0, 1, 0, 0, 0], "name": "box"}}},
    }


class FakeServer:
    """Accepts seeds >= ``accept_from``; rejects at most ``cap`` seeds per reset call."""

    def __init__(
        self,
        accept_from: int,
        cap: int = 90,
        shift: float = 0.0,
        code: dict | None = None,
        trusted_reset: bool = True,
    ) -> None:
        self.accept_from, self.cap, self.shift = accept_from, cap, shift
        self.code, self.trusted_reset = code or CODE, trusted_reset
        self.requests: list[dict] = []

    def handle(self, msg: dict) -> dict:
        self.requests.append(msg)
        if msg["cmd"] == "make":
            return {
                "max_episode_steps": 500,
                "code": self.code,
                "instruction_type": "unseen",
                "trusted_reset": self.trusted_reset,
            }
        if msg.get("trust"):
            return {**_scene(msg["seed"], self.shift), "instruction": msg["instruction"]}
        start = msg["seed"]
        if self.accept_from - start >= self.cap:
            return {
                "error": f"RuntimeError: seed search gave up: {self.cap} seeds rejected for task "
                f"'t' starting at seed {start}",
                "traceback": "",
            }
        return _scene(max(start, self.accept_from))

    def serve(self, conn: socket.socket) -> None:
        while True:
            try:
                msg = sim_protocol.recv(conn)
            except (ConnectionError, OSError):
                return
            sim_protocol.send(conn, self.handle(msg))


Connect = Callable[[FakeServer], socket.socket]


@pytest.fixture
def connect() -> Iterator[Connect]:
    opened = []

    def _connect(server: FakeServer) -> socket.socket:
        client, srv = socket.socketpair()
        threading.Thread(target=server.serve, args=(srv,), daemon=True).start()
        opened.extend([client, srv])
        return client

    yield _connect
    for s in opened:
        s.close()


def _env(conn: socket.socket, cache: Path | None, required: bool = True) -> RoboTwinEnv:
    return RoboTwinEnv(
        conn, "t", RoboTwinAdapter.cameras, RoboTwinAdapter.timeouts, cache, required
    )


def test_miss_searches_past_cap_then_hit_resets_trusted(tmp_path: Path, connect: Connect):
    first = FakeServer(accept_from=1200)
    reply = _env(connect(first), tmp_path).reset(1000)
    assert reply["seed_used"] == 1200
    assert [r["seed"] for r in first.requests if r["cmd"] == "reset"] == [1000, 1090, 1180]
    (entry_file,) = tmp_path.glob("demo_clean_unseen_*/t/1000.pkl")
    assert pickle.loads(entry_file.read_bytes())["instruction"] == "instr 1200"

    second = FakeServer(accept_from=10**9)  # would reject every seed: a hit must not search
    reply = _env(connect(second), tmp_path).reset(1000)
    assert second.requests[-1] == {
        "cmd": "reset",
        "seed": 1200,
        "trust": True,
        "instruction": "instr 1200",
    }
    assert reply["instruction"] == "instr 1200" and reply["seed_used"] == 1200


def test_other_fingerprint_misses(tmp_path: Path, connect: Connect):
    _env(connect(FakeServer(accept_from=5)), tmp_path).reset(3)
    other = FakeServer(accept_from=7, code={**CODE, "code_md5": "d" * 32})
    assert _env(connect(other), tmp_path).reset(3)["seed_used"] == 7
    assert not any(r.get("trust") for r in other.requests)
    assert len(list(tmp_path.glob("demo_clean_unseen_*/fingerprint.json"))) == 2


def test_entry_of_another_fingerprint_raises(tmp_path: Path, connect: Connect):
    env = _env(connect(FakeServer(accept_from=5)), tmp_path)
    env.reset(3)
    assert env.cache is not None
    path = env.cache.dir / "3.pkl"
    entry = pickle.loads(path.read_bytes())
    path.write_bytes(pickle.dumps({**entry, "fingerprint": {**entry["fingerprint"], "x": 1}}))
    with pytest.raises(RuntimeError, match="does not match its key"):
        env.reset(3)


def test_trusted_scene_mismatch_raises(tmp_path: Path, connect: Connect):
    _env(connect(FakeServer(accept_from=5)), tmp_path).reset(5)
    env = _env(connect(FakeServer(accept_from=5, shift=0.01)), tmp_path)
    with pytest.raises(RuntimeError, match="actor box"):
        env.reset(5)


def test_first_writer_wins(tmp_path: Path, connect: Connect):
    _env(connect(FakeServer(accept_from=5)), tmp_path).reset(3)
    cache = RoboTwinSeedCache(tmp_path, "t", FakeServer(0).handle({"cmd": "make"}))
    cache.put(3, _scene(9))
    entry = cache.get(3)
    assert entry is not None and entry["seed_used"] == 5
    assert not list(cache.dir.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("trusted_reset", "required", "cached"),
    [(True, False, True), (False, False, False), (True, True, True), (False, True, False)],
)
def test_cache_follows_the_server_trusted_reset_flag(
    tmp_path: Path, connect: Connect, trusted_reset: bool, required: bool, cached: bool
):
    env = _env(connect(FakeServer(accept_from=4, trusted_reset=trusted_reset)), tmp_path, required)
    assert (env.cache is not None) == cached
    assert env.reset(1)["seed_used"] == 4
    assert bool(list(tmp_path.rglob("*.pkl"))) == cached


def test_auto_cache_falls_back_on_unusable_fingerprint(tmp_path: Path, connect: Connect):
    bad = {**CODE, "code_md5": "error: git not found"}
    server = FakeServer(accept_from=4, code=bad)
    assert _env(connect(server), tmp_path, False).cache is None
    with pytest.raises(RuntimeError, match="unreadable"):
        _env(connect(FakeServer(accept_from=4, code=bad)), tmp_path, True)


def test_play_once_tables_match_the_arm_tag_patch(sim_server: tuple[ModuleType, Any]):
    module, _ = sim_server
    assert module.__file__ is not None
    patch = (Path(module.__file__).parent / "robotwin_arm_tag_fix.patch").read_text()
    files = dict(re.findall(r"^\+\+\+ b/envs/(\w+)\.py\n(.*?)(?=^diff |\Z)", patch, re.M | re.S))
    assert set(module._PLAY_ONCE_ATTRS) == set(files)
    for task, attrs in module._PLAY_ONCE_ATTRS.items():
        for a in attrs:
            assert re.search(rf"^\+\s+self\.{a} = ", files[task], re.M), (task, a)


def test_seed_cache_arg():
    from reki.eval.rollout.remote import seed_cache_root

    assert seed_cache_root("none") == (None, False)
    root, required = seed_cache_root("auto")
    assert root is not None and root.name == "robotwin_seed_cache" and not required
    assert seed_cache_root("~/c") == (Path("~/c").expanduser(), True)


def test_other_server_error_is_not_retried(connect: Connect):
    server = FakeServer(accept_from=0)
    server.handle = lambda msg: (  # type: ignore[method-assign]
        {"error": "boom", "traceback": ""}
        if msg["cmd"] == "reset"
        else FakeServer.handle(server, msg)
    )
    env = _env(connect(server), None)
    with pytest.raises(sim_protocol.SimServerError, match="boom"):
        env.reset(1)


class _Pose:
    def __init__(self, x: float) -> None:
        self.p, self.q = [x, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]


class _Entity:
    def __init__(self, name: str, sid: int) -> None:
        self.name, self.per_scene_id, self.entity = name, sid, self

    def get_name(self) -> str:
        return self.name

    def get_pose(self) -> _Pose:
        return _Pose(self.per_scene_id)


class _FakeTaskEnv:
    """The RoboTwin task env surface the server's reset touches; the expert never succeeds."""

    plan_success = False

    def __init__(self) -> None:
        self.setups: list[int] = []
        self.closes: list[bool] = []  # the clear_cache flag of each close_env
        actors = [_Entity("shoe", 1), _Entity("shoe", 2), _Entity("table", 3), _Entity("", 4)]
        link = _Entity("table", 9)  # an articulation link that shares an actor's name
        art = type("Art", (), {"get_links": lambda self: [link]})()
        self.scene = type(
            "Scene",
            (),
            {"get_all_actors": lambda self: actors, "get_all_articulations": lambda self: [art]},
        )()

    def setup_demo(self, *, seed: int, **_: object) -> None:
        self.setups.append(seed)

    def play_once(self) -> dict:
        return {"info": {}}

    def close_env(self, clear_cache: bool = False) -> None:
        self.closes.append(clear_cache)

    def check_success(self) -> bool:
        return False

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def get_obs(self) -> dict:
        import numpy as np

        img = {"rgb": np.zeros((2, 2, 3), np.uint8)}
        cams = ("head_camera", "left_camera", "right_camera")
        return {"observation": dict.fromkeys(cams, img), "joint_action": {"vector": [0.0] * 14}}


@pytest.fixture
def sim_server(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Any]:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts/eval/robotwin_sim_server.py"
    spec = importlib.util.spec_from_file_location("robotwin_sim_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "robotwin_sim_server", module)
    spec.loader.exec_module(module)
    sim = object.__new__(module.RoboTwinSim)
    sim.env, sim.args, sim.task_name, sim.instruction = _FakeTaskEnv(), {}, "t", ""
    sim.trusted_reset, sim._scene_open, sim._n_closed = True, False, 0
    sim._UnStableError = type("UnStableError", (Exception,), {})
    return module, sim


def test_server_trusted_reset(sim_server: tuple[ModuleType, Any]):
    _, sim = sim_server
    with pytest.raises(ValueError, match="instruction"):
        sim.reset(7, trust=True)
    assert sim.env.setups == []
    reply = sim.reset(7, trust=True, instruction="put the shoe")
    assert sim.env.setups == [7] and reply["seed_used"] == 7
    assert reply["instruction"] == sim.env.instruction == "put the shoe"
    assert sorted(reply["sim_state"]["actors"]) == ["#4", "shoe#1", "shoe#2", "table#3"]


def test_server_closes_the_last_scene_and_clears_the_render_cache(
    sim_server: tuple[ModuleType, Any],
):
    _, sim = sim_server
    sim.args["clear_cache_freq"] = 2
    for seed in range(4):
        sim.reset(seed, trust=True, instruction="x")
    assert sim.env.setups == [0, 1, 2, 3]
    # Each reset closes the scene of the one before; every 2nd close frees the render cache.
    assert sim.env.closes == [False, True, False]
    sim.close()
    assert sim.env.closes == [False, True, False, True]


def test_server_trusted_reset_needs_load_actors_to_set_the_play_once_state(
    sim_server: tuple[ModuleType, Any],
):
    module, sim = sim_server

    class open_laptop:  # noqa: N801
        def load_actors(self):
            self.laptop = None

    class open_laptop_fixed:  # noqa: N801
        def load_actors(self):
            self.laptop = None
            self.arm_tag = "left"

    def ok(task: str, cls: type) -> bool:
        return module.RoboTwinSim._trusted_reset_ok(type("M", (), {task: cls}), task)

    assert not ok("open_laptop", open_laptop)
    assert ok("open_laptop", open_laptop_fixed)
    assert ok("adjust_bottle", open_laptop)
    sim.trusted_reset, sim.task_name = False, "open_laptop"
    with pytest.raises(RuntimeError, match="robotwin_arm_tag_fix.patch"):
        sim.reset(7, trust=True, instruction="open the laptop")
    assert sim.env.setups == []


def test_server_search_gives_up_in_the_client_format(
    sim_server: tuple[ModuleType, Any], connect: Connect
):
    module, sim = sim_server
    with pytest.raises(RuntimeError) as e:
        sim.reset(100)
    assert sim.env.setups == list(range(100, 100 + module._MAX_REJECTED_SEEDS))
    server = FakeServer(accept_from=0)
    server.handle = lambda msg: (  # type: ignore[method-assign]
        {"error": f"RuntimeError: {e.value}", "traceback": ""}
        if msg["cmd"] == "reset" and msg["seed"] == 100
        else FakeServer.handle(server, msg)
    )
    assert _env(connect(server), None).reset(100)["seed_used"] == 100 + module._MAX_REJECTED_SEEDS


def test_server_error_reply_keeps_serving(sim_server: tuple[ModuleType, Any], tmp_path: Path):
    import time

    module, sim = sim_server
    port_file = tmp_path / "port"
    args = ("127.0.0.1", 0, str(port_file), sim)
    threading.Thread(target=module.serve, args=args, daemon=True).start()
    for _ in range(200):
        if port_file.exists() and port_file.read_text():
            break
        time.sleep(0.05)
    with socket.create_connection(("127.0.0.1", int(port_file.read_text())), timeout=10) as conn:
        with pytest.raises(sim_protocol.SimServerError, match="instruction"):
            sim_protocol.rpc(conn, {"cmd": "reset", "seed": 1, "trust": True})
        msg = {"cmd": "reset", "seed": 1, "trust": True, "instruction": "x"}
        assert sim_protocol.rpc(conn, msg)["seed_used"] == 1
