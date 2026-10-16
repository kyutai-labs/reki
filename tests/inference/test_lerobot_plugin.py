"""``lerobot_policy_reki``: LeRobot's LIBERO eval path (env processor → plugin processors →
``select_action``) must give the actions of ``scripts/eval/rollout.py``'s LIBERO loop, and a
training-dataset frame must give the actions of ``RekiPolicy.predict`` for any export."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import QWEN, TINY_EXPERT

TASK = "put the black bowl on the plate"
LIBERO_CAMERAS = {
    "observation.images.image": "camera_top",
    "observation.images.image2": "camera_wrist_left",
}
N_STEPS = 30  # 3 predicts at n_action_steps=10: the register memory carries 2 past frames


class _Env:
    task_description = TASK


def _raw_obs(rng: np.random.Generator, hw: tuple[int, int] = (24, 32)) -> dict[str, Any]:
    """One ``LiberoEnv._format_raw_obs`` observation (float64 state, uint8 HWC images at the
    tiny leaf's size, which the export declares)."""
    quat = rng.normal(size=4)
    q = rng.uniform(0.0, 0.04)
    return {
        "pixels": {k: rng.integers(0, 256, (*hw, 3), dtype=np.uint8) for k in ("image", "image2")},
        "robot_state": {
            "eef": {
                "pos": rng.uniform(-0.5, 0.5, 3),
                "quat": quat / np.linalg.norm(quat),
                "mat": np.eye(3),
            },
            "gripper": {"qpos": np.asarray([q, -q]), "qvel": np.zeros(2)},
            "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
        },
    }


def _batched(tree: Any) -> Any:
    """Add the vector env's leading batch dim (``eval.batch_size=1``)."""
    if isinstance(tree, dict):
        return {k: _batched(v) for k, v in tree.items()}
    return np.asarray(tree)[None]


def _lerobot_pipelines(cfg: Any, ckpt: Path, device: str) -> tuple[Any, Any, Any]:
    """The env + policy processors as ``lerobot_eval.eval_main`` builds them."""
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors

    env_pre, _ = make_env_pre_post_processors(LiberoEnv(), cfg)
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    return env_pre, pre, post


def _lerobot_batch(raw: dict[str, Any], env_pre: Any, pre: Any) -> dict[str, Any]:
    from lerobot.envs.utils import preprocess_observation

    obs = preprocess_observation(_batched(raw))
    obs["task"] = [TASK]
    return pre(env_pre(obs))


def _train_cfg(dataset: str, **overrides: Any) -> Any:
    from reki.policy.train_config import TrainConfig

    kw: dict[str, Any] = dict(
        dataset=dataset,
        vlm_model_id=QWEN,
        chunk_duration_s=1,
        history_k=3,
        action_expert_num_registers=8,
        state_encoding="adaln",
        max_cameras=3,
        **TINY_EXPERT,
    )
    kw.update(overrides)
    return TrainConfig(**kw)


def _spec_export(
    leaf_maker: Callable[..., Path], out: Path, dataset: str, benchmark: str, **overrides: Any
) -> Any:
    """A weightless export of a tiny ``dataset`` leaf: train config, spec and LeRobot files."""
    from lerobot.configs.policies import PreTrainedConfig

    from lerobot_policy_reki.export import write_lerobot_files
    from reki.inference.spec import build_inference_spec, save_inference_spec

    leaf_maker(dataset)
    cfg = _train_cfg(dataset, **overrides)
    out.mkdir()
    cfg.save_yaml(out / "train_config.yaml")
    save_inference_spec(out, *build_inference_spec(cfg, benchmark=benchmark))
    write_lerobot_files(out)
    return PreTrainedConfig.from_pretrained(out)


@pytest.mark.cpu
def test_observation_matches_rollout(
    tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State and images reach ``RekiPolicy.predict`` as rollout.py builds them: LiberoEnv
    emits the frames of the LIBERO training dataset, so the generic export needs no adapter.
    The plugin passes the native state; ``RekiPolicy`` maps it as training does, bit for bit."""
    from conftest import robot_dataset
    from lerobot.utils.constants import OBS_STATE

    from lerobot_policy_reki.modeling_reki import to_uint8_hwc
    from reki.eval.rollout.libero import libero_obs_to_envobs
    from reki.inference.features import model_state
    from reki.inference.spec import load_inference_spec
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    out = tmp_path / "export"
    cfg = _spec_export(tiny_lerobot_leaf, out, "libero", "libero")
    assert cfg.camera_map == LIBERO_CAMERAS
    spec, _ = load_inference_spec(out)
    assert spec.state_dim == 8 and spec.state_euler_blocks == [[3, 6]] and spec.state_recenter_roll
    monkeypatch.setattr(Qwen3_5Preprocessor, "from_pretrained", lambda *a, **k: None)
    train = robot_dataset("libero")
    env_pre, pre, _ = _lerobot_pipelines(cfg, out, "cpu")
    rng = np.random.default_rng(0)
    for _ in range(5):
        raw = _raw_obs(rng)
        batch = _lerobot_batch(raw, env_pre, pre)
        native = batch[OBS_STATE][0].float().cpu().numpy()  # what the plugin passes to predict
        assert native.shape == (spec.state_dim,)
        item = {OBS_STATE: torch.from_numpy(native.copy())}
        train._state_action_to_model_layout(item, 0)
        assert np.array_equal(model_state(spec, native), item[OBS_STATE].numpy())
        ref = libero_obs_to_envobs(_Env(), raw, convert_rot_to_euler=True, recenter_roll=True)
        np.testing.assert_allclose(model_state(spec, native), ref.state_raw, rtol=0, atol=1e-6)
        for key, img in zip(LIBERO_CAMERAS, ref.images, strict=True):
            assert np.array_equal(to_uint8_hwc(batch[key][0]), np.asarray(img))
        assert batch["task"] == [TASK]


def _dataset_batches(leaf: Path, n: int) -> list[dict[str, Any]]:
    """The first ``n`` frames of a leaf as LeRobot batches (``observation.*`` + ``task``)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"local/{leaf.name}", root=leaf)
    out = []
    for i in range(n):
        frame = ds[i]
        batch = {k: v[None] for k, v in frame.items() if k.startswith("observation.")}
        batch["task"] = [frame["task"]]
        out.append(batch)
    return out


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("dataset", "benchmark"),
    [
        ("robotwin2_clean", "robotwin"),
        ("robocasa365", "robocasa365"),
    ],
)
def test_dataset_frame_reaches_predict(
    tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path, dataset: str, benchmark: str
) -> None:
    """A non-LIBERO export takes the training dataset's features: its image keys and sizes and
    its full state, which the preprocessor passes through and ``RekiPolicy`` slices."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_STATE

    from reki.inference.spec import load_inference_spec

    out = tmp_path / "export"
    cfg = _spec_export(tiny_lerobot_leaf, out, dataset, benchmark)
    spec, _ = load_inference_spec(out)
    leaf = next(p for p in tmp_path.rglob("meta") if p.parent != out).parent
    batches = _dataset_batches(leaf, 3)
    images = sorted(k for k in batches[0] if k.startswith("observation.images."))
    assert sorted(cfg.camera_map) == images
    assert [cfg.camera_map[k] for k in cfg.camera_map] == spec.camera_keys
    for key in images:
        assert tuple(cfg.input_features[key].shape) == tuple(batches[0][key].shape[1:])
    assert cfg.input_features[OBS_STATE].shape == batches[0][OBS_STATE].shape[1:]
    assert cfg.output_features["action"].shape == (spec.action_dim,)
    assert (cfg.n_action_steps, cfg.fps) == (spec.n_action_steps, spec.prompt_fps)

    pre, _ = make_pre_post_processors(cfg, pretrained_path=str(out))
    for batch in batches:
        got = pre(dict(batch))
        assert torch.equal(got[OBS_STATE], batch[OBS_STATE])
        assert got[OBS_STATE].shape == (1, spec.state_dim)


@pytest.mark.cpu
def test_export_needs_a_spec(tmp_path: Path) -> None:
    """The pretrained checkpoint has no robot spec: the export says how to get one."""
    from lerobot_policy_reki.export import write_lerobot_files

    _train_cfg("libero").save_yaml(tmp_path / "train_config.yaml")
    with pytest.raises(FileNotFoundError, match="finetune it on your robot"):
        write_lerobot_files(tmp_path)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.usefixtures("tiny_qwen")
def test_actions_match_rollout(
    synthetic_group: Callable[..., tuple[Any, dict]], tmp_path: Path
) -> None:
    """A tiny exported checkpoint, loaded by LeRobot's factory, acts like rollout.py over three
    register-memory predicts under the same noise."""
    import safetensors.torch as st
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies.factory import make_policy
    from lerobot.utils.import_utils import register_third_party_plugins

    from reki.eval.rollout import libero as rollout_libero
    from reki.eval.rollout import loop as rollout_loop
    from reki.eval.rollout.libero import decode_libero_action
    from reki.inference.benchmarks import BENCHMARKS
    from reki.inference.features import build_inference_sample
    from reki.policy.configuration import RekiConfig
    from reki.policy.modeling import RekiModel
    from reki.policy.optim import EMAModel
    from scripts.export_checkpoint import export_checkpoint

    cfg, _group = synthetic_group(k=3)
    torch.manual_seed(0)
    ckpt = tmp_path / "checkpoint-1"
    source = RekiModel(RekiConfig.from_train_config(cfg, device="cuda"))
    source.save_pretrained(ckpt)
    st.save_file(EMAModel(source, decay=0.999).state_dict(), str(ckpt / "ema.safetensors"))
    cfg.save_yaml(str(ckpt / "train_config.yaml"))
    out = tmp_path / "export"
    manifest = export_checkpoint(ckpt, out, merge_ema=True, source_rev="test", benchmark="libero")
    assert manifest["lerobot"] == [
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]

    register_third_party_plugins()
    policy_cfg = PreTrainedConfig.from_pretrained(out)
    assert policy_cfg.type == "reki"
    policy_cfg.pretrained_path = out
    policy = make_policy(
        policy_cfg, env_cfg=LiberoEnv(observation_height=256, observation_width=256)
    )
    env_pre, pre, post = _lerobot_pipelines(policy_cfg, out, "cuda")
    reki = policy.reki
    n = reki.n_action_steps
    assert n == 10

    rng = np.random.default_rng(0)
    raws = [_raw_obs(rng) for _ in range(N_STEPS)]

    policy.reset()
    ours = []
    for step, raw in enumerate(raws):
        torch.manual_seed(step)
        with torch.inference_mode():
            ours.append(post(policy.select_action(_lerobot_batch(raw, env_pre, pre)))[0].numpy())

    # the rollout loop's LIBERO path (reki/eval/rollout/loop.py + libero.py) on the same weights
    vla, train_cfg = reki.policy, reki.train_cfg
    fps, stats = rollout_loop.eval_stats(out, "libero", train_cfg)
    state_stats, action_stats = rollout_loop._task_stats(stats, None)
    vla.reset_episode()
    ref = []
    for step, raw in enumerate(raws):
        if step % n:
            continue
        torch.manual_seed(step)
        batch = build_inference_sample(
            rollout_libero.libero_obs_to_envobs(
                _Env(), raw, convert_rot_to_euler=True, recenter_roll=True
            ),
            preprocessor=reki.preprocessor,
            train_cfg=train_cfg,
            state_stats=state_stats,
            embodiment=BENCHMARKS["libero"].embodiment,
            control_mode=BENCHMARKS["libero"].control_mode,
            device="cuda",
            robot_type=BENCHMARKS["libero"].robot_type,
            fps=fps,
            action_dim=reki.spec.action_dim,
            dataset="libero",
        )
        with torch.no_grad():
            chunk = vla.predict_action_chunk(batch)[0, :n, : reki.spec.action_dim].float()
        ref.extend(
            decode_libero_action(
                chunk[t],
                action_stats,
                unify_ee_euler_rotation=train_cfg.unify_ee_euler_rotation,
            )
            for t in range(n)
        )
    diff = np.abs(np.stack(ours) - np.stack(ref)).max()
    assert diff < 1e-5, diff


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.usefixtures("tiny_qwen")
def test_dataset_actions_match_predict(
    tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path
) -> None:
    """A tiny RoboCasa365 export, loaded by LeRobot's factory from the dataset metadata, returns
    ``RekiPolicy.predict``'s actions for dataset frames over three register-memory predicts."""
    import safetensors.torch as st
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import OBS_STATE
    from lerobot.utils.import_utils import register_third_party_plugins

    from lerobot_policy_reki.modeling_reki import to_uint8_hwc
    from reki.policy.configuration import RekiConfig
    from reki.policy.modeling import RekiModel
    from reki.policy.optim import EMAModel
    from scripts.export_checkpoint import export_checkpoint

    leaf = tiny_lerobot_leaf("robocasa365")
    cfg = _train_cfg("robocasa365", image_size=224)
    torch.manual_seed(0)
    ckpt = tmp_path / "checkpoint-1"
    source = RekiModel(RekiConfig.from_train_config(cfg, device="cuda"))
    source.save_pretrained(ckpt)
    st.save_file(EMAModel(source, decay=0.999).state_dict(), str(ckpt / "ema.safetensors"))
    cfg.save_yaml(str(ckpt / "train_config.yaml"))
    out = tmp_path / "export"
    manifest = export_checkpoint(
        ckpt, out, merge_ema=True, source_rev="test", benchmark="robocasa365"
    )
    assert manifest["lerobot"][0] == "config.json"

    register_third_party_plugins()
    policy_cfg = PreTrainedConfig.from_pretrained(out)
    policy_cfg.pretrained_path = out
    policy = make_policy(policy_cfg, ds_meta=LeRobotDatasetMetadata("local/robocasa365", root=leaf))
    pre, post = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(out),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    reki = policy.reki
    n = reki.n_action_steps
    batches = _dataset_batches(leaf, 3 * n)

    policy.reset()
    ours = []
    for step, batch in enumerate(batches):
        torch.manual_seed(step)
        with torch.inference_mode():
            ours.append(post(policy.select_action(pre(dict(batch))))[0].numpy())

    reki.reset()
    ref = []
    for step, batch in enumerate(batches):
        if step % n:
            continue
        torch.manual_seed(step)
        images = {cam: to_uint8_hwc(batch[key][0]) for key, cam in policy_cfg.camera_map.items()}
        state = batch[OBS_STATE][0].numpy()
        ref.extend(reki.predict(images, state, batch["task"][0]))
    diff = np.abs(np.stack(ours) - np.stack(ref)).max()
    assert diff < 1e-5, diff


@pytest.mark.cpu
def test_load_turns_tf32_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """lerobot-eval turns TF32 matmuls on before it loads the policy; the plugin turns them off,
    as rollout.py runs."""
    from types import SimpleNamespace

    from lerobot_policy_reki import RekiConfig, RekiPolicy
    from reki.inference import policy

    inner = SimpleNamespace(
        policy=torch.nn.Linear(1, 1), spec=SimpleNamespace(n_action_steps=1, prompt_fps=0.0)
    )
    monkeypatch.setattr(policy.RekiPolicy, "from_pretrained", lambda *a, **k: inner)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    RekiPolicy.from_pretrained("unused", config=RekiConfig())
    assert not torch.backends.cuda.matmul.allow_tf32


@pytest.mark.cpu
def test_rejects_batches_and_image_sizes() -> None:
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import OBS_STATE

    from lerobot_policy_reki import RekiConfig, RekiPolicy

    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
    cfg = RekiConfig(
        input_features=dict.fromkeys(LIBERO_CAMERAS, image), camera_map=dict(LIBERO_CAMERAS)
    )
    policy = RekiPolicy(cfg, SimpleNamespace(policy=torch.nn.Linear(1, 1)))  # type: ignore[arg-type]
    batch = {key: torch.zeros(2, 3, 256, 256) for key in LIBERO_CAMERAS}
    batch[OBS_STATE] = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="batch_size=1"):
        policy.select_action({**batch, "task": [TASK, TASK]})
    big = {key: torch.zeros(1, 3, 360, 360) for key in LIBERO_CAMERAS}
    with pytest.raises(ValueError, match="expects"):
        policy.select_action({**big, OBS_STATE: torch.zeros(1, 8), "task": [TASK]})


@pytest.mark.cpu
def test_env_needs_batch_size_one(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot_policy_reki import RekiEnv

    monkeypatch.setattr("sys.argv", ["lerobot-eval", "--env.type=reki"])
    with pytest.raises(ValueError, match="batch_size=1"):
        RekiEnv()
    monkeypatch.setattr("sys.argv", ["lerobot-eval", "--env.type=reki", "--eval.batch_size=1"])
    RekiEnv()


# ``--env.type=reki`` against rollout.py on fake sims and a fake model. The model input is the
# ``build_inference_sample`` call and the model output a torch.randn chunk, so equal calls and
# equal actions mean equal episodes, flow noise included.
HORIZON = 50  # 3 predicts per episode at 20 Hz


class _FakeModel(torch.nn.Module):
    """``RekiModel`` at inference: a chunk from the global torch RNG."""

    def __init__(self, chunk_size: int = 1) -> None:
        from types import SimpleNamespace

        super().__init__()
        self.config = SimpleNamespace(chunk_size=chunk_size)
        self._register_cache = None

    def reset_episode(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        return torch.randn(1, self.config.chunk_size, 32)


def _record_samples(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record each ``build_inference_sample`` call of rollout.py and of ``RekiPolicy``."""
    from reki.eval.rollout import loop
    from reki.inference import policy

    calls: list[dict[str, Any]] = []

    def record(obs: Any, **kw: Any) -> dict:
        skip = ("preprocessor", "train_cfg", "device")
        calls.append(
            {
                "images": [np.asarray(im) for im in obs.images],
                "state": obs.state_raw,
                "task": obs.task_text,
                **{k: v for k, v in kw.items() if k not in skip and v is not None},
            }
        )
        return {}

    monkeypatch.setattr(loop, "build_inference_sample", record)
    monkeypatch.setattr(policy, "build_inference_sample", record)
    return calls


def _rollout_py(monkeypatch: pytest.MonkeyPatch, adapter: Any, export: Path, out: Path) -> None:
    """rollout.py's loop on the export, 2 episodes per task, with the fake model."""
    from reki.eval.rollout import loop

    monkeypatch.setattr(loop, "build_policy", lambda c, d: _FakeModel())
    monkeypatch.setattr(loop, "load_pretrained_weights", lambda *a, **k: None)
    monkeypatch.setattr(loop, "build_eval_preprocessor", lambda c: None)
    monkeypatch.setattr(loop, "_prompt_debug_lines", lambda *a: "")
    args = loop.RolloutArgs(
        ckpt_dir=str(export), n_episodes=2, device="cpu", log_dir=str(out), compile=False
    )
    loop.run(adapter, args)


def _lerobot_eval(export: Path, env_cfg: Any, n_episodes: int) -> list[bool]:
    """lerobot-eval's loop (``eval_policy``) on ``env_cfg``, with the plugin around the fake
    model. Returns the success of each episode."""
    from dataclasses import replace

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    from lerobot_policy_reki import RekiPolicy
    from reki.inference.policy import RekiPolicy as InnerPolicy
    from reki.inference.spec import load_inference_spec
    from reki.policy.train_config import TrainConfig

    spec, stats = load_inference_spec(export)
    train_cfg = TrainConfig.from_checkpoint(export)
    model = _FakeModel(spec.chunk_size)
    inner = InnerPolicy(model, None, train_cfg, spec, stats, "cpu")  # type: ignore[arg-type]
    cfg = PreTrainedConfig.from_pretrained(export)
    # lerobot-eval reads it from --policy.path
    env_cfg = replace(env_cfg, n_action_steps=cfg.n_action_steps)
    vec = make_env(env_cfg, n_envs=1)["reki"][0]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, cfg)
    _, pre, post = _lerobot_pipelines(cfg, export, "cpu")
    policy = RekiPolicy(cfg, inner)
    info = eval_policy(vec, policy, env_pre, env_post, pre, post, n_episodes, start_seed=1000)
    vec.close()
    return [ep["success"] for ep in info["per_episode"]]


def _rollout_successes(adapter: Any, out: Path) -> list[bool]:
    """The success of each episode that rollout.py wrote, in the order of its units."""
    import json

    return [
        e["success"]
        for u in adapter.units()
        for e in json.loads((out / u.rel_dir / "results.json").read_text())["episodes"]
    ]


def _assert_same_calls(ours: list[dict], ref: list[dict], state_atol: float = 0.0) -> None:
    assert len(ours) == len(ref) > 0
    for a, b in zip(ours, ref, strict=True):
        assert a.keys() == b.keys()
        for x, y in zip(a["images"], b["images"], strict=True):
            assert np.array_equal(x, y)
        np.testing.assert_allclose(a["state"], b["state"], rtol=0, atol=state_atol)
        for k, v in b["state_stats"].items():
            assert torch.equal(a["state_stats"][k], v)
        rest = set(a) - {"images", "state", "state_stats"}
        assert {k: a[k] for k in rest} == {k: b[k] for k in rest}


def _tiny_frames(monkeypatch: pytest.MonkeyPatch, benchmark: str) -> None:
    """The env emits 24×32 images: the image size of the tiny export."""
    from dataclasses import replace

    from lerobot_policy_reki import env_reki

    sim = env_reki.SIMS[benchmark]
    tiny = replace(sim, images=dict.fromkeys(sim.images, (24, 32)))
    monkeypatch.setitem(env_reki.SIMS, benchmark, tiny)


SERVER_CAMERAS = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
)


def _serve(srv: Any, log: list, state_dim: int) -> None:
    """A fake sim server for one client. Its observation is a function of the reset seed and
    the step. A seed with seed % 4 == 1 (the 2nd episode of --seed 42) succeeds at step 33."""
    from reki.eval import sim_protocol

    def obs(seed: int, step: int) -> dict:
        rng = np.random.default_rng([seed, step])
        return {
            "images": {
                c: rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for c in SERVER_CAMERAS
            },
            "state": rng.uniform(-0.5, 0.5, state_dim).astype(np.float32),
            "instruction": f"task of seed {seed}",
        }

    conn, _ = srv.accept()
    seed = step = 0
    while True:
        req = sim_protocol.recv(conn)
        if req["cmd"] == "make":
            log.append(("make", req.get("task") or req.get("env_name")))
            sim_protocol.send(conn, {"max_episode_steps": HORIZON, "instruction_type": "unseen"})
        elif req["cmd"] == "reset":
            seed, step = req["seed"], 0
            log.append(("reset", seed))
            sim_protocol.send(conn, obs(seed, step))
        elif req["cmd"] == "step":
            step += 1
            log.append(("step", req["action"]))
            done = {
                "success": seed % 4 == 1 and step == 33,
                "terminated": False,
                "truncated": False,
            }
            sim_protocol.send(conn, {**obs(seed, step), **done})
        else:
            sim_protocol.send(conn, {"ok": True})
            break
    conn.close()
    srv.close()


def _start_server(log: list, state_dim: int) -> tuple[int, Any]:
    import threading

    from reki.eval import sim_protocol

    srv = sim_protocol.listen("127.0.0.1", 0, None)
    thread = threading.Thread(target=_serve, args=(srv, log, state_dim), daemon=True)
    thread.start()
    return srv.getsockname()[1], thread


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("dataset", "benchmark", "task", "state_dim"),
    [
        ("robotwin2_clean", "robotwin", "adjust_bottle", 14),
        ("robocasa365", "robocasa365", "CloseFridge", 16),
    ],
)
def test_sim_env_matches_rollout(
    tiny_lerobot_leaf: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    benchmark: str,
    task: str,
    state_dim: int,
) -> None:
    """``--env.type=reki`` on a sim server gives the sim commands, model inputs and successes of
    rollout.py: the same episode seeds, observations, prompts, cadence and actions."""
    from lerobot_policy_reki import RekiEnv
    from reki.eval.rollout import remote

    export = tmp_path / "export"
    _spec_export(tiny_lerobot_leaf, export, dataset, benchmark)
    _tiny_frames(monkeypatch, benchmark)
    calls = _record_samples(monkeypatch)

    ref_log: list = []
    port, server = _start_server(ref_log, state_dim)
    adapter_cls = {
        "robotwin": remote.RoboTwinAdapter,
        "robocasa365": remote.RoboCasa365Adapter,
    }[benchmark]
    adapter = adapter_cls(task, str(export), "127.0.0.1", port)
    _rollout_py(monkeypatch, adapter, export, tmp_path / "out")
    server.join(timeout=10)
    ref_calls = calls[:]
    calls.clear()

    log: list = []
    port, server = _start_server(log, state_dim)
    env_cfg = RekiEnv(
        benchmark=benchmark, task=task, n_episodes=2, sim_port=port, robotwin_seed_cache="none"
    )
    successes = _lerobot_eval(export, env_cfg, n_episodes=2)
    server.join(timeout=10)

    assert [cmd for cmd, _ in log].count("reset") == 2
    assert log == ref_log
    assert successes == _rollout_successes(adapter, tmp_path / "out")
    assert sorted(successes) == [False, True]
    _assert_same_calls(calls, ref_calls)


class _FakeLiberoEnv:
    """LeRobot's ``LiberoEnv`` for one task: each reset takes the next init state, also the reset
    it runs itself after a terminal step. Init states i with i % 3 != 2 succeed at step 13."""

    def __init__(self, log: list, task: str, task_id: int) -> None:
        self.log = log
        self.task_description = f"{task} task {task_id}"
        self._max_episode_steps = 25
        self.next_init = 10 * task_id
        self.init = self.step_i = 0

    def _raw(self) -> dict[str, Any]:
        rng = np.random.default_rng([self.init, self.step_i])
        quat = rng.normal(size=4)
        q = rng.uniform(0.0, 0.04)
        pixels = {k: rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for k in ("image", "image2")}
        return {
            "pixels": pixels,
            "robot_state": {
                "eef": {"pos": rng.uniform(-0.5, 0.5, 3), "quat": quat / np.linalg.norm(quat)},
                "gripper": {"qpos": np.asarray([q, -q])},
            },
        }

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self.init, self.step_i = self.next_init, 0
        self.next_init += 1
        self.log.append(("reset", seed, self.init))
        return self._raw(), {}

    def step(self, action: np.ndarray) -> tuple:
        self.step_i += 1
        self.log.append(("step", np.asarray(action).tolist()))
        success = self.init % 3 != 2 and self.step_i == 13
        raw = self._raw()
        if success:
            self.reset()
        return raw, 0.0, success, False, {"is_success": success}

    def close(self) -> None:
        pass


@pytest.mark.cpu
def test_libero_env_matches_rollout(
    tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--env.type=reki --env.benchmark=libero`` over 2 suites × 2 tasks gives the init states,
    episode seeds, model inputs and actions of rollout.py. The state matches to float32
    rounding: the plugin converts axis-angle to euler in torch, rollout.py in scipy."""
    from lerobot_policy_reki import RekiEnv
    from reki.eval.rollout import libero

    export = tmp_path / "export"
    _spec_export(tiny_lerobot_leaf, export, "libero", "libero")
    _tiny_frames(monkeypatch, "libero")
    calls = _record_samples(monkeypatch)
    log: list = []
    monkeypatch.setattr(
        libero, "make_libero_env", lambda task, task_id=0: _FakeLiberoEnv(log, task, task_id)
    )
    suites = ["libero_spatial", "libero_goal"]

    adapter = libero.LiberoAdapter(suites, [0, 1])
    _rollout_py(monkeypatch, adapter, export, tmp_path / "out")
    ref_log, ref_calls = log[:], calls[:]
    log.clear()
    calls.clear()

    env_cfg = RekiEnv(benchmark="libero", task=",".join(suites), task_ids=[0, 1], n_episodes=2)
    successes = _lerobot_eval(export, env_cfg, n_episodes=8)

    # the env's own reset after a success takes an init state, in both loops
    assert any(e[0] == "reset" and e[1] is None for e in log)
    assert log == ref_log
    assert successes == _rollout_successes(adapter, tmp_path / "out")
    assert 0 < sum(successes) < len(successes)
    _assert_same_calls(calls, ref_calls, state_atol=1e-6)
