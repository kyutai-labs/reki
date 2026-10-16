"""``--num_workers K``: K rollout processes on one GPU that pull units from a shared claim dir.

Each worker runs ``scripts/eval/rollout.py`` again with ``--num_workers 1`` over the full unit
grid and a shared ``--_claim_dir``. A unit runs on the first worker that claims it. Every unit
reseeds its own rng, so the results are bit-identical to one worker. A socket benchmark needs
one sim server per worker, because a server serves one client at a time. Pass
``--sim_server_cmd`` to start the servers here, or one ``--sim_port`` per running server.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class Claims:
    """Unit ownership among the workers. Without a claim dir (one worker) every unit belongs to
    this worker."""

    def __init__(self, claim_dir: str | None) -> None:
        self.dir = None if claim_dir is None else Path(claim_dir)

    def take(self, unit: str) -> bool:
        """Claim ``unit`` (atomic file create); False when another worker has it."""
        if self.dir is None:
            return True
        try:
            os.close(os.open(self.dir / unit.replace("/", "_"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return False
        return True


def _strip_cli_flags(argv: list[str], flags: set[str]) -> list[str]:
    """``argv`` without the ``--flag=v`` / ``--flag v`` occurrences of ``flags``."""
    out: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok.split("=", 1)[0] in flags:
            skip_next = "=" not in tok
            continue
        out.append(tok)
    return out


def await_sim_port(proc: subprocess.Popen, port_file: Path, timeout_s: float = 600.0) -> int:
    """The port a started sim server wrote to its port file. Raises if the server exits or
    writes nothing within ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return int(port_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            pass
        if proc.poll() is not None:
            raise SystemExit(
                f"[shard] sim server exited with code {proc.returncode} before writing {port_file}"
            )
        time.sleep(1.0)
    proc.terminate()
    raise SystemExit(f"[shard] sim server wrote no port to {port_file} within {timeout_s:.0f}s")


def check_sharding(
    socket_bench: bool, num_workers: int, sim_server_cmd: str | None, sim_ports: list[int]
) -> None:
    """Fail on a worker / sim-server combination that cannot run."""
    if not socket_bench:
        if sim_server_cmd is not None or len(sim_ports) > 1:
            raise SystemExit(
                "--sim_server_cmd / several --sim_port are for socket benchmarks; "
                "LIBERO runs its env in-process"
            )
        return
    if sim_server_cmd is not None and len(sim_ports) > 1:
        raise SystemExit("pass --sim_server_cmd (local servers) or several --sim_port, not both")
    if len(sim_ports) > 1 and len(sim_ports) != num_workers:
        raise SystemExit(f"got {len(sim_ports)} sim ports for --num_workers {num_workers}")
    if num_workers > 1 and sim_server_cmd is None and len(sim_ports) == 1:
        raise SystemExit(
            "--num_workers > 1 on a socket benchmark needs one sim server per worker: pass "
            "--sim_server_cmd (with a {portfile} placeholder) or one --sim_port per worker"
        )


DEFAULT_COMPILE_CACHE_DIR = Path.home() / ".cache" / "reki" / "inductor"


def set_compile_cache_dir(path: str | None) -> Path:
    """Set the Inductor compile cache to ``path``, else ``$TORCHINDUCTOR_CACHE_DIR``, else
    ``DEFAULT_COMPILE_CACHE_DIR``, and return it. Every process of the eval and later evals use
    it. The Inductor default is under /tmp, which is local to the node and can be wiped."""
    cache = Path(
        path or os.environ.get("TORCHINDUCTOR_CACHE_DIR") or DEFAULT_COMPILE_CACHE_DIR
    ).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    # One Triton cache per eval: evals on one node that unpack the same kernel from the
    # Inductor cache into a shared Triton cache race on the rename (Errno 39).
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="reki_triton_")
    return cache


def spawn_workers(
    units: list[str],
    num_workers: int,
    sim_server_cmd: str | None = None,
    remote_ports: list[int] | None = None,
    warm_compile: bool = False,
) -> None:
    """Run the workers (and their local sim servers) to completion. Raises if a worker fails.
    With ``warm_compile``, one process first fills the compile cache (``--_warm_compile_cache``)
    and the workers then load the compiled graphs."""
    n_procs = min(num_workers, len(units))
    base_argv = _strip_cli_flags(
        sys.argv[1:], {"--num_workers", "--_claim_dir", "--sim_server_cmd", "--sim_port"}
    )
    if warm_compile and n_procs > 1:
        print("[shard] filling the compile cache in one process")
        warm = [sys.executable, sys.argv[0], *base_argv, "--num_workers", "1"]
        if code := subprocess.run([*warm, "--_warm_compile_cache"]).returncode:
            raise SystemExit(f"[shard] the compile warm-up exited with code {code}")
    claim_dir = tempfile.mkdtemp(prefix="eval_claims_")
    servers: list[subprocess.Popen] = []
    port_files: list[Path] = []
    pf_dir: Path | None = None
    ports: list[int | None] = list(remote_ports[:n_procs]) if remote_ports else [None] * n_procs
    procs: list[subprocess.Popen] = []
    try:
        if sim_server_cmd is not None:
            if "{portfile}" not in sim_server_cmd:
                raise SystemExit("--sim_server_cmd must contain a {portfile} placeholder")
            pf_dir = Path(tempfile.mkdtemp(prefix="eval_simports_"))
            port_files = [pf_dir / f"port_{i}" for i in range(n_procs)]
            servers = [
                subprocess.Popen(
                    sim_server_cmd.replace("{portfile}", str(pf)),
                    shell=True,
                    executable="/bin/bash",
                    start_new_session=True,
                )
                for pf in port_files
            ]
            ports = [await_sim_port(sp, pf) for sp, pf in zip(servers, port_files)]
        print(f"[shard] {len(units)} units over {n_procs} worker(s) (claims in {claim_dir})")
        for i in range(n_procs):
            port_argv = [] if ports[i] is None else ["--sim_port", str(ports[i])]
            print(f"[shard]   worker {i}" + (f" (sim port {ports[i]})" if port_argv else ""))
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        sys.argv[0],
                        *base_argv,
                        "--num_workers",
                        "1",
                        "--_claim_dir",
                        claim_dir,
                        *port_argv,
                    ]
                )
            )
        codes = [p.wait() for p in procs]
    finally:
        shutil.rmtree(claim_dir, ignore_errors=True)
        for sp in servers:
            if sp.poll() is None:
                try:
                    os.killpg(sp.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for sp in servers:
            try:
                sp.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        for i, pf in enumerate(port_files):
            log = pf.with_name(pf.name + ".log")
            if log.exists() and log.stat().st_size:
                tail = log.read_text(errors="replace").splitlines()[-100:]
                print(f"\n=== sim server {i} log [tail] ({log}) ===")
                print("\n".join(tail))
        if pf_dir is not None:
            shutil.rmtree(pf_dir, ignore_errors=True)
    bad = [c for c in codes if c != 0]
    if bad:
        raise SystemExit(f"[shard] {len(bad)}/{len(procs)} workers failed (exit codes {codes})")
    print(f"[shard] all {len(procs)} workers finished")
