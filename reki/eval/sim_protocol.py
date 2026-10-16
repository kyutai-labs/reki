"""Wire format between ``scripts/eval/rollout.py`` and the sim servers
(``scripts/eval/*_sim_server.py``): one request gives one reply, each a pickled dict after its
8-byte big-endian length.

Standard library only: the servers run in their own Python envs and import this file by name
(they append ``reki/eval/`` to ``sys.path``). Pickle runs code when it loads, so a server binds
loopback only; reach a remote server through an SSH tunnel.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import traceback
from collections.abc import Callable, Mapping

_LEN = struct.Struct(">Q")
# Far above any real message (a 3-camera 640x480 observation is ~3 MB). A non-protocol probe
# such as HTTP "GET /" reads as a length of ~5e18, and recv() would raise MemoryError.
MAX_FRAME = 64 << 20
Handler = Callable[[dict], dict]


class BadRequest(Exception):
    """A whole frame arrived but is not a dict message; the stream is still in sync."""


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("sim socket closed")
        buf += chunk
    return buf


def send(conn: socket.socket, msg: dict) -> None:
    """Send one length-prefixed pickled message."""
    body = pickle.dumps(msg)
    conn.sendall(_LEN.pack(len(body)) + body)


def recv(conn: socket.socket) -> dict:
    """The next message. Raises ``ConnectionError`` when the peer closed the socket or sent a
    frame over ``MAX_FRAME``, and ``BadRequest`` when the frame is not a pickled dict."""
    (n,) = _LEN.unpack(_recv_exact(conn, _LEN.size))
    if n > MAX_FRAME:
        raise ConnectionError(f"frame of {n} bytes is not this protocol")
    body = _recv_exact(conn, n)
    try:
        msg = pickle.loads(body)
    except Exception as e:  # noqa: BLE001 — garbage can raise almost any type
        raise BadRequest(f"undecodable message: {type(e).__name__}: {e}") from e
    if not isinstance(msg, dict):
        raise BadRequest(f"message must be a dict, got {type(msg).__name__}")
    return msg


def serve(conn: socket.socket, handlers: Mapping[str, Handler], name: str) -> None:
    """Answer requests until the client sends close (replied {"ok": True}, then return). A
    handler that raises, or an unknown cmd, gets {"error", "traceback"}; the connection stays
    usable, as after a ``BadRequest`` frame. Raises ``OSError`` (``ConnectionError`` among them)
    when the client goes away or sends a frame over ``MAX_FRAME``."""
    while True:
        try:
            req = recv(conn)
        except BadRequest as e:
            print(f"[{name}] bad request: {e}", flush=True)
            send(conn, {"error": f"BadRequest: {e}", "traceback": ""})
            continue
        cmd = req.get("cmd")
        if cmd == "close":
            send(conn, {"ok": True})
            return
        try:
            handler = handlers.get(cmd) if isinstance(cmd, str) else None
            if handler is None:
                raise ValueError(f"unknown cmd {cmd!r}")
            reply = handler(req)
        except Exception as e:  # noqa: BLE001 — the client gets the error
            tb = traceback.format_exc()
            print(f"[{name}] request failed: {e!r}\n{tb}", flush=True)
            reply = {"error": f"{type(e).__name__}: {e}", "traceback": tb}
        send(conn, reply)


def add_server_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--host`` / ``--port`` / ``--port-file`` arguments of a simulator server."""
    parser.add_argument("--host", default="127.0.0.1", help="bind address; keep it loopback")
    parser.add_argument(
        "--port", type=int, default=0, help="0 = ephemeral, reported via --port-file"
    )
    parser.add_argument(
        "--port-file", default=None, help="written with the bound port once listening"
    )


class SimServerError(RuntimeError):
    """The server answered a request with an error reply; the connection stays usable."""

    def __init__(self, error: str, request: dict, traceback: str) -> None:
        super().__init__(f"sim server error on {request}: {error}\n{traceback}")
        self.error = error


def rpc(conn: socket.socket, msg: dict) -> dict:
    """Send ``msg`` and return the reply. Raises ``SimServerError`` on an error reply."""
    send(conn, msg)
    reply = recv(conn)
    if "error" in reply:
        request = {k: v for k, v in msg.items() if k in ("cmd", "task", "seed", "trust")}
        raise SimServerError(reply["error"], request, reply.get("traceback", ""))
    return reply


def listen(host: str, port: int, port_file: str | None) -> socket.socket:
    """A listening socket (one client at a time). ``port`` 0 takes a free port; the bound port
    is written to ``port_file`` once listening (the client's readiness signal)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    port = srv.getsockname()[1]
    if port_file:
        with open(port_file, "w") as f:
            f.write(str(port))
    print(f"[sim server] listening on {host}:{port}", flush=True)
    return srv
