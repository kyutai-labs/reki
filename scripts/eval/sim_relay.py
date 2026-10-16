#!/usr/bin/env python3
"""TCP relay: listen on this machine's network address and forward to a loopback port.

Compute nodes cannot reach the loopback of the Slurm submit node, where the SSH tunnel to the
sim machine ends. Submitting an eval job with `--sim_port` and `--sim_relay=True`
(`launchers/sft.py select <i> eval ... --sim_port <ports> --sim_relay=True submit`) starts one relay
per port.
To run one yourself:

    python3 scripts/eval/sim_relay.py 29747 29747   # <listen_port> <to_port>

The sim protocol is unauthenticated pickle: every machine that reaches <listen_host> can run
code on the sim machine. Stop the relay when the evaluation ends. See docs/evaluation.md.
Stdlib only.
"""

from __future__ import annotations

import argparse
import socket
import threading


def cluster_address() -> str:
    """The address that other nodes resolve this host name to."""
    return socket.gethostbyname(socket.gethostname())


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    """Copy ``src`` to ``dst``, then pass the end of stream on (half-close)."""
    try:
        while chunk := src.recv(65536):
            dst.sendall(chunk)
    except OSError:
        pass
    try:
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def _handle(client: socket.socket, to_host: str, to_port: int) -> None:
    try:
        upstream = socket.create_connection((to_host, to_port))
    except OSError as e:
        print(f"[relay] {to_host}:{to_port} unreachable: {e}", flush=True)
        client.close()
        return
    pipes = [
        threading.Thread(target=_pipe, args=(client, upstream), daemon=True),
        threading.Thread(target=_pipe, args=(upstream, client), daemon=True),
    ]
    for t in pipes:
        t.start()

    def _close() -> None:
        for t in pipes:
            t.join()
        client.close()
        upstream.close()

    threading.Thread(target=_close, daemon=True).start()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("listen_port", type=int)
    p.add_argument("to_port", type=int)
    p.add_argument("--listen_host", default=None, help="default: the cluster address")
    p.add_argument("--to_host", default="127.0.0.1")
    args = p.parse_args()
    listen_host = args.listen_host or cluster_address()
    if listen_host.startswith("127."):
        raise SystemExit(f"{listen_host} is a loopback address: pass --listen_host")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((listen_host, args.listen_port))
    srv.listen(16)
    print(f"[relay] {listen_host}:{args.listen_port} -> {args.to_host}:{args.to_port}", flush=True)
    while True:
        client, _ = srv.accept()
        _handle(client, args.to_host, args.to_port)


if __name__ == "__main__":
    main()
