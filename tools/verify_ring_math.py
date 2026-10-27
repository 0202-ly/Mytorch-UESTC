"""Check Ring reduce-scatter/all-gather against a direct gradient average."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from distributed.mytorch_ring_allreduce import _ring_sockets, _ring_grpc_links, ring_allreduce


def worker(rank: int, workers: int, port: int, transport: str, queue: mp.Queue) -> None:
    gradient = np.arange(17, dtype=np.float32) + rank * 2
    send, recv = (_ring_grpc_links(rank, workers, "127.0.0.1", port, 30)
                  if transport == "grpc_protobuf" else
                  _ring_sockets(rank, workers, "127.0.0.1", port, 30))
    try:
        result, exchanged = ring_allreduce(gradient, rank, workers, send, recv, transport)
        queue.put((rank, result, exchanged))
    finally:
        send.close()
        recv.close()


def main() -> None:
    for transport_index, transport in enumerate(("binary", "socket_json", "grpc_protobuf")):
      for workers, offset in ((2, 0), (4, 10)):
        port = 54400 + transport_index * 100 + offset
        queue = mp.Queue()
        processes = [mp.Process(target=worker, args=(rank, workers, port, transport, queue))
                     for rank in range(workers)]
        for process in processes:
            process.start()
        rows = [queue.get(timeout=35) for _ in processes]
        for process in processes:
            process.join(timeout=35)
            if process.exitcode != 0:
                raise RuntimeError(f"{workers}-worker Ring process failed: {process.exitcode}")
        expected = np.arange(17, dtype=np.float32) + workers - 1
        for rank, result, exchanged in rows:
            np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)
            print(f"transport={transport} workers={workers} rank={rank} max_abs_error="
                  f"{np.max(np.abs(result - expected)):.8f} bytes={exchanged}")


if __name__ == "__main__":
    main()
