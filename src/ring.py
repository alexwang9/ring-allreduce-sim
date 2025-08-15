# src/ring.py
from __future__ import annotations
import numpy as np
from typing import List
from .tensors import split_into_chunks, merge_chunks

class RingNode:
    def __init__(self, rank: int, world_size: int, vector: np.ndarray):
        self.rank = rank
        self.world_size = world_size
        # Start with my local chunks; accumulators hold running sums
        self.accum = split_into_chunks(vector, world_size)

    def left(self) -> int:
        return (self.rank - 1) % self.world_size

    def right(self) -> int:
        return (self.rank + 1) % self.world_size


def _reduce_scatter(nodes: List[RingNode]) -> List[np.ndarray]:
    """
    In-memory ring reduce-scatter.
    After N-1 steps, node i holds the *sum-reduced* chunk with index (i + 1) % N.
    Returns a list 'reduced' where reduced[i] is the chunk held by node i.
    """
    N = len(nodes)

    # Perform N-1 ring steps
    for s in range(N - 1):
        # Node i sends chunk index k_send = (i - s) mod N to its right neighbor
        send_idx = [ (i - s) % N for i in range(N) ]

        # Deliver outgoing data to the right neighbor
        incoming: list[tuple[int, np.ndarray]] = [None] * N  # (chunk_index, data)
        for i in range(N):
            r = (i + 1) % N
            k = send_idx[i]
            incoming[r] = (k, nodes[i].accum[k].copy())

        # Each node receives chunk index k_recv and accumulates (adds) it
        for i in range(N):
            k_recv, data = incoming[i]
            nodes[i].accum[k_recv] += data

    # IMPORTANT: ownership after reduce-scatter:
    # node i now owns the reduced sum for chunk index (i + 1) % N
    owner_index = [ (i + 1) % N for i in range(N) ]
    reduced = [ nodes[i].accum[owner_index[i]] for i in range(N) ]
    return reduced


def _allgather(nodes: List[RingNode], owner_chunks: List[np.ndarray]) -> List[np.ndarray]:
    """
    In-memory ring allgather.
    Input: owner_chunks[i] is the averaged chunk whose *index* is (i + 1) % N and currently resides at node i.
    Output: full vectors for each node (all chunks gathered and merged).
    """
    N = len(nodes)

    # have[i][k] = chunk with index k currently known by node i (may start as None)
    have: list[list[np.ndarray | None]] = [[None] * N for _ in range(N)]

    # Seed: node i starts with the chunk whose index is (i + 1) % N
    for i in range(N):
        idx = (i + 1) % N
        have[i][idx] = owner_chunks[i].copy()

    # Circulate chunks N-1 times
    for s in range(N - 1):
        outgoing = [None] * N
        # Node i sends the chunk whose index (i + 1 - s) % N (the most recently "advanced" index)
        for i in range(N):
            idx_to_send = ( (i + 1) - s ) % N
            outgoing[i] = (idx_to_send, have[i][idx_to_send].copy())

        # Deliver to right neighbor
        incoming: list[tuple[int, np.ndarray]] = [None] * N
        for i in range(N):
            r = (i + 1) % N
            incoming[r] = outgoing[i]

        # Each node stores the received chunk by its *index*
        for i in range(N):
            idx_recv, data = incoming[i]
            have[i][idx_recv] = data

    # Merge chunks back to full vectors
    full_vectors = [ merge_chunks(have[i]) for i in range(N) ]
    return full_vectors


def ring_allreduce(vectors: List[np.ndarray], average: bool = True) -> List[np.ndarray]:
    """
    High-level in-memory ring allreduce:
      1) reduce-scatter to sum chunks,
      2) (optional) divide by N to average,
      3) allgather to broadcast all chunks to everyone.
    """
    N = len(vectors)
    nodes = [RingNode(i, N, vectors[i]) for i in range(N)]

    # 1) reduce-scatter: node i ends with reduced chunk index (i + 1) % N
    reduced = _reduce_scatter(nodes)

    # 2) average once
    if average:
        for i in range(N):
            reduced[i] = reduced[i] / N

    # 3) allgather to reconstruct full averaged vector at each node
    full = _allgather(nodes, reduced)
    return full
