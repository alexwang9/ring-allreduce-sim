from __future__ import annotations
import numpy as np
from typing import List
from .tensors import split_into_chunks, merge_chunks

class RingNode:
    def __init__(self, rank: int, world_size: int, vector: np.ndarray):
        self.rank = rank
        self.world_size = world_size
        # local chunks of my vector
        self.chunks = split_into_chunks(vector, world_size)
        # running sums for reduce-scatter (initialized with my chunks)
        self.accum = [c.copy() for c in self.chunks]

    def left(self) -> int:
        return (self.rank - 1) % self.world_size
    
    def right(self) -> int:
        return (self.rank + 1) % self.world_size

def reduce_scatter(nodes: List[RingNode]) -> List[np.ndarray]:
    """
    Perform in-memory ring reduce-scatter
    After this, node i holds the SUM of chunk i over all nodes.
    Returns the list of per-node reduced chunks (sums).
    """
    N = len(nodes)
    # Send buffers: each node starts by sending chunk indexed by (rank - 0) mod N
    send_idx = [n.rank for n in nodes] # what chunk each node sends this step

    # Perform N-1 steps
    for step in range(N - 1):
        # Step 1: prepare outgoing copies
        outgoing = [nodes[i].accum[send_idx[i]].copy() for i in range(N)]

        # Step 2: deliver to right neighbors
        # right neighbor receives this chunk as "incoming"
        incoming = [None] * N
        for i in range(N):
            r = nodes[i].right()
            incoming[r] = outgoing[i]
        
        # Step 3: reduce (add) incoming into my accumulator at that index
        for i in range(N):
            idx = send_idx[(i - 1) % N] # the index that arrived at me
            nodes[i].accum[idx] += incoming[i]

        # Step 4: rotate which chunk each node will send next
        for i in range(N):
            send_idx[i] = (send_idx[i] - 1) % N
        
    # After N-1 steps, node i holds SUM for chunk i
    reduced_chunks = [nodes[i].accum[i] for i in range(N)]
    return reduced_chunks

def allgather(nodes: List[RingNode], my_chunk_sums: List[np.ndarray]) -> List[np.ndarray]:
    """
    Allgather the reduced chunks so every node gets all chunks.
    Input: my_chunk_sums[i] is the reduced chunk that belongs to index i (held by node i).
    Output: list of full vectors per node (averaged is done by caller).
    """
    N = len(nodes)
    # Each node starts with its own chunk i
    have = [[None] * N for _ in range(N)]
    for i in range(N):
        have[i][i] = my_chunk_sums[i].copy()
    
    # circulate N-1 times around the ring
    for step in range(N - 1):
        outgoing = [None] * N
        # Each node sends the latest chunk it learned in previous step
        for i in range(N):
            # find which indices I currently have; send the one with index (i - step) mod N
            idx_to_send = (i - step) % N
            outgoing[i] = have[i][idx_to_send].copy()
        
        incoming = [None] * N
        for i in range(N):
            r = nodes[i].right()
            incoming[r] = outgoing[i]
        
        # Store the received chunk at the correct index for each node
        for i in range(N):
            idx_received = (i - step - 1) % N
            have[i][idx_received] = incoming[i]
        
    # Now each node has all N chunks; merge them to full vectors
    full_vectors = [merge_chunks(have[i]) for i in range(N)]
    return full_vectors

def ring_allreduce(vectors: List[np.ndarray], average: bool = True) -> List[np.ndarray]:
    """In-memory ring-allreduce across vectors from N nodes."""
    N = len(vectors)
    nodes = [RingNode(i, N, vectors[i]) for i in range(N)]
    # 1) reduce-scatter to get sums for each chunk index at its owner node
    reduced_per_owner = reduce_scatter(nodes) # list length N, chunk i at node i
    # 2) (optionally) average the reduced chunks
    if average:
        for i in range(N):
            reduced_per_owner[i] /= N
    
    # 3) allgather so every node gets all averaged chunks
    full = allgather(nodes, reduced_per_owner)
    return full