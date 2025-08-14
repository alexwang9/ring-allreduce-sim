import numpy as np
from typing import List, Tuple

def split_into_chunks(arr: np.ndarray, num_chunks: int) -> List[np.ndarray]:
	"""Split 1D array into num_chunks nearly-equal chunks"""
	n = arr.shape[0]
	base = n // num_chunks
	rem = n % num_chunks;
	chunks = []
	start = 0
	for i in range(num_chunks):
		size = base + (1 if i < rem else 0)
		end = start + size
		chunks.append(arr[start:end].copy())
		start = end
	return chunks

def merge_chunks(chunks: List[np.ndarray]) -> np.ndarray:
	"""Concatenate chunks back to one 1D array."""
	return np.concatenate(chunks, axis=0)

def stack_and_average(vectors: List[np.ndarray]) -> np.ndarray:
	"""Ground-truth average across nodes (for verification)."""
	return np.mean(np.stack(vectors, axis=0), axis=0)

def random_grad(length: int, seed: int | None = None) -> np.ndarray:
	rng = np.random.default_rng(seed)
	return rng.standard_normal(length, dtype=np.float32)