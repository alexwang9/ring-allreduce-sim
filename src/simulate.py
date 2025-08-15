import numpy as np
from .tensors import random_grad, stack_and_average
from .ring import ring_allreduce

def run_demo(world_size=4, length=23, seed=0):
  # 1) create fake gradients - one vector per node
  vecs = [random_grad(length, seed + i) for i in range(world_size)]

  # 2) ground truth average
  truth = stack_and_average(vecs)

  # 3) ring-allreduce
  reduced = ring_allreduce(vecs, average=True)

  # 4) verify: every node should get the same vector = truth
  for i, out in enumerate(reduced):
    ok = np.allclose(out, truth, rtol=1e-5, atol=1e-6)
    print(f"Node {i} match truth? {ok}")
    if not ok:
      # Print a small diff summary if mismatch
      max_abs = np.max(np.abs(out - truth))
      print(f"  Max abs diff: {max_abs:.3e}")

if __name__ == "__main__":
  run_demo()