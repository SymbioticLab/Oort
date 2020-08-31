from mpyc.runtime import mpc
from mpyc.random import *
import time
import numpy as np

K = 30
N = 1000
population = [i for i in range(N)]
print(len(population))

# Start MPC clients
secint = mpc.SecInt()
mpc.run(mpc.start())

mu, sigma = 0.5, 0.1
weights = np.random.normal(mu, sigma, N).tolist()
weights = [int(i * 100) for i in weights]

start_time = time.time()
print(f"Sampling {K} numbers of out {N} numbers uniformly")
rand = mpc.run(mpc.output(choices(secint, population, k = K)))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(f"Sampling {K} numbers of out {N} numbers with weights")
rand = mpc.run(mpc.output(choices(secint, population, weights, k = K)))
print("--- %s seconds ---" % (time.time() - start_time))

