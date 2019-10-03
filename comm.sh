export OMP_NUM_THREADS=1
taskset -c 1 python3 /users/fanlai/SSPTorch/param_server.py --ps_ip=10.0.0.1 --ps_port=29500 --this_rank=0 --learners=1-2-3-4 --epochs=500 --model=vgg --depth=19 --stale_threshold=1

