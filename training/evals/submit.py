# submit job to the remote cluster

import yaml
import sys
import time
import random
import os

def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def process_cmd(yaml_file):

    yaml_conf = load_yaml_conf(yaml_file)

    ps_ip = yaml_conf['ps_ip']
    worker_ips, total_gpus = [], []
    
    for ip_gpu in yaml_conf['worker_ips']:
        ip, num_gpu = ip_gpu.strip().split(':')
        worker_ips.append(ip)
        total_gpus.append(int(num_gpu))

    time_stamp = int(time.time())

    job_conf = {'time_stamp':time_stamp, 
            'total_worker': sum(total_gpus),
            'ps_port':random.randint(1000, 60000), 
            'manager_port':random.randint(1000, 60000)}

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''
    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'

    # =========== Submit job to parameter server ============
    ps_cmd = f"python {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 "
    print(f"ssh {yaml_conf['auth']['ssh_user']}@{ps_ip} {ps_cmd}")

    time.sleep(2)
    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        for _  in range(gpu):
            time.sleep(1)
            worker_cmd = f"python {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} "
            rank_id += 1

            print(f"ssh {yaml_conf['auth']['ssh_user']}@{worker} {worker_cmd}")
            
process_cmd('cluster.yaml')


