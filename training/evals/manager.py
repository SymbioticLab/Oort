
# Submit job to the remote cluster

import yaml
import sys
import time
import random
import os, subprocess
import pickle

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
    running_vms = set()
    job_name = 'kuiper_job'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp':time_stamp, 
            'total_worker': sum(total_gpus),
            'ps_port':random.randint(1000, 60000), 
            'manager_port':random.randint(1000, 60000)}

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''
    cmd_prefix = ''
    if yaml_conf['setup_commands'] is not None:
        cmd_prefix = yaml_conf['setup_commands'][0] + ' & ' 
        for item in yaml_conf['setup_commands'][1:]:
            cmd_prefix += (item + ' & ')

    cmd_sufix = f" & sleep {yaml_conf['max_duration']}"


    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]

    learner_conf = '-'.join([str(_) for _ in list(range(1, sum(total_gpus)+1))])
    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    ps_cmd = cmd_prefix + \
            f" python {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 --learner={learner_conf} " + \
            cmd_sufix

    print(f"ssh {submit_user}{ps_ip} {ps_cmd}")
    #subprocess.Popen(f"ssh {submit_user}{ps_ip} {ps_cmd}", shell=True)

    #time.sleep(2)
    worker_cmds = []
    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        for _  in range(gpu):
            #time.sleep(1)
            worker_cmd = cmd_prefix + \
                    f" python {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} --learner={learner_conf} " + \
                    cmd_sufix
            rank_id += 1
            worker_cmds.append({'ip':worker, 'cmds': worker_cmd, 'user':yaml_conf['auth']['ssh_user']})
            print(f"ssh {submit_user}{worker} {worker_cmd}")
            #subprocess.Popen(f"ssh {submit_user}{worker} {worker_cmd}", shell=True)

    # dump the address of running workers
    with open(job_name, 'wb') as fout:
        job_meta = {'user':submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    #return ps_cmd, worker_cmds

def terminate(job_name):
    if not os.path.isfile(job_name):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_name, 'rb') as fin:
        job_meta = pickle.load(fin)

    scripts = f"""os.system('ps -ef | grep "python" | grep "job_name={job_name}" > temp')\nids=[l.split()[1] for l in open('temp').readlines()]\n[os.system("kill -9 "+id) for id in ids]"""
    for vm_ip in job_meta['vms']:
        # subprocess.Popen(f'ssh {job_meta["user"]}@{vm_ip} python -c """{scripts}"""')
        print(f'ssh {job_meta["user"]}{vm_ip} python -c """{scripts}"""')

if sys.argv[1] == 'submit':
    process_cmd(sys.argv[2])
elif sys.argv[1] == 'stop':
    terminate(sys.argv[2])
else:
    print("Unknown cmds ...")



