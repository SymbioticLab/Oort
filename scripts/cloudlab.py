import sys, os
from fabric import Config, Connection
import threadpool, time

vms = ['dc3master', 'h1', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10']#, 'h11', 'h12']
exp='fame-sim'
prefix = "." + exp + ".gaia-pg0.clemson.cloudlab.us"
pool = threadpool.ThreadPool(50)

numOfWorkers = len(vms) - 1 if len(sys.argv[1].split('_')) == 1 else int(sys.argv[1].split('_')[1])
# get the join of parameters
learn = ' --learners='
for i in range(1, numOfWorkers):
    learn = learn + str(i) + '-'
learn = learn + str(numOfWorkers)

params = ' '.join(sys.argv[2:]) + ' ' + learn + ' '

# deal with ps
rawCmdPs = 'python3 ~/DMFL/param_server.py --ps_ip=10.0.0.253 --model=squeezenet1_1 --epochs=20000 --upload_epoch=50  --dump_epoch=500 --learning_rate=0.005 --decay_epoch=50 --model_avg=True --batch_size=256 --this_rank=0 ' + params

scriptPS = rawCmdPs

# for workers
rawCmd = 'python3 ~/DMFL/learner.py --ps_ip=10.0.0.253 --model=squeezenet1_1 --epochs=20000 --upload_epoch=50  --dump_epoch=500 --learning_rate=0.005 --decay_epoch=50 --model_avg=True --batch_size=256 '

def start_eval(_args):
    items = _args.split('__')
    url = items[0]
    hostId = 0

    if len(items) > 1:
        hostId = int(items[1])

    print('....', _args)
    c = Connection(url)

    try:
        if 'master' in url:
            c.run(scriptPS, hide=True)
        else:
            time.sleep(5)

            rankId = ' --this_rank=' + str(hostId)
            runCmd = rawCmd + params + rankId

            print(runCmd)
            if hostId != 1:
                c.run(runCmd, hide=True)
            else:
                c.run(runCmd)

        c.close()

    except Exception as e:
        print (str(e))
        pass

    print('Done ....', url)

def shutdown(url):
    print('....', url)
    c = Connection(url)

    try:
        # kill all thread
        try:
            c.run("kill $(ps aux | grep 'param_server.py' | awk '{print $2}')")
        except:
            pass

        try:
            c.run("kill $(ps aux | grep 'learner.py' | awk '{print $2}')")
        except:
            pass
        c.run('kill -9 -$(jobs -p)')
        c.close()
    except Exception as e:
        print (str(e))
        pass

    print('Done ....', url)

def exec_cmd(_mode):
    print('Exec on masters and workers')

    modes = _mode.split('_')
    mode = modes[0]

    if mode == 'training':
        vmsToRun = vms[0]
        workersets = vms[1:]
        workersToRun = [vms[0]]

        if len(modes) > 1:
            numOfClients = int(modes[1])

            for hostId in range(numOfClients):
                workersToRun.append(workersets[hostId%len(workersets)] + '__' + str(hostId + 1))
        else:
            for hostId, vm in enumerate(workersets):
                workersToRun.append(vm + '__' + str(hostId + 1))

        conn_job = threadpool.makeRequests(start_eval, workersToRun)
        [pool.putRequest(req) for req in conn_job]

    elif mode == 'shutdown':
        conn_job = threadpool.makeRequests(shutdown, vms)
        [pool.putRequest(req) for req in conn_job]

    pool.wait()

if __name__ == "__main__":
    exec_cmd(sys.argv[1])
