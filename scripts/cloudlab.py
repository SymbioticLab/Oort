import sys, os
from fabric import Config, Connection
import threadpool, time

vms = ['dc3master', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10']
numOfWorkers = len(vms) - 1
prefix = "." + exp + ".gaia-pg0.clemson.cloudlab.us"

# get the join of parameters
params = ' '.join(sys.argv[2:]) + learner + ' '

# deal with ps
rawCmdPs = '\npython ~/DMFL/param_server.py --ps_ip=10.0.0.253 --model=squeezenet1_1 --epochs=20000 --upload_epoch=50  --dump_epoch=500 --learning_rate=0.005 --decay_epoch=50 --model_avg=True --batch_size=256 --this_rank=0 '
            + params
            
learn = ' --learners='
for i in range(1, numOfWorkers):
    learn = learn + str(i) + '-'
learn = learn + str(numOfWorkers)

scriptPS = rawCmdPs + learn

# for workers
rawCmd = '\npython ~/DMFL/learner.py --ps_ip=10.0.0.253 --model=squeezenet1_1 --epochs=20000 --upload_epoch=50  --dump_epoch=500 --learning_rate=0.005 --decay_epoch=50 --model_avg=True --batch_size=256 '

def start_eval(url):
    print('....', url)
    c = Connection(url)

    try:
        if 'master' in url:
            c.run(scriptPS)
        else:
            time.sleep(5)
            rankId = ' --this_rank=' + url.replace(prefix, '').replace('h','')
            runCmd = rawCmd + params + rankId
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

        c.close()
    except Exception as e:
        print (str(e))
        pass

    print('Done ....', url)

def exec_cmd(mode):
    print('Exec on masters and workers')

    if mode == 'training':
        conn_job = threadpool.makeRequests(start_eval, vms)
        [pool.putRequest(req) for req in conn_job]

    elif mode == 'shutdown':
        conn_job = threadpool.makeRequests(shutdown, vms)
        [pool.putRequest(req) for req in conn_job]

    pool.wait()

if __name__ == "__main__":
    exec_cmd(sys.argv[1])

