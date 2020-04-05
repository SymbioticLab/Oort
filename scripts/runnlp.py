import sys, os, time, datetime


paramsCmd = ' --ps_ip=10.255.11.92 --model=albert-base-v2 --epochs=20000 --upload_epoch=20  --dump_epoch=2 --learning_rate=4e-3 --min_learning_rate=1e-4 --decay_epoch=50 --model_avg=True '


os.system("bhosts > vms")
os.system("rm *.o")
os.system("rm *.e")

avaiVms = {}
blacklist = set()

with open('blacklist', 'r') as fin:
    for v in fin.readlines():
        blacklist.add(v.strip())

threadQuota = 10

with open('vms', 'r') as fin:
    lines = fin.readlines()
    for line in lines:
        if 'gpu-cn0' in line:
            # and 'gpu-cn011' not in line and 'gpu-cn008' not in line and 'gpu-cn001' not in line and 'gpu-cn012' not in line and 'gpu-cn005' not in line and 'gpu-cn006' not in line and 'gpu-cn004' not in line: #and 'gpu-cn012' not in line and 'gpu-cn011' not in line:
            items = line.strip().split()
            if items[0] not in blacklist:
                status = items[1]
                threadsGpu = int(items[5])

                if status == "ok" and (40-threadsGpu) >= threadQuota:
                    avaiVms[items[0]] = 40 - threadsGpu

# remove all log files, and scripts first
files = [f for f in os.listdir('.') if os.path.isfile(f)]

for file in files:
    if 'learner' in file or 'server' in file:
        os.remove(file)
        
# get the number of workers first
numOfWorkers = int(sys.argv[1])
learner = ' --learners=1'

for w in range(2, numOfWorkers+1):
    learner = learner + '-' + str(w)

# load template
with open('template.lsf', 'r') as fin:
    template = ''.join(fin.readlines())

_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
timeStamp = str(_time_stamp) + '_'
jobPrefix = 'learner' + timeStamp

# get the join of parameters
params = ' '.join(sys.argv[2:]) + learner + ' --time_stamp=' + _time_stamp + ' '

rawCmd = '\npython ~/DMFL/learner.py' + paramsCmd


if 'gpu-cn002' in avaiVms:
    avaiVms['gpu-cn002'] -= threadQuota

assignedVMs = []
# assert(len(availGPUs) > numOfWorkers)

# generate new scripts, assign each worker to different vms
for w in range(1, numOfWorkers + 1):
    print('assign ...' + str(w))
    rankId = ' --this_rank=' + str(w)
    fileName = jobPrefix+str(w)
    jobName = 'learner' + str(w)

    _vm = sorted(avaiVms, key=avaiVms.get, reverse=True)[0]
    assignedVMs.append(_vm)

    avaiVms[_vm] -= threadQuota
    if avaiVms[_vm] < threadQuota:
        del avaiVms[_vm]

    assignVm = '\n#BSUB -m "{}"\n'.format(_vm)
    runCmd = template + assignVm + '\n#BSUB -J ' + jobName + '\n#BSUB -e ' + fileName + '.e\n'  + '#BSUB -o '+ fileName + '.o\n'+ rawCmd + params + rankId

    with open('learner' + str(w) + '.lsf', 'w') as fout:
        fout.writelines(runCmd)

# deal with ps
rawCmdPs = '\npython ~/DMFL/param_server.py ' + paramsCmd + ' --this_rank=0 ' + params

with open('server.lsf', 'w') as fout:
    scriptPS = template + '\n#BSUB -J server\n#BSUB -e server{}'.format(timeStamp) + '.e\n#BSUB -o server{}'.format(timeStamp) + '.o\n' + '#BSUB -m "gpu-cn002"\n\n' + rawCmdPs
    fout.writelines(scriptPS)

# execute ps
os.system('bsub < server.lsf')

time.sleep(3)
os.system('rm vms')

vmSets = set()
for w in range(1, numOfWorkers + 1):
    # avoid gpu contention on the same machine
    if assignedVMs[w-1] in vmSets:
        time.sleep(3)
    vmSets.add(assignedVMs[w-1])
    os.system('bsub < learner' + str(w) + '.lsf')
