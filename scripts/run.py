import sys, os, time, datetime

os.system("bhosts > vms")
avaiVms = {}


with open('vms', 'r') as fin:
    lines = fin.readlines()
    for line in lines:
        if 'gpu-cn0' in line and 'gpu-cn001' not in line and 'gpu-cn002' not in line and 'gpu-cn005' not in line and 'gpu-cn007' not in line and 'gpu-cn015' not in line and 'gpu-cn012' not in line:
            items = line.strip().split()
            status = items[1]
            threadsGpu = int(items[5])

            if status == "ok":
                avaiVms[items[0]] = threadsGpu

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

rawCmd = '\npython ~/DMFL/learner.py --ps_ip=10.255.11.92 --model=squeezenet1_1 --epochs=20000 --upload_epoch=20  --dump_epoch=250 --learning_rate=0.01 --decay_epoch=50 --model_avg=True --batch_size=32 '

availGPUs = list(avaiVms.keys())
assignedVMs = []
# assert(len(availGPUs) > numOfWorkers)

# generate new scripts, assign each worker to different vms
for w in range(1, numOfWorkers + 1):
    rankId = ' --this_rank=' + str(w)
    fileName = jobPrefix+str(w)
    jobName = 'learner' + str(w)
    _vm = availGPUs[(w-1)%len(availGPUs)]
    assignedVMs.append(_vm)
    assignVm = '\n#BSUB -m "{}"\n'.format(_vm)
    runCmd = template + assignVm + '\n#BSUB -J ' + jobName + '\n#BSUB -e ' + fileName + '.e\n'  + '#BSUB -o '+ fileName + '.o\n'+ rawCmd + params + rankId

    with open('learner' + str(w) + '.lsf', 'w') as fout:
        fout.writelines(runCmd)

# deal with ps
rawCmdPs = '\npython ~/DMFL/param_server.py --ps_ip=10.255.11.92 --model=squeezenet1_1 --epochs=20000 --upload_epoch=20  --dump_epoch=250 --learning_rate=0.01 --decay_epoch=50 --model_avg=True --batch_size=32 --this_rank=0 ' + params

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
