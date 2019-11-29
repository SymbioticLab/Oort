import sys, os, time, datetime

os.system("bhosts > vms")
avaiVms = {}

with open('vms', 'r') as fin:
    lines = fin.readlines()
    for line in lines:
        if 'gpu-cn0' in line and 'gpu-cn001' not in line and 'gpu-cn005' not in line:
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

# get the join of parameters
params = ' '.join(sys.argv[2:]) + learner + ' '
timeStamp = str(datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')) + '_'
jobPrefix = 'learner' + timeStamp

rawCmd = '\nsource activate pytorch3 \npython ~/DMFL/learner.py --ps_ip=10.255.11.91 --model=lenet --epochs=20000 --upload_epoch=30  --dump_epoch=200 --learning_rate=0.1 --decay_epoch=50 --model_avg=True --total_worker=30 --resampling_interval=1 --batch_size=64 --data_dir=~/cifar10/ --backend=tcp '
availGPUs = list(avaiVms.keys())

assert(len(availGPUs) > numOfWorkers)

# generate new scripts, assign each worker to different vms
for w in range(1, numOfWorkers + 1):
    rankId = ' --this_rank=' + str(w)
    fileName = jobPrefix+str(w)
    jobName = 'learner' + str(w)
    assignVm = '\n#BSUB -m "{}"\n'.format(availGPUs[w-1])
    runCmd = template + assignVm + '\n#BSUB -J ' + jobName + '\n#BSUB -e ' + fileName + '.e\n'  + '#BSUB -o '+ fileName + '.o\n'+ rawCmd + params + rankId

    with open('learner' + str(w) + '.lsf', 'w') as fout:
        fout.writelines(runCmd)

# deal with ps
rawCmdPs = '\nsource activate pytorch3 \npython ~/DMFL/param_server.py --ps_ip=10.255.11.91 --model=lenet --epochs=20000 --upload_epoch=30  --dump_epoch=200 --learning_rate=0.1 --decay_epoch=50 --model_avg=True --total_worker=30 --resampling_interval=1 --batch_size=64 --data_dir=~/cifar10/ --backend=tcp --this_rank=0 ' + params
with open('server.lsf', 'w') as fout:
    scriptPS = template + '\n#BSUB -J server\n#BSUB -e server{}'.format(timeStamp) + '.e\n#BSUB -o server{}'.format(timeStamp) + '.o\n' + '#BSUB -m "gpu-cn001"\n\n' + rawCmdPs
    fout.writelines(scriptPS)

# execute ps
os.system('bsub < server.lsf')

time.sleep(5)

for w in range(1, numOfWorkers + 1):
    os.system('bsub < learner' + str(w) + '.lsf')
