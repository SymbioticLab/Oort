import sys, os, time

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
jobPrefix = 'learner'

rawCmd = '\nsource activate pytorch3 \npython ~/DMFL/learner.py --ps_ip=10.255.11.91 --model=lenet --epochs=20000 --upload_epoch=30  --dump_epoch=200 --learning_rate=0.2 --decay_epoch=50 --model_avg=True --total_worker=30 --resampling_interval=1 --batch_size=64 --data_dir=~/cifar10/ --backend=tcp '
# generate new scripts
for w in range(1, numOfWorkers + 1):
    rankId = ' --this_rank=' + str(w)
    jobName = jobPrefix+str(w)
    runCmd = template + '\n#BSUB -J ' + jobName + '\n#BSUB -e ' + jobName + '.e\n'  + '#BSUB -o '+ jobName + '.o\n'+ rawCmd + params + rankId

    with open('learner' + str(w) + '.lsf', 'w') as fout:
        fout.writelines(runCmd)

# deal with ps
rawCmdPs = '\nsource activate pytorch3 \npython ~/DMFL/param_server.py --ps_ip=10.255.11.91 --model=lenet --epochs=20000 --upload_epoch=30  --dump_epoch=200 --learning_rate=0.2 --decay_epoch=50 --model_avg=True --total_worker=30 --resampling_interval=1 --batch_size=64 --data_dir=~/cifar10/ --backend=tcp --this_rank=0 ' + params
with open('server.lsf', 'w') as fout:
    scriptPS = template + '\n#BSUB -J server\n#BSUB -e server.e\n#BSUB -o server.o\n' + '#BSUB -m "gpu-cn001"\n\n' + rawCmdPs
    fout.writelines(scriptPS)

# execute ps
os.system('bsub < server.lsf')

time.sleep(5)

for w in range(1, numOfWorkers + 1):
    os.system('bsub < learner' + str(w) + '.lsf')
