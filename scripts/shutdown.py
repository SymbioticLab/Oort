import os

os.system('bjobs > jobinfo')
tries = 3

with open('jobinfo', 'r') as fin:
    line = fin.readlines()

for ii in range(tries):
    for job in line[1:]:
        os.system('bkill ' + str(job.split()[0]))

os.system('rm jobinfo')
