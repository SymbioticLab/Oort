import os
import sys
import argparse

job_name = sys.argv[1]

if job_name == 'all':
    os.system("ps -ef | grep python | grep Oort > oort_running_temp")
else:
    os.system("ps -ef | grep python | grep job_name={} > oort_running_temp".format(job_name))
[os.system("kill -9 "+str(l.split()[1])) for l in open("oort_running_temp").readlines()]
os.system("rm oort_running_temp")
