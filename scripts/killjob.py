import os, sys

base = int(sys.argv[1])

for i in range(20):
	print(os.system("bkill -r " + str(i+base)))
