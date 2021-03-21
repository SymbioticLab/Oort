# Libraries for central aggregator
from flLibs import *

job_name = args.job_name if args.job_name is not None else args.time_stamp
logDir = os.path.join(log_path, args.model, job_name, 'aggregator')
logFile = logDir + 'log'

def init_logging():
    global logDir

    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    with open(logFile, 'w') as fin:
        pass

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler(logFile, mode='a'),
                            logging.StreamHandler()
                        ])

def dump_ps_ip():
    hostname_map = {}
    with open('ipmapping', 'rb') as fin:
        hostname_map = pickle.load(fin)

    ps_ip = str(hostname_map[str(socket.gethostname())])
    args.ps_ip = ps_ip

    with open(logDir+'ip', 'wb') as fout:
        pickle.dump(ps_ip, fout)

def initiate_aggregator_setting():
    init_logging()
    #dump_ps_ip()
