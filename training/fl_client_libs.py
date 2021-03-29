# package for client
from flLibs import *

job_name = args.job_name if args.job_name is not None else args.time_stamp
logDir = os.path.join(args.log_path, args.model, job_name, 'worker') #os.getcwd() + "/../../models/" + args.model + '/' + job_name+ '/learner/'
logFile = os.path.join(logDir, 'log_'+str(args.this_rank))

def init_logging():
    global logDir, logging

    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logFile, mode='a'),
                        logging.StreamHandler()
                    ])

def get_ps_ip():
    global args

    ip_file = logDir + '../server/ip'
    ps_ip = None
    while not os.path.exists(ip_file):
        time.sleep(1)

    with open(ip_file, 'rb') as fin:
        ps_ip = pickle.load(fin)

    args.ps_ip = ps_ip
    logging.info('====Config ps_ip on {}, args.ps_ip is {}'.format(ps_ip, args.ps_ip))

def initiate_client_setting():
    init_logging()
    #get_ps_ip()
