# Make sure you run this script on remote machine

from fabric import Config, Connection
import threadpool
import sys, time

pool = threadpool.ThreadPool(30)

address = ["h1", "h2", "h3", "h4", "h5"]
#, "h7", "h8"]
#address = ["h1", "h2", "h3","h4","h5", "h6", "h8", "h9", "h10"]
workers = address
#[x+prefix for x in address]
_arg = []

cmdSubmitTime = int(time.time() * 1000.0)

if len(sys.argv) > 2:
    _arg = sys.argv[2:]

def cnn(url):
    print('....', url)
    c = Connection(url)
    try:
        # kill all thread
        try:
            c.run("kill $(ps aux | grep 'startserver.py' | awk '{print $2}')")
        except:
            pass

        if "dc3master" in url:
            # start the sever
            c.run('python /users/fanlai/distcnn/startserver.py --deploy_mode=cluster --job_name=ps --task_index=0')
        else:
            # start the worker
            c.run('python /users/fanlai/distcnn/startserver.py --deploy_mode=cluster --job_name=worker --task_index='+str(int(url.replace(prefix, '').replace('h','')) - 1))
        c.close()

    except Exception as e:
        print (str(e))
        pass
    print('Done ....', url)


def cnn_bench(url):
    print('....', url)
    c = Connection(url)
    try:
        # kill all thread
        try:
            c.run("kill $(ps aux | grep 'startserver.py' | awk '{print $2}')")
            c.run("kill $(ps aux | grep 'tf_cnn_benchmarks.py' | awk '{print $2}')")
            c.run("rm -rf /tmp/cnn/")
        except:
            pass

        if "dc3master" in url:
            # start the sever
            c.run('python3 /users/fanlai/CNNBenchmark/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet56_v2 --variable_update=parameter_server --ps_hosts=10.0.0.253:2222 --worker_hosts=10.0.0.1:2222,10.0.0.2:2222,10.0.0.3:2222,10.0.0.4:2222,10.0.0.5:2222,10.0.0.6:2222 --data_name=cifar10 --data_dir=/users/fanlai/cifar-10-batches-py/ --job_name=ps --task_index=0  ' + ' '.join(_arg))
        else:
            # start the worker
            c.run('python3 /users/fanlai/CNNBenchmark/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet56_v2 --variable_update=parameter_server --ps_hosts=10.0.0.253:2222 --worker_hosts=10.0.0.1:2222,10.0.0.2:2222,10.0.0.3:2222,10.0.0.4:2222,10.0.0.5:2222,10.0.0.6:2222 --data_name=cifar10 --job_name=worker  --summary_verbosity=2 --display_every=10 --print_training_accuracy=True --data_dir=/users/fanlai/cifar-10-batches-py/ --task_index='+str(int(url.replace(prefix, '').replace('h',''))-1)  +' '+ ' '.join(_arg))
        c.close()

    except Exception as e:
        print (str(e))
        pass
    print('Done ....', url)

def torch_cifar(url):
    print('....', url)
    c = Connection(url)
    try:
        # kill all thread
        try:
            c.run("kill $(ps aux | grep 'python3 /users/fanlai/SSPTorch' | awk '{print $2}')")
        except:
            pass

        if "h1" in url:
            # start the sever
            c.run('python3 /users/fanlai/SSPTorch/param_server.py --ps_ip=10.0.0.1 --ps_port=29500 --this_rank=0 --learners=1-2-3-4 ' + ' '.join(_arg))
            pass
        else:
            # start the worker
            c.run('sleep 5 && python3 /users/fanlai/SSPTorch/learner.py --ps_ip=10.0.0.1 --ps_port=29500 --this_rank='+str(int(url.replace('h','')) - 1)+' --learners=1-2-3-4 '  + ' '.join(_arg))
        c.close()

    except Exception as e:
        print (str(e))
        pass
    print('Done ....', url)


def cnn_bench_tf(url):
    print('....', url)
    c = Connection(url)
    try:
        # kill all thread
        try:
            c.run("kill $(ps aux | grep 'startserver.py' | awk '{print $2}')")
            c.run("kill $(ps aux | grep 'tf_cnn_benchmarks.py' | awk '{print $2}')")
            c.run("rm -rf /tmp/cnn/")
        except:
            pass

        if "dc3master" in url:
            # start the sever
            c.run('python3 /users/fanlai/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=vgg16 --variable_update=parameter_server --ps_hosts=10.0.0.253:2222 --worker_hosts=10.0.0.1:2222,10.0.0.2:2222,10.0.0.3:2222,10.0.0.4:2222,10.0.0.5:2222,10.0.0.6:2222 --data_name=cifar10 --data_dir=/users/fanlai/cifar-10-batches-py --job_name=ps --task_index=0  ' + ' '.join(_arg))
        else:
            # start the worker
            c.run('python3 /users/fanlai/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=vgg16 --variable_update=parameter_server --ps_hosts=10.0.0.253:2222 --worker_hosts=10.0.0.1:2222,10.0.0.2:2222,10.0.0.3:2222,10.0.0.4:2222,10.0.0.5:2222,10.0.0.6:2222 --data_name=cifar10 --job_name=worker  --summary_verbosity=2 --display_every=10 --print_training_accuracy=True --data_dir=/users/fanlai/cifar-10-batches-py --task_index='+str(int(url.replace(prefix, '').replace('h',''))-1)  +' '+ ' '.join(_arg))
        c.close()

    except Exception as e:
        print (str(e))
        pass
    print('Done ....', url)

def logisticR(url):
    print('....', url)
    c = Connection(url)
    try:
        # kill all thread
        try:
            c.run("kill $(ps aux | grep 'tf_cnn_benchmarks.py' | awk '{print $2}')")
            c.run()
        except:
            pass

        if "dc3master" in url:
            # start the sever
            c.run('python /users/fanlai/distcnn/lr.py --job_name=ps --task_index=0 ' + ' '.join(_arg))
        else:
            # start the worker

            cmds = 'python /users/fanlai/distcnn/lr.py --job_name=worker --task_index='+str(int(url.replace(prefix, '').replace('h','')) - 1) + ' ' + ' '.join(_arg) + ' --submitTime=' + str(cmdSubmitTime)
            print(cmds)
            c.run(cmds)
        c.close()

    except Exception as e:
        print (str(e))
        pass
    print('Done ....', url)

def install(url):
    c = Connection(url)
    #c.run('pip install keras_applications keras_preprocessing mock absl-py && pip install /tmp/tensorflow-1.12.3-cp27-cp27mu-linux_x86_64.whl')
    c.run('pip install /tmp/tensorflow-1.12.3-cp27-cp27mu-linux_x86_64.whl')
    c.close()

def shutdown(url):
    print('....', url)
    c = Connection(url)
    #c.run('cp /tmp/log /tmp/res20SynLinkM_Var_E')
    try:
        # kill all thread
        try:
            #c.run("")
            c.run("kill $(ps aux | grep 'SSPTorch' | awk '{print $2}')")
            c.run("kill $(ps aux | grep 'startserver.py' | awk '{print $2}')")
        except:
            pass

        try:
            c.run("kill $(ps aux | grep 'SSP_Demo' | awk '{print $2}')")
        except:
            pass
        try:
            c.run("kill $(ps aux | grep 'lr.py' | awk '{print $2}')")
        except:
            pass

        try:
            c.run("kill $(ps aux | grep 'tf_cnn_benchmarks.py' | awk '{print $2}')")
        except:
            pass
        #c.run('cp /tmp/log /tmp/res20Syn')
        #c.run('python ~/benchmarks/scripts/tf_cnn_benchmarks/shutdown.py')

        c.run('rm -rf /users/fanlai/cnn')
    except Exception as e:
        print (str(e))
        pass
    print('Done ....', url)

def restart_tf(mode):
    print('Exec on masters and workers')

    if mode == "cnn-bench":
        conn_job = threadpool.makeRequests(cnn, workers)
        [pool.putRequest(req) for req in conn_job]
    elif mode == "lr":
        conn_job = threadpool.makeRequests(logisticR, workers)
        [pool.putRequest(req) for req in conn_job]
    elif mode == "shutdown":
        conn_job = threadpool.makeRequests(shutdown, workers)
        [pool.putRequest(req) for req in conn_job]
    elif mode == "cnn":
        conn_job = threadpool.makeRequests(cnn_bench, workers)
        [pool.putRequest(req) for req in conn_job]
    elif mode == "tfcnn":
        conn_job = threadpool.makeRequests(cnn_bench_tf, workers)
        [pool.putRequest(req) for req in conn_job]
    elif mode == 'install':
        conn_job = threadpool.makeRequests(install, workers)
        [pool.putRequest(req) for req in conn_job]
    elif mode == 'torch-cifar':
        conn_job = threadpool.makeRequests(torch_cifar, workers)
        [pool.putRequest(req) for req in conn_job]
    else:
        print(mode)

    pool.wait()

if __name__ == "__main__":
    restart_tf(sys.argv[1])
