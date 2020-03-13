import argparse
import torch

parser = argparse.ArgumentParser()
# The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29501')
parser.add_argument('--manager_port', type=int, default='9005')
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--learners', type=str, default='1-2-3-4')
parser.add_argument('--total_worker', type=int, default=0)
parser.add_argument('--duplicate_data', type=int, default=1)
parser.add_argument('--data_mapfile', type=str, default=None)
parser.add_argument('--to_device', type=str, default='cuda')

# The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='~/cifar10/')
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--client_path', type=str, default='/tmp/client.cfg')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--depth', type=int, default=18)
parser.add_argument('--data_set', type=str, default='cifar10')
parser.add_argument('--sample_mode', type=str, default='random')
parser.add_argument('--score_mode', type=str, default='dis')
parser.add_argument('--proxy_avg', type=bool, default=False)
parser.add_argument('--filter_less', type=int, default=0)

# The configuration of different hyper-parameters for training
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_bsz', type=int, default=256)
parser.add_argument('--heterogeneity', type=float, default=1.0)
parser.add_argument('--hetero_allocation', type=str, default='1.0-1.0-1.0-1.0-1.0-1.0')
parser.add_argument('--backend', type=str, default="nccl")
parser.add_argument('--display_step', type=int, default=20)
parser.add_argument('--upload_epoch', type=int, default=1)
parser.add_argument('--validate_interval', type=int, default=999999)
parser.add_argument('--stale_threshold', type=int, default=0)
parser.add_argument('--sleep_up', type=int, default=0)
parser.add_argument('--force_read', type=bool, default=False)
parser.add_argument('--test_interval', type=int, default=999999)
parser.add_argument('--resampling_interval', type=int, default=99999999)
parser.add_argument('--sequential', type=str, default='0')
parser.add_argument('--single_sim', type=int, default=0)
parser.add_argument('--filter_class', type=int, default=0)
parser.add_argument('--num_class', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--model_avg', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--dump_epoch', type=int, default=100)
parser.add_argument('--decay_factor', type=float, default=0.9)
parser.add_argument('--decay_epoch', type=float, default=500)
parser.add_argument('--threads', type=str, default=str(torch.get_num_threads()))
parser.add_argument('--num_loaders', type=int, default=2)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--eval_interval_prior', type=int, default=9999999)
parser.add_argument('--gpu_device', type=int, default=0)
parser.add_argument('--zipf_alpha', type=str, default='5')
parser.add_argument('--timeout', type=float, default=9999999)
parser.add_argument('--full_gradient_interval', type=int, default=20)
parser.add_argument('--is_even_avg', type=bool, default=True)
parser.add_argument('--sample_seed', type=int, default=233)

args = parser.parse_args()

