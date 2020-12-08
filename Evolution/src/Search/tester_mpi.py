import os
import time
import argparse
from mpi4py import MPI
import torch
import _init_paths
from Evolution.src.Search.tester import get_cand_acc_tracking, get_cand_acc_tracking_ablation
from lib.core.supernet_function_tracking import name2path, name2path_ablation


parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--gpu_nums', default=8, type=int, help='test start epoch')
parser.add_argument('--path_file', type=str, required=True, help='path name file')
parser.add_argument('--total_num', type=int, required=True, help='total times of running. 24 for mut or cross, 48 for random')
parser.add_argument('--supernet_checkpoint', type=str, help='path of the pretrained supernet')
parser.add_argument('--flops_name', type=str, required=True)
parser.add_argument('--super_model_name', type=str, default='Ocean_SuperNet_BN_before', required=True)
parser.add_argument('--ablation', type=int, default=0)
parser.add_argument('--max_flops_backbone', type=int)
parser.add_argument('--tower_num', type=int, default=8)

args = parser.parse_args()

# parse file content
with open(args.path_file,'r') as f:
    path_list = f.readlines()

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums
node_name = MPI.Get_processor_name()  # get the name of the node

print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
torch.cuda.set_device(GPU_ID)


def check_path_exist(path_name, flops_name='600M'):
    exist_flag = False
    if 'ops' in path_name:
        result_dir = os.path.join('results/%s_DP/GOT-10k' % flops_name, path_name)
    else:
        result_dir = os.path.join('results/%s/GOT-10k'%flops_name,path_name)
    print('********** %s **********' % result_dir)
    if os.path.exists(result_dir):
        num_item = len(os.listdir(result_dir))
        if num_item == 180:
            exist_flag = True
    return exist_flag

def check_path_exist_ablation(path_name, flops_name='600M'):
    exist_flag = False
    result_dir = os.path.join('results_ablation/%s' % flops_name, path_name)
    print('********** %s **********' % result_dir)
    if os.path.exists(result_dir):
        num_item = len(os.listdir(result_dir))
        if num_item == 180:
            exist_flag = True
    return exist_flag

path_ID = rank
'''2020.10.17'''
if '470M' in args.super_model_name or args.max_flops_backbone == 470:
    sta_num = [2, 4, 4, 4, 4]
else:
    sta_num = [4, 4, 4, 4, 4]

if args.ablation:
    path = name2path_ablation(path_list[path_ID][:-1], sta_num=sta_num, num_tower=args.tower_num)  # [:-1] to remove '\n'
else:
    path = name2path(path_list[path_ID][:-1], sta_num=sta_num) # [:-1] to remove '\n'
if args.ablation:
    exist_flag = check_path_exist_ablation(path_list[path_ID][:-1], flops_name=args.flops_name)
else:
    exist_flag = check_path_exist(path_list[path_ID][:-1], flops_name=args.flops_name)
if args.ablation:
    get_cand_acc_tracking_ablation(path, gpu_id=GPU_ID, exist_flag=exist_flag, supernet_checkpoint=args.supernet_checkpoint,
                          flops_name=args.flops_name, super_model_name=args.super_model_name)
    # get_cand_acc_tracking_toy(path_list[path_ID][:-1])
else:
    get_cand_acc_tracking(path, gpu_id=GPU_ID, exist_flag=exist_flag, supernet_checkpoint=args.supernet_checkpoint,
                          flops_name=args.flops_name, super_model_name=args.super_model_name)
torch.cuda.empty_cache()