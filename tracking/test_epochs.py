import os
import time
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--arch', dest='arch', default='SiamFCIncep22',
                    help='architecture of model')
parser.add_argument('--start_epoch', default=30, type=int, required=True, help='test end epoch')
parser.add_argument('--end_epoch', default=50, type=int, required=True,
                    help='test end epoch')
parser.add_argument('--gpu_nums', default=4, type=int, required=True, help='test start epoch')
parser.add_argument('--threads', default=16, type=int, required=True)
parser.add_argument('--dataset', default='VOT0219', type=str, help='benchmark to test')
parser.add_argument('--resume_dir',type= str,help='resume dir')
parser.add_argument('--stride', type=int, help='stride')
parser.add_argument('--effi', type=int, default=0)
parser.add_argument('--path_name', type=str, default='NULL')
args = parser.parse_args()

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums
node_name = MPI.Get_processor_name()  # get the name of the node
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)

# run test scripts -- two epoch for each thread
for i in range(2):
    arch = args.arch
    dataset = args.dataset
    try:
        epoch_ID += args.threads   # for 16 queue
    except:
        epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    resume = '{0}/checkpoint_e{1}.pth'.format(args.resume_dir,epoch_ID)

    print('==> test {}th epoch'.format(epoch_ID))
    command = 'python ./tracking/test_lighttrack.py --arch {0} --resume {1} --dataset {2} --epoch_test True --stride {3} --effi {4} --path_name {5}'\
            .format(arch, resume, dataset, args.stride, args.effi, args.path_name)
    print(command)
    os.system(command)