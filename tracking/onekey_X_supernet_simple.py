import _init_paths
import os
import yaml
import argparse
from os.path import exists

def parse_args():
    """
    args for onekey.
    """
    parser = argparse.ArgumentParser(description='Train SiamFC with onekey')
    # for train
    parser.add_argument('--cfg', type=str, default='experiments/train/Ocean.yaml', help='yaml configure file name')
    parser.add_argument('--WORK_DIR', type=str, default='NULL')
    parser.add_argument('--back_super_dir', type=str, default='NULL') # pretrained supernet backbone dir


    args = parser.parse_args()

    return args

'''2020.09.07 Training supernet backbone and supernet head together'''
def main():
    args = parse_args()
    model_name = args.cfg.split('/')[-1].strip('yaml')[:-1]
    # train - test - tune information
    info = yaml.load(open(args.cfg, 'r').read())
    info = info['OCEAN']
    # info = info['MODEL']
    trainINFO = info['TRAIN']
    '''2020.7.26'''
    if args.WORK_DIR != 'NULL':
        info['CHECKPOINT_DIR'] = os.path.join(args.WORK_DIR,'checkpoints',model_name)
        info['OUTPUT_DIR'] = os.path.join(args.WORK_DIR,'logs',model_name)
        info['TRAIN']['CHECKPOINT_DIR'] = info['CHECKPOINT_DIR']
        supernet_checkpoint = os.path.join(args.back_super_dir, 'model_best.pth.tar')
        print('********** printing directory **********')
        print('CHECKPOINT_DIR:',info['CHECKPOINT_DIR'])
        print('OUTPUT_DIR:',info['OUTPUT_DIR'])
    else:
        supernet_checkpoint = 'NULL'
        print('Warning: Not loading pretrained model !')
    # epoch training -- train 50 or more epochs
    if trainINFO['ISTRUE']:
        print('==> train phase')
        DP = 1 if 'DP' in model_name else 0 # whether to use DP
        train_command = 'python ./tracking/train_X_supernet.py --cfg {0} --gpus {1} --workers {2} --OUTPUT_DIR {3} --CHECKPOINT_DIR {4} --SUPERNET_PATH {5} --DP {6} 2>&1 | tee {3}/siamrpn_train.log'\
            .format(args.cfg, info['GPUS'], info['WORKERS'], info['OUTPUT_DIR'], info['CHECKPOINT_DIR'], supernet_checkpoint, DP)
        print(train_command)

        if not exists(info['OUTPUT_DIR']):
            os.makedirs(info['OUTPUT_DIR'])

        os.system(train_command)

if __name__ == '__main__':
    main()
