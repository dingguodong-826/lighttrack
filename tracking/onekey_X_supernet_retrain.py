
import _init_paths
import os
import yaml
import argparse
from os.path import exists
from lib.utils.utils import load_yaml, extract_logs

def parse_args():
    """
    args for onekey.
    """
    parser = argparse.ArgumentParser(description='Train SiamFC with onekey')
    # for train
    parser.add_argument('--cfg', type=str, default='experiments/train/Ocean.yaml', help='yaml configure file name')
    parser.add_argument('--WORK_DIR', type=str, default='NULL')
    parser.add_argument('--back_exp_dir', type=str)
    parser.add_argument('--HOME_DIR', type=str, default='NULL')
    parser.add_argument('--test_data', type=str, default='NULL')
    parser.add_argument('--effi',type=int, default=1)

    args = parser.parse_args()

    return args


'''2020.10.05 Retrain the backbone and supernet head together'''
def main():
    args = parse_args()
    model_name = args.cfg.split('/')[-1].strip('yaml')[:-1]
    if '_SP_' in model_name:
        path_name = model_name.split('_SP_')[0]
    else:
        path_name = model_name
    # train - test - tune information
    info = yaml.load(open(args.cfg, 'r').read())
    info = info['OCEAN']
    # info = info['MODEL']
    trainINFO = info['TRAIN']
    testINFO = info['TEST']
    tuneINFO = info['TUNE']
    dataINFO = info['DATASET']
    Effi = args.effi
    '''2020.7.26'''
    if args.WORK_DIR != 'NULL':
        info['CHECKPOINT_DIR'] = os.path.join(args.WORK_DIR,'checkpoints',model_name)
        info['OUTPUT_DIR'] = os.path.join(args.WORK_DIR,'logs',model_name)
        info['TRAIN']['CHECKPOINT_DIR'] = info['CHECKPOINT_DIR']
        pretrain_dir = os.path.join(args.back_exp_dir, 'retrain', path_name)
        print('********** printing directory **********')
        print('CHECKPOINT_DIR:',info['CHECKPOINT_DIR'])
        print('OUTPUT_DIR:',info['OUTPUT_DIR'])
    else:
        pretrain_dir = os.path.join('checkpoints_supernet_470M', 'retrain', path_name)
    # epoch training -- train 50 or more epochs
    if trainINFO['ISTRUE']:
        print('==> train phase')
        train_command = 'python ./tracking/retrain_supernet.py --cfg {0} --gpus {1} --workers {2} --OUTPUT_DIR {3} --CHECKPOINT_DIR {4} --path_name {5} --pretrain_dir {6} 2>&1 | tee {3}/siamrpn_train.log'\
            .format(args.cfg, info['GPUS'], info['WORKERS'], info['OUTPUT_DIR'], info['CHECKPOINT_DIR'], path_name, pretrain_dir)

        print(train_command)

        if not exists(info['OUTPUT_DIR']):
            os.makedirs(info['OUTPUT_DIR'])

        os.system(train_command)

    # epoch testing -- test 30-50 epochs (or more)
    if testINFO['ISTRUE']:
        print('==> test phase')
        if args.test_data != 'NULL':
            test_data = args.test_data
        else:
            test_data = testINFO['DATA']
        test_command = 'mpiexec -n {0} python ./tracking/test_epochs.py --arch {1} --start_epoch {2} --end_epoch {3} --gpu_nums={4} \
                          --threads {0} --dataset {5}  --resume_dir {6} --stride {8} --effi {9} --path_name {10} 2>&1 | tee {7}/ocean_epoch_test.log'\
            .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                      (len(info['GPUS']) + 1) // 2, test_data,
                      info['CHECKPOINT_DIR'], info['OUTPUT_DIR'], trainINFO['STRIDE'], Effi, model_name)
        print(test_command)
        if not exists(info['OUTPUT_DIR']):
            os.makedirs(info['OUTPUT_DIR'])

        os.system(test_command)
        # test on vot or otb benchmark
        if args.WORK_DIR != 'NULL':
            result_dir = os.path.join(args.WORK_DIR, 'result')
        else:
            result_dir = './result'
        print('====> use new testing toolkit')
        trackers = os.listdir(os.path.join(result_dir, test_data, model_name))
        trackers = " ".join(trackers)
        if test_data == 'LASOT' or test_data == 'LaSOT':
            dataset_name = 'LaSOT'
            dataset_dir = 'dataset/LaSOT'
            os.system(
                'python lib/eval_toolkit/bin/eval.py --dataset_dir {5} --dataset {0} --tracker_result_dir {4}/LASOT/{2} --trackers {1} 2>&1 | tee {3}/ocean_eval_epochs.log'.format(
                    dataset_name, trackers, model_name, info['OUTPUT_DIR'], result_dir, dataset_dir))

        elif 'VOT' in test_data:
            dataset_name = test_data
            dataset_dir = 'dataset'
            print('python lib/eval_toolkit/bin/eval.py --dataset_dir {5} --dataset {0} --tracker_result_dir {4}/{0}/{2} --trackers {1} 2>&1 | tee {3}/ocean_eval_epochs.log'.format(dataset_name, trackers, model_name, info['OUTPUT_DIR'], result_dir, dataset_dir))
            os.system('python lib/eval_toolkit/bin/eval.py --dataset_dir {5} --dataset {0} --tracker_result_dir {4}/{0}/{2} --trackers {1} 2>&1 | tee {3}/ocean_eval_epochs.log'.format(dataset_name, trackers, model_name, info['OUTPUT_DIR'], result_dir, dataset_dir))
        # else:
        #     raise ValueError('not supported now, please add new dataset')

if __name__ == '__main__':
    main()
