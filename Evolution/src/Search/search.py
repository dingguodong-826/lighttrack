import os
import time
import numpy as np
import torch
import random
import _init_paths

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

'''2020.09.11'''
from lib.core.supernet_function_tracking import get_cand_with_prob, get_cand_head # 这一行放在最前面
'''2020.09.21 Evaluating on GOT-10K'''
from Evolution.src.Search.flops import get_cand_flops
from run_got10k_supernet import get_path_name
'''2020.09.26 WE NEED DEEPCOPY HERE'''
from copy import deepcopy
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)

from supernet_backbone.tools.supernet_bin import build_supernet
import yaml
import glob
choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))


class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit
        self.flops_name = '%dM' % (self.flops_limit / 1e6)
        #
        self.supernet_checkpoint = args.checkpoint_path
        self.CHOICE_NUM = 6

        self.log_dir = args.log_dir
        # self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        '''2020.10.17'''
        _, self.sta_num = build_supernet(flops_maximum=args.max_flops_backbone)
        print('sta_num: ',self.sta_num)
        self.super_model_name, self.super_mac_model_name = self.get_super_model_name(args.model_name)

    def get_super_model_name(self, yaml_name):
        yaml_file = 'experiments/train/%s.yaml'%yaml_name
        with open(yaml_file, 'r') as f:
            obj = yaml.load(f)
            super_model_name = obj['OCEAN']['TRAIN']['MODEL']
            super_mac_model_name = super_model_name + '_MACs'
        return super_model_name, super_mac_model_name

    def save_checkpoint(self, epoch=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        save_path = os.path.join(self.log_dir, 'checkpoint_%02d.pth.tar'%epoch)
        torch.save(info, save_path)
        print('save checkpoint to', save_path)

    def load_checkpoint(self):
        if not os.path.exists(self.log_dir):
            return False
        # if not os.path.exists(self.checkpoint_name):
        #     return False
        num_item = len(os.listdir(self.log_dir))
        if num_item == 0:
            return False
        checkpoint_name = sorted(os.listdir(self.log_dir))[-1]
        checkpoint_name = os.path.join(self.log_dir, checkpoint_name)
        info = torch.load(checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', checkpoint_name)
        return True

    def is_legal(self, cand):
        # assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        assert isinstance(cand,tuple) and isinstance(cand[0],list) and isinstance(cand[1],dict)
        cand_name = get_path_name(cand)
        if cand_name not in self.vis_dict:
            self.vis_dict[cand_name] = {}
        info = self.vis_dict[cand_name]
        if 'visited' in info:
            return False

        if 'flops' not in info:
            info['flops'] = get_cand_flops(cand, mac_model_name=self.super_mac_model_name)

        print(cand_name, 'MACs: %.2f M'%(info['flops']/1e6))

        if info['flops'] > self.flops_limit:
            print('flops limit exceed')
            return False
        info['visited'] = True

        return True
    '''2020.09.22 We use reverse=True (from the biggest to the smallest)'''
    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                cand_name = get_path_name(cand)
                if cand_name not in self.vis_dict:
                    self.vis_dict[cand_name] = {}
                info = self.vis_dict[cand_name]
            for cand in cands:
                yield cand
    def get_one_cand(self):
        # backbone
        get_random_cand = get_cand_with_prob(self.CHOICE_NUM, sta_num=self.sta_num)
        # add head and tail
        get_random_cand.insert(0, [0])
        get_random_cand.append([0])
        # head
        cand_h_dict = {}
        cand_h_dict['cls'] = get_cand_head()
        cand_h_dict['reg'] = get_cand_head()
        return (get_random_cand, cand_h_dict)
    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(
            lambda: self.get_one_cand())
        with open('ES_log.txt','a') as f:
            f.write('Population size is %d. We need %d.\n'%(len(self.candidates),num))

        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
            '''get the performance of paths'''
            with open('paths.txt','w') as f:
                for cand in self.candidates:
                    f.write(get_path_name(cand)+'\n')
            # evaluation in parallel
            print('Evaluating population ...')
            with open('paths_metric.txt','w') as f:
                f.write('This is a new file\n')
            os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s'
                      %(len(self.candidates),len(self.candidates),self.supernet_checkpoint, self.flops_name, self.super_model_name))
            with open('paths_metric.txt','r') as f:
                path_metric_list = f.readlines()
            self.parse_metric(path_metric_list[1:])
        print('random_num = {}'.format(len(self.candidates)))
    def parse_metric(self, path_metric_list):
        for line in path_metric_list:
            path_name, acc_str = line[:-1].split('\t')
            acc = float(acc_str)
            self.vis_dict[path_name]['acc'] = acc
    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            '''given one sample, get a mutated one'''
            # parse
            '''PAY ATTENTION! .copy() is necessary here, or self.keep_top_k will be modified!'''
            cand = choice(self.keep_top_k[k])
            cand_b = deepcopy(cand[0])
            cand_h_cls = deepcopy(cand[1]['cls'])
            cand_h_reg = deepcopy(cand[1]['reg'])
            # mutatation
            for stage_idx in range(len(self.sta_num)):
                for block_idx in range(self.sta_num[stage_idx]):
                    if np.random.random_sample() < m_prob:
                        cand_b[1+stage_idx][block_idx] = np.random.randint(6)
            for i in range(9):
                if np.random.random_sample() < m_prob:
                    if i == 0:
                        cand_h_cls[0] = random.randint(0, 2)
                    else:
                        if i == 1:
                            cand_h_cls[1][i-1] = random.randint(0, 1)
                        else:
                            cand_h_cls[1][i-1] = random.randint(0, 2)
            for i in range(9):
                if np.random.random_sample() < m_prob:
                    if i == 0:
                        cand_h_reg[0] = random.randint(0, 2)
                    else:
                        if i == 1:
                            cand_h_reg[1][i-1] = random.randint(0, 1)
                        else:
                            cand_h_reg[1][i-1] = random.randint(0, 2)
            # fuse
            cand_new = [None, None]
            cand_new[0] = cand_b
            cand_new[1] = {}
            cand_new[1]['cls'] = cand_h_cls
            cand_new[1]['reg'] = cand_h_reg
            return tuple(cand_new)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))
        with open('paths.txt','w') as f:
            for cand in res:
                f.write(get_path_name(cand)+'\n')
        print('Evaluating mutation population ...')
        with open('paths_metric.txt','w') as f:
            f.write('This is a new file\n')
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s'
                  %(len(res),len(res),self.supernet_checkpoint, self.flops_name, self.super_model_name))
        with open('paths_metric.txt','r') as f:
            path_metric_list = f.readlines()
        self.parse_metric(path_metric_list[1:])
        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            p = [None, {'cls':[None,[]], 'reg':[None,[]]}]
            p[0] = [choice([i, j]) for i, j in zip(p1[0], p2[0])]
            for idx_out, item_out in enumerate(zip(p1[1]['cls'],p2[1]['cls'])):
                if idx_out == 0:
                    p[1]['cls'][0] = choice(item_out)
                else:
                    list1, list2 = item_out
                    for idx_in, item_in in enumerate(zip(list1, list2)):
                        p[1]['cls'][1].append(choice(item_in))
            for idx_out, item_out in enumerate(zip(p1[1]['reg'], p2[1]['reg'])):
                if idx_out == 0:
                    p[1]['reg'][0] = choice(item_out)
                else:
                    list1, list2 = item_out
                    for idx_in, item_in in enumerate(zip(list1, list2)):
                        p[1]['reg'][1].append(choice(item_in))
            return tuple(p)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))
        with open('paths.txt','w') as f:
            for cand in res:
                f.write(get_path_name(cand)+'\n')
        print('Evaluating crossover population ...')
        with open('paths_metric.txt','w') as f:
            f.write('This is a new file\n')
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s'
                  %(len(res), len(res), self.supernet_checkpoint, self.flops_name, self.super_model_name))
        with open('paths_metric.txt','r') as f:
            path_metric_list = f.readlines()
        self.parse_metric(path_metric_list[1:])
        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        with open('ES_time_log.txt','w') as f:
            f.write('Evolutionary Search begins.\n')
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.load_checkpoint()
        start = time.time()
        self.get_random(self.population_num)
        interval = time.time() - start
        with open('ES_time_log.txt','a') as f:
            f.write('Generating and Evaluating the initial population:\t%.2f mins.\n'%(interval / 60))
        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[get_path_name(x)]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[get_path_name(x)]['acc'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 acc = {}'.format(
                    i + 1, cand, self.vis_dict[get_path_name(cand)]['acc']))
                ops = [i for i in cand]
                print(ops)
            # Mutation
            start_m = time.time()
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            interval_m = time.time() - start_m
            with open('ES_time_log.txt', 'a') as f:
                f.write('Epoch %d:\tMutation:\t%.2f mins.\n' % (self.epoch, interval_m / 60))
            # CrossOver
            start_c = time.time()
            crossover = self.get_crossover(self.select_num, self.crossover_num)
            interval_c = time.time() - start_c
            with open('ES_time_log.txt', 'a') as f:
                f.write('Epoch %d:\tCrossOver:\t%.2f mins.\n' % (self.epoch, interval_c / 60))
            self.candidates = mutation + crossover
            with open('ES_log.txt','a') as f:
                num_cand = len(self.candidates)
                f.write('Population size is %d\n'%num_cand)
            self.get_random(self.population_num)

            self.save_checkpoint(self.epoch)

            self.epoch += 1
        '''2020.10.19 get the best path'''
        checkpoint_path = os.path.join(self.log_dir, 'checkpoint_%d.pth.tar'%(self.max_epochs-1))
        info = torch.load(checkpoint_path)['vis_dict']
        best_cand = sorted([cand for cand in info if 'acc' in info[cand]],
                       key=lambda cand: info[cand]['acc'], reverse=True)[0]
        oup_filename = os.path.join(self.log_dir, 'best_path.txt')
        with open(oup_filename, 'w') as f:
            f.write(best_cand)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--checkpoint_path',type=str, default='checkpoints/SuperNet_mul3_BN_before/checkpoint_e50.pth')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--m_prob', type=float, default=0.1)
    '''Modified for higher efficiency on machines with 8 cards'''
    parser.add_argument('--population-num', type=int, default=48)
    parser.add_argument('--crossover-num', type=int, default=24)
    parser.add_argument('--mutation-num', type=int, default=24)
    # parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
    parser.add_argument('--flops_limit', type=float, default=10 * 1e9)
    parser.add_argument('--max-train-iters', type=int, default=200)
    parser.add_argument('--max-test-iters', type=int, default=40)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=200)
    '''2020.10.17 build model based on the model name'''
    parser.add_argument('--max_flops_backbone', type=int)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()

    t = time.time()

    searcher = EvolutionSearcher(args)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
