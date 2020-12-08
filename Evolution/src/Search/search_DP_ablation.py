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

'''2020.09.21 Evaluating on GOT-10K'''
'''2020.10.24 compute macs for ablation study'''
from Evolution.src.Search.flops import get_cand_flops_ablation
from run_got10k_supernet import get_path_name_ablation
'''2020.09.26 WE NEED DEEPCOPY HERE'''
from copy import deepcopy
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)


choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))
'''2020.10.17 Inherit EvolutionSearcher'''
from lib.core.supernet_function_tracking import get_cand_with_prob, get_cand_head, get_oup_pos
'''2020.10.24 Inherit EvolutionSearcher_DP'''
from Evolution.src.Search.search_DP import EvolutionSearcher_DP

class EvolutionSearcher_ablation(EvolutionSearcher_DP):

    def __init__(self, args):
        super(EvolutionSearcher_ablation, self).__init__(args)  # Inherit EvolutionSearcher_DP
        self.search_back = args.search_back # backbone operations
        self.search_ops = args.search_ops # output positions
        self.search_head = args.search_head # head operations
        self.max_flops_backbone = args.max_flops_backbone
        self.ablation = 1

    def is_legal(self, cand):

        cand_name = get_path_name_ablation(cand)
        if cand_name not in self.vis_dict:
            self.vis_dict[cand_name] = {}
        info = self.vis_dict[cand_name]
        if 'visited' in info:
            return False

        if 'flops' not in info:
            info['flops'] = get_cand_flops_ablation(cand, mac_model_name=self.super_mac_model_name)

        print(cand_name, 'MACs: %.2f M'%(info['flops']/1e6))

        if info['flops'] > self.flops_limit:
            print('flops limit exceed')
            return False
        info['visited'] = True

        return True

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                cand_name = get_path_name_ablation(cand)
                if cand_name not in self.vis_dict:
                    self.vis_dict[cand_name] = {}
                info = self.vis_dict[cand_name]
            for cand in cands:
                yield cand
    def get_one_cand(self):
        cand_back, cand_OP, cand_h_dict = None, None, None
        if self.search_back or self.search_ops:
            # backbone operations
            cand_back = get_cand_with_prob(self.CHOICE_NUM, sta_num=self.sta_num)
            # add head and tail
            cand_back.insert(0, [0])
            cand_back.append([0])
        if self.search_ops:
            # backbone output positions
            cand_OP = get_oup_pos(self.sta_num)
        if self.search_head:
            # head operations
            cand_h_dict = {}
            cand_h_dict['cls'] = get_cand_head()
            cand_h_dict['reg'] = get_cand_head()
        return {'back':cand_back, 'ops':cand_OP, 'head':cand_h_dict}

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
                f.write(get_path_name_ablation(cand)+'\n')
        # evaluation in parallel
        print('Evaluating population ...')
        with open('paths_metric.txt','w') as f:
            f.write('This is a new file\n')
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s --ablation %d --max_flops_backbone %d'
                  %(len(self.candidates),len(self.candidates),self.supernet_checkpoint, self.flops_name, self.super_model_name, self.ablation, self.max_flops_backbone))
        with open('paths_metric.txt','r') as f:
            path_metric_list = f.readlines()
        self.parse_metric(path_metric_list[1:])
        print('random_num = {}'.format(len(self.candidates)))

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
            cand_b = deepcopy(cand['back'])
            cand_h = deepcopy(cand['head'])
            cand_op = deepcopy(cand['ops'])
            # mutatation
            '''backbone'''
            if self.search_back or self.search_ops:
                for stage_idx in range(len(self.sta_num)):
                    for block_idx in range(self.sta_num[stage_idx]):
                        if np.random.random_sample() < m_prob:
                            cand_b[1+stage_idx][block_idx] = np.random.randint(6)
            '''head'''
            if self.search_head:
                '''cls head'''
                cand_h_cls = cand_h['cls']
                for i in range(9):
                    if np.random.random_sample() < m_prob:
                        if i == 0:
                            cand_h_cls[0] = random.randint(0, 2)
                        else:
                            if i == 1:
                                cand_h_cls[1][i-1] = random.randint(0, 1)
                            else:
                                cand_h_cls[1][i-1] = random.randint(0, 2)
                '''reg head'''
                cand_h_reg = cand_h['reg']
                for i in range(9):
                    if np.random.random_sample() < m_prob:
                        if i == 0:
                            cand_h_reg[0] = random.randint(0, 2)
                        else:
                            if i == 1:
                                cand_h_reg[1][i-1] = random.randint(0, 1)
                            else:
                                cand_h_reg[1][i-1] = random.randint(0, 2)
            '''output position'''
            if self.search_ops:
                if np.random.random_sample() < m_prob:
                    cand_op[0] = random.randint(1,3)
                if np.random.random_sample() < m_prob:
                    num_block = self.sta_num[cand_op[0]]
                    cand_op[1] = random.randint(0, num_block - 1)

            # fuse
            cand_new = {'back': cand_b, 'head': cand_h, 'ops': cand_op}
            return cand_new

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
                f.write(get_path_name_ablation(cand)+'\n')
        print('Evaluating mutation population ...')
        with open('paths_metric.txt','w') as f:
            f.write('This is a new file\n')
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s --ablation %d --max_flops_backbone %d'
                  %(len(res),len(res),self.supernet_checkpoint, self.flops_name, self.super_model_name, self.ablation, self.max_flops_backbone))
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
            p = {'back':None, 'ops':None, 'head':None}
            '''backbone'''
            if self.search_back:
                p['back'] = [choice([i, j]) for i, j in zip(p1['back'], p2['back'])]
            '''head'''
            if self.search_head:
                p['head'] = {'cls':[None,[]], 'reg':[None,[]]}
                for idx_out, item_out in enumerate(zip(p1['head']['cls'],p2['head']['cls'])):
                    if idx_out == 0:
                        p['head']['cls'][0] = choice(item_out)
                    else:
                        list1, list2 = item_out
                        for idx_in, item_in in enumerate(zip(list1, list2)):
                            p['head']['cls'][1].append(choice(item_in))
                for idx_out, item_out in enumerate(zip(p1['head']['reg'], p2['head']['reg'])):
                    if idx_out == 0:
                        p['head']['reg'][0] = choice(item_out)
                    else:
                        list1, list2 = item_out
                        for idx_in, item_in in enumerate(zip(list1, list2)):
                            p['head']['reg'][1].append(choice(item_in))
            '''ops'''
            if self.search_ops:
                p['ops'] = choice((p1['ops'], p2['ops']))
            return p

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
                f.write(get_path_name_ablation(cand)+'\n')
        print('Evaluating crossover population ...')
        with open('paths_metric.txt','w') as f:
            f.write('This is a new file\n')
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s --ablation %d --max_flops_backbone %d'
                  %(len(res), len(res), self.supernet_checkpoint, self.flops_name, self.super_model_name, self.ablation, self.max_flops_backbone))
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
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[get_path_name_ablation(x)]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[get_path_name_ablation(x)]['acc'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 acc = {}'.format(
                    i + 1, cand, self.vis_dict[get_path_name_ablation(cand)]['acc']))
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
    parser.add_argument('--flops_limit', type=float, default=10 * 1e9)
    '''2020.10.17 build model based on the model name'''
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_flops_backbone', type=int)
    '''2020.10.24 Support only search some part of the supernet'''
    parser.add_argument('--search_back', type=int, default=0, choices=[0, 1])
    parser.add_argument('--search_ops', type=int, default=0, choices=[0, 1])
    parser.add_argument('--search_head', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    t = time.time()

    searcher = EvolutionSearcher_ablation(args)

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
