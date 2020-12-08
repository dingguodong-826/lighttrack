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
from run_got10k_supernet import get_path_name_ablation
'''2020.09.26 WE NEED DEEPCOPY HERE'''
from copy import deepcopy
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)


choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))

'''2020.11.2 the most advanced version of evolutionary search'''
'''inherit ablation model'''
from Evolution.src.Search.search_DP_ablation import EvolutionSearcher_ablation
from lib.models import models_supernet

class EvolutionSearcher_advanced(EvolutionSearcher_ablation):

    def __init__(self, args):
        super(EvolutionSearcher_advanced, self).__init__(args)  # Inherit EvolutionSearcher_ablation
        '''2020.11.2'''
        model = models_supernet.__dict__[self.super_model_name](build_module=False)
        self.toolbox = models_supernet.__dict__["SuperNet_Toolbox"](model)
        self.tower_num = self.toolbox.model.tower_num

    def get_one_cand(self):
        '''2020.11.2 Using the toolbox'''
        # print('Using advanced toolbox')
        return self.toolbox.get_one_path()

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
        for cand in self.candidates:
            print(get_path_name_ablation(cand))
        '''get the performance of paths'''
        with open('paths.txt','w') as f:
            for cand in self.candidates:
                f.write(get_path_name_ablation(cand)+'\n')
        # evaluation in parallel
        print('Evaluating population ...')
        with open('paths_metric.txt','w') as f:
            f.write('This is a new file\n')
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s --ablation %d --max_flops_backbone %d --tower_num %d'
                  %(len(self.candidates),len(self.candidates),self.supernet_checkpoint, self.flops_name, self.super_model_name, self.ablation, self.max_flops_backbone, self.tower_num))
        with open('paths_metric.txt','r') as f:
            path_metric_list = f.readlines()
        self.parse_metric(path_metric_list[1:])
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        '''2020.11.2 Using the toolbox'''
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10
        '''2020.11.2 more general form'''
        num_choice_channel_head = self.toolbox.model.num_choice_channel_head
        num_choice_kernel_head = self.toolbox.model.num_choice_kernel_head
        tower_num = self.toolbox.model.tower_num
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
                for i in range(tower_num+1):
                    if np.random.random_sample() < m_prob:
                        if i == 0:
                            cand_h_cls[0] = random.randint(0, num_choice_channel_head-1)
                        else:
                            if i == 1:
                                cand_h_cls[1][i-1] = random.randint(0, num_choice_kernel_head-2) # can not be skip connection
                            else:
                                cand_h_cls[1][i-1] = random.randint(0, num_choice_kernel_head-1)
                '''reg head'''
                cand_h_reg = cand_h['reg']
                for i in range(tower_num+1):
                    if np.random.random_sample() < m_prob:
                        if i == 0:
                            cand_h_reg[0] = random.randint(0, num_choice_channel_head-1)
                        else:
                            if i == 1:
                                cand_h_reg[1][i-1] = random.randint(0, num_choice_kernel_head-2)
                            else:
                                cand_h_reg[1][i-1] = random.randint(0, num_choice_kernel_head-1)
            '''output position'''
            if self.search_ops:
                if np.random.random_sample() < m_prob:
                    '''This form is right. The earlier version is wrong!'''
                    cand_op = self.toolbox.get_path_ops()

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
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s --ablation %d --max_flops_backbone %d --tower_num %d'
                  %(len(res),len(res),self.supernet_checkpoint, self.flops_name, self.super_model_name, self.ablation, self.max_flops_backbone, self.tower_num))
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
        os.system('mpiexec -n %d python Evolution/src/Search/tester_mpi.py --path_file paths.txt --total_num %d --supernet_checkpoint %s --flops_name %s --super_model_name %s --ablation %d --max_flops_backbone %d --tower_num %d'
                  %(len(res), len(res), self.supernet_checkpoint, self.flops_name, self.super_model_name, self.ablation, self.max_flops_backbone, self.tower_num))
        with open('paths_metric.txt','r') as f:
            path_metric_list = f.readlines()
        self.parse_metric(path_metric_list[1:])
        print('crossover_num = {}'.format(len(res)))
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--checkpoint_path',type=str, default='snapshot/SuperNet_mul10_BN_before_600M_simple_head12_DP/checkpoint_e30.pth')
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
    '''2020.10.24 Add "restrict_sample" function, which doesn't use identity'''
    parser.add_argument('--restrict_sample', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    t = time.time()

    searcher = EvolutionSearcher_advanced(args)

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
