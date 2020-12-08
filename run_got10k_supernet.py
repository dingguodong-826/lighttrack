import torch

from easydict import EasyDict as edict


from toolkit.got10k.experiments import ExperimentGOT10k
'''2020.09.21 Evaluate Path'''
from lib.tracker.supernet_got10k import SuperTracker
'''2020.10.24 SuperTracker for ablation study'''
from lib.tracker.supernet_got10k_ablation import SuperTracker_ablation

def simple_name(ori_name):
    return ori_name.replace('[','').replace(']','').replace(', ','')

'''path --> name'''
def get_path_name(cand):
    str_back = simple_name(str(cand[0]))
    str_cls = simple_name(str(cand[1]['cls']))
    str_reg = simple_name(str(cand[1]['reg']))
    cand_name = 'back_%s+cls_%s+reg_%s' % (str_back, str_cls, str_reg)
    if len(cand) > 2:
        cand_name = cand_name + '_ops_%d%d'%(cand[2][0],cand[2][1])
    return cand_name


def get_path_name_ablation(cand: dict):
    str_back = 'back_' + simple_name(str(cand['back'])) if cand['back'] is not None else ''
    str_cls = 'cls_' + simple_name(str(cand['head']['cls'])) if cand['head'] is not None else ''
    str_reg = 'reg_' + simple_name(str(cand['head']['reg'])) if cand['head'] is not None else ''
    str_oup = 'ops_%d%d'%(cand['ops'][0],cand['ops'][1]) if cand['ops'] is not None else ''
    str_list = [str_back, str_cls, str_reg, str_oup]
    str_list_new = [item for item in str_list if item != '']
    cand_name = '+'.join(str_list_new)
    return cand_name

def eval_path(path, model=None, metric=True, write=True, exist_flag=False, flops_name='600M'):
    effi = 1
    info = edict()
    info.TRT = False
    info.epoch_test = False
    info.stride = 16

    siam_info = edict()
    siam_info.online = False
    siam_info.epoch_test = info.epoch_test
    siam_info.TRT = False
    siam_info.stride = info.stride
    siam_info.align = False

    path_name = get_path_name(path)
    if len(path) == 3:
        result_dir = 'results/%s_DP'%flops_name
        report_dir = 'reports/%s_DP'%flops_name
    elif len(path) == 2:
        result_dir = 'results/%s'%flops_name
        report_dir = 'reports/%s'%flops_name
    else:
        raise ValueError ('Length of path must be 2 or 3')
    experiment = ExperimentGOT10k(root_dir='GOT-10K', subset='val', result_dir=result_dir, report_dir=report_dir)

    if exist_flag is False and model is not None:
        '''build the complete tracker'''
        siam_tracker = SuperTracker(siam_info, name=path_name, effi=effi, model=model, cand=path)
        '''run experiments on got10k'''
        experiment.run(siam_tracker, visualize=False, overwrite_result=False)
    if metric:
        performance = experiment.report([path_name],plot_curves=False)
        ao = performance[path_name]['overall']['ao']
        sr = performance[path_name]['overall']['sr']
        arr = (ao + sr) / 2
        print('AO is %.2f, SR is %.2f'%(ao,sr))
        if write:
            with open('paths_metric.txt','a') as f:
                f.write('%s\t%.4f\n'%(path_name, arr))
        return arr
    else:
        return None

def eval_path_ablation(path, model=None, metric=True, write=True, exist_flag=False, flops_name='600M'):
    effi = 1
    info = edict()
    info.TRT = False
    info.epoch_test = False
    info.stride = 16

    siam_info = edict()
    siam_info.online = False
    siam_info.epoch_test = info.epoch_test
    siam_info.TRT = False
    siam_info.stride = info.stride
    siam_info.align = False

    path_name = get_path_name_ablation(path)
    result_dir = 'results_ablation/%s' % flops_name
    report_dir = 'reports_ablation/%s' % flops_name

    experiment = ExperimentGOT10k(root_dir='GOT-10K', subset='val', result_dir=result_dir, report_dir=report_dir)

    if exist_flag is False and model is not None:
        '''build the complete tracker'''
        siam_tracker = SuperTracker_ablation(siam_info, name=path_name, effi=effi, model=model, cand=path)
        '''run experiments on got10k'''
        experiment.run(siam_tracker, visualize=False, overwrite_result=False)
    if metric:
        performance = experiment.report([path_name],plot_curves=False)
        ao = performance[path_name]['overall']['ao']
        sr = performance[path_name]['overall']['sr']
        arr = (ao + sr) / 2
        print('AO is %.2f, SR is %.2f'%(ao,sr))
        if write:
            with open('paths_metric.txt','a') as f:
                f.write('%s\t%.4f\n'%(path_name, arr))
        return arr
    else:
        return None
