import torch
import tqdm
import random
'''2020.09.21 dataloader about GOT-10K'''
from lib.dataset.ocean_normalize import OceanDataset
from run_got10k_supernet import eval_path, eval_path_ablation
from torch.utils.data import DataLoader
from config_got10k import config
import lib.models.models as models_supernet

assert torch.cuda.is_available()
train_dataprovider = None

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func
def get_train_loader():
    # gpu_num = 8
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build dataloader, benefit to tracking
    train_set = OceanDataset(config)
    # train_loader = DataLoader(train_set, batch_size=config.OCEAN.TRAIN.BATCH * gpu_num,
    #                           num_workers=config.WORKERS, pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_set, batch_size=config.OCEAN.TRAIN.BATCH,
                              num_workers=2, pin_memory=True, drop_last=True)
    return DataIterator(train_loader)

'''2020.09.21 Evaluating one path on GOT-10K (Don't forget to recompute BN)'''
@no_grad_wrapper
def get_cand_acc_tracking(cand, metric=True, gpu_id=None, exist_flag=False,
                          super_model_name='SuperNet_mul3_BN_before',
                          supernet_checkpoint='checkpoints/SuperNet_mul3/checkpoint_e50.pth',
                          flops_name='600M'):
    '''metric: whether to compute metric'''
    if not exist_flag:
        '''build a dataloader'''
        train_dataprovider = get_train_loader()
        max_train_iters = 200
        '''build a model'''
        # model = Ocean_SuperNet()
        '''2020.10.17 build model based on model name'''
        model = models_supernet.__dict__[super_model_name]()
        print('loading from %s'%supernet_checkpoint)
        supernet_state_dict = torch.load(supernet_checkpoint, map_location='cpu')['state_dict']
        model.load_state_dict(supernet_state_dict, strict=False)
        del supernet_state_dict
        '''Step1: Recompute BN statistics'''
        print('clear bn statics....')
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        print('train bn with training set (BN sanitize) ....')
        model.train()
        model.cuda(gpu_id)

        for _ in tqdm.tqdm(range(max_train_iters)):
            # print('train step: {} total: {}'.format(step,max_train_iters))
            input = train_dataprovider.next()
            template = input[0].cuda(gpu_id)
            search = input[1].cuda(gpu_id)

            if 'DP' in super_model_name:
                model(template, search, cand_b=cand[0], cand_h_dict=cand[1], backbone_index=cand[2])
            else:
                model(template, search, cand_b=cand[0], cand_h_dict=cand[1])

            del input, template, search

        '''Step2: Evaluate on GOT-10K eval'''
        model.eval()
        metric = eval_path(cand, model=model, metric=metric, exist_flag=exist_flag, flops_name=flops_name)
    else:
        '''Step2: Evaluate on GOT-10K eval'''
        metric = eval_path(cand, model=None, metric=metric, exist_flag=exist_flag, flops_name=flops_name)
    return metric

@no_grad_wrapper
def get_cand_acc_tracking_ablation(cand, metric=True, gpu_id=None, exist_flag=False,
                          super_model_name='SuperNet_mul3_BN_before',
                          supernet_checkpoint='checkpoints/SuperNet_mul3/checkpoint_e50.pth',
                          flops_name='600M'):
    '''metric: whether to compute metric'''
    if not exist_flag:
        '''build a dataloader'''
        train_dataprovider = get_train_loader()
        max_train_iters = 200
        '''build a model'''
        # model = Ocean_SuperNet()
        '''2020.10.17 build model based on model name'''
        model = models_supernet.__dict__[super_model_name]()
        print('loading from %s'%supernet_checkpoint)
        supernet_state_dict = torch.load(supernet_checkpoint, map_location='cpu')['state_dict']
        try:
            model.load_state_dict(supernet_state_dict, strict=True)
        except:
            model.load_state_dict(supernet_state_dict, strict=False)
        del supernet_state_dict
        '''Step1: Recompute BN statistics'''
        '''2020.10.24 add clean_BN method'''
        model.clean_BN()

        print('train bn with training set (BN sanitize) ....')
        model.train()
        model.cuda(gpu_id)

        for _ in tqdm.tqdm(range(max_train_iters)):
            # print('train step: {} total: {}'.format(step,max_train_iters))
            input = train_dataprovider.next()
            template = input[0].cuda(gpu_id)
            search = input[1].cuda(gpu_id)
            if isinstance(cand, dict):
                model(template, search, cand_b=cand['back'], cand_h_dict=cand['head'], backbone_index=cand['ops'])
            else:
                raise ValueError ('Unsupported type')

            del input, template, search

        '''Step2: Evaluate on GOT-10K eval'''
        model.eval()
        metric = eval_path_ablation(cand, model=model, metric=metric, exist_flag=exist_flag, flops_name=flops_name)
    else:
        '''Step2: Evaluate on GOT-10K eval'''
        metric = eval_path_ablation(cand, model=None, metric=metric, exist_flag=exist_flag, flops_name=flops_name)
    return metric

@no_grad_wrapper
def get_cand_acc_tracking_toy(path_name):
    arr = random.random()
    with open('paths_metric.txt', 'a') as f:
        f.write('%s\t%.4f\n' % (path_name, arr))


if __name__ == "__main__":
    '''build a model'''
    model_name = 'SuperNet_mul3_BN_before'
    model = models_supernet.__dict__[model_name]()
    supernet_state_dict = torch.load('checkpoints/SuperNet_mul3/checkpoint_e50.pth')['state_dict']
    model.load_state_dict(supernet_state_dict)
    del supernet_state_dict
    model = model.cuda()
    # parse file content
    with open('paths.txt', 'r') as f:
        path_list = f.readlines()
    path_ID = 0
    path = name2path(path_list[path_ID])
    get_cand_acc_tracking(model, path)