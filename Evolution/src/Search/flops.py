import torch
import torch.nn as nn

from thop import profile
from thop.utils import clever_format
import torch
from efficientnet_pytorch.utils import Conv2dDynamicSamePadding
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from efficientnet_pytorch.utils import MemoryEfficientSwish
from thop.vision.basic_hooks import count_convNd, zero_ops

from lib.core.supernet_function_tracking import get_cand_with_prob, get_cand_head
from lib.models.models_supernet import *

MODEL_DICT = {
    'Ocean_SuperNet_BN_before_470M_simple_head_DP_MACs':{'mac_model':Ocean_SuperNet_BN_before_470M_simple_head_DP_MACs().cuda(),
                                                         'zf': [torch.randn(1, 40, 16, 16).cuda(),torch.randn(1, 80, 8, 8).cuda(),torch.randn(1, 96, 8, 8).cuda()],
                                                         'search': torch.rand(1,3,256,256).cuda()},
}


def get_cand_flops(cand, mac_model_name='Ocean_SuperNet_BN_before_MACs'):
    if len(cand) == 2:
        cand_b, cand_h_dict = cand
        # return compute_macs(cand_b, cand_h_dict)
        return compute_macs(cand_b, cand_h_dict, mac_model_name=mac_model_name)
    elif len(cand) == 3 and 'DP' in mac_model_name:
        cand_b, cand_h_dict, cand_op = cand
        return compute_macs_DP(cand_b, cand_h_dict, cand_op, mac_model_name=mac_model_name)
    else:
        raise ValueError ('Wrong cand or mac_model_name !')

def get_cand_flops_ablation(cand, mac_model_name='Ocean_SuperNet_BN_before_MACs'):

    cand_b, cand_h_dict, cand_op = cand['back'], cand['head'], cand['ops']
    return compute_macs_DP(cand_b, cand_h_dict, cand_op, mac_model_name=mac_model_name)


def compute_macs(cand_b, cand_h_dict, mac_model_name='Ocean_SuperNet_BN_before_MACs'):
    # model
    print(mac_model_name)
    model_dict = MODEL_DICT[mac_model_name]
    mac_model, zf, search = model_dict['mac_model'], model_dict['zf'], model_dict['search']
    # extra operations
    custom_ops = {
        Conv2dDynamicSamePadding: count_convNd,
        Conv2dStaticSamePadding: count_convNd,
        MemoryEfficientSwish: zero_ops,
    }
    macs, params = profile(mac_model, inputs=(zf, search, cand_b, cand_h_dict), custom_ops=custom_ops, verbose=False)
    return macs

'''2020.10.17'''
def compute_macs_DP(cand_b, cand_h_dict, backbone_index, mac_model_name='Ocean_SuperNet_BN_before_470M_simple_head_DP_MACs'):
    # model
    model_dict = MODEL_DICT[mac_model_name]
    mac_model, zf, search = model_dict['mac_model'], model_dict['zf'], model_dict['search']
    # extra operations
    custom_ops = {
        Conv2dDynamicSamePadding: count_convNd,
        Conv2dStaticSamePadding: count_convNd,
        MemoryEfficientSwish: zero_ops,
    }
    macs, params = profile(mac_model, inputs=(zf, search, cand_b, cand_h_dict, backbone_index), custom_ops=custom_ops, verbose=False)
    return macs


