import torch


def main():
    checkpoint_path = 'Evolution/SuperNet_mul10_BN_before_470M_simple_head_DP/600M/log/checkpoint_19.pth.tar'
    info = torch.load(checkpoint_path)['vis_dict']
    cand = sorted([cand for cand in info if 'acc' in info[cand]],
                   key=lambda cand: info[cand]['acc'], reverse=True)[0]
    print('The best path is:',cand)


if __name__ == '__main__':
    main()
