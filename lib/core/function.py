import math
import time
import torch
from utils.utils import print_speed
'''2020.07.31'''
# from lib.models.reid import contrastive_loss
import numpy as np
# -----------------------------
# Main training code for Ocean
# -----------------------------
def ocean_train(train_loader, model, optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()
    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)

        cls_loss_ori, cls_loss_align, reg_loss = model(template, search, label_cls, reg_target=reg_label, reg_weight=reg_weight)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        if cls_loss_align is not None:
            cls_loss_align = torch.mean(cls_loss_align)
            loss = cls_loss_ori + cls_loss_align + 1.2 * reg_loss
        else:
            cls_loss_align = 0
            loss = cls_loss_ori +  1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item() # 当前结果
        losses.update(loss, template.size(0)) # 根据历史值算出的平均结果

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps) # 写到tensorboard里的是当前数据的结果，打印到控制台的是平均结果
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

'''2020.08.19 train with ema'''
def ocean_train_ema(train_loader, model, optimizer, ema, epoch, cur_lr, cfg, writer_dict, logger, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()
    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)

        cls_loss_ori, cls_loss_align, reg_loss = model(template, search, label_cls, reg_target=reg_label, reg_weight=reg_weight)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        if cls_loss_align is not None:
            cls_loss_align = torch.mean(cls_loss_align)
            loss = cls_loss_ori + cls_loss_align + 1.2 * reg_loss
        else:
            cls_loss_align = 0
            loss = cls_loss_ori +  1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()
        '''ema'''
        ema.update(model.parameters())
        # record loss
        loss = loss.item() # 当前结果
        losses.update(loss, template.size(0)) # 根据历史值算出的平均结果

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps) # 写到tensorboard里的是当前数据的结果，打印到控制台的是平均结果
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

'''2020.08.12 Ocean_train with more negative samples'''
def ocean_train_nega(train_loader, model,  optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    cls_losses_neg = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    '''2020.08.13 NOT HERE TO SET MODEL.TRAIN()'''
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)
        '''add cls_loss for negative samples'''
        cls_loss_ori, cls_loss_align, reg_loss, cls_loss_neg = \
            model(template, search, label_cls, reg_target=reg_label, reg_weight=reg_weight)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)
        cls_loss_neg = torch.mean(cls_loss_neg)
        '''add cls_loss_neg'''
        if cls_loss_align is not None:
            cls_loss_align = torch.mean(cls_loss_align)
            loss = cls_loss_ori + cls_loss_align + 1.2 * reg_loss + \
                   cfg.OCEAN.TRAIN.NEGA_WEIGHT * cls_loss_neg
        else:
            cls_loss_align = 0
            loss = cls_loss_ori +  1.2 * reg_loss + \
                   cfg.OCEAN.TRAIN.NEGA_WEIGHT * cls_loss_neg

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item() # 当前结果
        losses.update(loss, template.size(0)) # 根据历史值算出的平均结果

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        cls_loss_neg = cls_loss_neg.item()
        cls_losses_neg.update(cls_loss_neg, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t '
                'Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t '
                'CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t '
                'CLS_NEG Loss:{cls_loss_neg.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses, cls_loss_neg=cls_losses_neg))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps) # 写到tensorboard里的是当前数据的结果，打印到控制台的是平均结果
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict


def ocean_reid_train(train_loader, model,  optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()


    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        '''reid branch'''
        # print('before squeeze, dim is: ',input[1].size())
        reid_patch = input[1].to(device)
        # print('after squeeze, dim is: ',reid_patch.size())
        # input and output/loss
        label_cls = input[0][2].type(torch.FloatTensor)  # BCE need float
        template = input[0][0].to(device)
        search = input[0][1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[0][3].float().to(device)
        reg_weight = input[0][4].float().to(device)

        cls_loss_ori, cls_loss_align, reg_loss, reid_loss = model(template, search, label_cls,
                                                                  reg_target=reg_label,
                                                                  reg_weight=reg_weight,
                                                                  reid=True,reid_input=reid_patch)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        if cls_loss_align is not None:
            cls_loss_align = torch.mean(cls_loss_align)
            loss = cls_loss_ori + cls_loss_align + 1.2 * reg_loss
        else:
            cls_loss_align = 0
            loss = cls_loss_ori +  1.2 * reg_loss

        loss = torch.mean(loss)
        reid_loss = torch.mean(reid_loss)
        '''2020.07.31'''
        total_loss = loss + 10 * reid_loss
        # compute gradient and do update step
        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item() # 当前结果
        losses.update(loss, template.size(0)) # 根据历史值算出的平均结果

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        reid_loss = reid_loss.item() # 当前reid loss
        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps) # 写到tensorboard里的是当前数据的结果，打印到控制台的是平均结果
        writer.add_scalar('reid_loss',reid_loss,global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

'''2020.08.03 reid_train_new'''
def reid_train_new(train_loader, model,  optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    reid_losses = AverageMeter()
    end = time.time()


    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # print('input shape is ',input.size())
        # measure data loading time
        data_time.update(time.time() - end)
        # input and output/loss
        '''reid branch'''
        reid_patch = input.to(device)
        # print('after squeeze, dim is: ',reid_patch.size())
        reid_loss = model(reid=True, reid_input=reid_patch)
        reid_loss = torch.mean(reid_loss)
        # compute gradient and do update step
        optimizer.zero_grad()
        reid_loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(reid_loss.item()):
            optimizer.step()

        # record loss
        reid_loss = reid_loss.item() # 当前结果
        reid_losses.update(reid_loss, 1) # 根据历史值算出的平均结果


        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'ReID_Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t ReID Loss:{reid_loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    reid_loss=reid_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('reid_loss',reid_loss,global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

# ------------------------------------------
# Main code for Ocean Plus training
# ------------------------------------------
def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()

def oceanplus_train(train_loader, model, optimizer, epoch, cur_lr, cfg, writer_dict, logger, device):
    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    print('====> fix again <=====')
    model.train()

    try:
        model.module.features.features.eval()
    except:
        model.module.features.eval()

    try:
        model.module.neck.eval()
        model.module.neck.apply(BNtoFixed)
    except:
        pass

    try:
        model.module.connect_model.eval()
        model.module.connect_model.apply(BNtoFixed)
    except:
        pass

    try:
        model.module.bbox_tower.eval()
        model.module.bbox_tower.apply(BNtoFixed)
    except:
        pass

    try:
        model.module.features.features.apply(BNtoFixed)
    except:
        model.module.features.apply(BNtoFixed)

    model.module.mask_model.train()
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)

        mask = input[6].float().to(device)
        template_mask = input[-1].float().to(device)
        mask_weight = input[7].float().to(device)

        _, _, loss = model(template, search, label_cls, reg_target=reg_label, reg_weight=reg_weight,
                                              mask=mask, mask_weight=mask_weight, template_mask=template_mask)

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # gradient clip
        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t MASK Loss:{mask_loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time, mask_loss=losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.FREEMASK.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

# ===========================================================
# Main code for SiamDW train
# ===========================================================
def siamdw_train(train_loader, model,  optimizer, epoch, cur_lr, cfg, writer_dict, logger):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)  # you may try freeze-out

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.cuda()

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].cuda()
        search = input[1].cuda()
        label_cls = label_cls.cuda()

        loss = model(template, search, label_cls)
        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.data[0]):
            optimizer.step()

        # record loss
        loss = loss.data[0]
        losses.update(loss, template.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t Loss:{loss.avg:.5f}'.format(
                epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time, loss=losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg, cfg.SIAMFC.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict


def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
