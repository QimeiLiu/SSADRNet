import argparse
import logging
import math
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.training_utils import DiceLoss, FocalLoss, powerset, val_single_volume
from lib.abl import mutation_loss, adamw_param_groups

            
def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=getattr(args, 'num_workers', 8))
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
        
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        # seed ALL RNGs per-worker. Augmentation uses np.random.randint (flip k/axis, rotate angle)
        # but only `random` was seeded before -> np.random was the MAIN non-determinism source
        # (worker fork-order non-deterministic). Now reproducible regardless of worker scheduling.
        s = args.seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    g = torch.Generator()
    g.manual_seed(args.seed)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=getattr(args, 'num_workers', 8), pin_memory=True,
                             worker_init_fn=worker_init_fn, generator=g)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    if getattr(args, 'train_only_c_gate', 0):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    elif getattr(args, 'train_only_c', 0):
        for name, module in model.named_modules():
            if not name.startswith(('decoder.C1', 'decoder.C2', 'decoder.C3')) \
                    and isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    # class-imbalance: optionally up-weight the weak small organs (c4 gallbladder, c5 pancreas) in CE
    wb = getattr(args, 'weak_boost', 1.0)
    if wb and wb != 1.0:
        cw = torch.ones(num_classes)
        if num_classes > 5:
            cw[4] = wb; cw[5] = wb                              # gallbladder, pancreas
        ce_loss = CrossEntropyLoss(weight=cw.cuda())
        print('class-imbalance: CE weak_boost=%.2f on c4(gallbladder)/c5(pancreas)' % wb)
    else:
        ce_loss = CrossEntropyLoss()
    if getattr(args, 'focal', 0):
        fg = getattr(args, 'focal_gamma', 2.0)
        fa = bool(getattr(args, 'focal_adaptive', 0))
        ce_loss = FocalLoss(gamma=fg, adaptive=fa, num_classes=num_classes)
        print('using FocalLoss(gamma=%.1f, adaptive=%s) instead of CE (small-organ focus)' % (fg, fa))
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(adamw_param_groups(model, 0.0001), lr=base_lr)
    if getattr(args, 'no_tb', 0):
        class _DummyWriter:
            def add_scalar(self, *args, **kwargs):
                pass
            def close(self):
                pass
        writer = _DummyWriter()
    else:
        writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    ss = None  # built lazily from the model's actual number of prediction maps
    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()

            P = model(image_batch)
            if ss is None:
                ss = [x for x in powerset(list(range(len(P))))]
                print('MUTATION subsets over %d maps:' % len(P), len(ss))

            lc1, lc2 = 0.3, 0.7
            loss = mutation_loss(P, label_batch, ce_loss, dice_loss, ss,
                                 lc1=lc1, lc2=lc2, ug_alpha=getattr(args, 'ug_alpha', 0.0))

            # lr schedule: cosine+warmup if --warmup_epochs>0 (AgileFormer recipe), else constant
            we = getattr(args, 'warmup_epochs', 0)
            if we and we > 0:
                wu = max(1, we * len(trainloader))
                if iter_num < wu:
                    lr_ = base_lr * (iter_num + 1) / wu
                else:
                    prog = (iter_num - wu) / max(1, max_iterations - wu)
                    lr_ = base_lr * 0.5 * (1 + math.cos(math.pi * prog))
            else:
                lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            optimizer.zero_grad()
            loss.backward()
            cg = getattr(args, 'clip_grad', 0.0)
            if cg and cg > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cg)
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))                
             
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))        
       
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)

        # OFFLINE mode (--save_tail N>0): no eval during training (uninterrupted); just dump every
        # epoch's checkpoint in the last N epochs -> pick best offline with the fast batched reeval.
        save_tail = getattr(args, 'save_tail', 0)
        if save_tail and save_tail > 0:
            if epoch_num >= max_epoch - save_tail:
                torch.save(model.state_dict(), os.path.join(snapshot_path, 'tail_%d.pth' % epoch_num))
            if epoch_num >= max_epoch - 1:
                iterator.close()
                break
            continue

        # eval schedule: every eval_interval epochs, plus every epoch in the last
        # eval_dense_last epochs (where the best epoch usually lands), plus the final.
        eval_interval = getattr(args, 'eval_interval', 1)
        eval_dense_last = getattr(args, 'eval_dense_last', 0)
        do_eval = ((epoch_num + 1) % eval_interval == 0) \
            or (epoch_num >= max_epoch - eval_dense_last) \
            or (epoch_num >= max_epoch - 1)

        save_interval = 50

        if do_eval:
            performance = inference(args, model, best_performance)
            if(best_performance <= performance):
                best_performance = performance
                save_mode_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
