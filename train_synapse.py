import argparse
import logging
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from lib.abl import build_model
from trainer import trainer_synapse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/synapse/train_npz_new', help='root dir for data')
parser.add_argument('--volume_path', type=str, default='./data/synapse/test_vol_h5_new', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iterations to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
parser.add_argument('--no_tb', type=int, default=0, help='disable tensorboard writer (sandbox-safe)')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0002, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--gpu', type=int, default=-1, help='physical GPU index via set_device (bypasses CUDA_VISIBLE_DEVICES)')
parser.add_argument('--seed', type=int, default=2222, help='random seed')
# ---- ablation knobs ----
parser.add_argument('--arch', type=str, default='a_b_c_unified', help='model architecture (full model: a_b_c_unified)')
parser.add_argument('--model_scale', type=str, default='small', help='tiny | small')
parser.add_argument('--ug_alpha', type=float, default=0.0, help='UG-MUTATION temperature; 0 == standard MUTATION')
parser.add_argument('--tag', type=str, default='', help='extra tag for the exp dir name')
parser.add_argument('--save_tail', type=int, default=0, help='offline mode: no eval during training, dump every-epoch ckpt in last N epochs -> pick best offline')
parser.add_argument('--focal', type=int, default=0, help='use FocalLoss(gamma=2) instead of CE in MUTATION (small-organ focus)')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='FocalLoss focusing strength')
parser.add_argument('--focal_adaptive', type=int, default=0, help='adaptive per-class inverse-freq weighting in FocalLoss')
parser.add_argument('--clip_grad', type=float, default=8.0, help='grad clip norm (AgileFormer recipe=8); 0=off')
parser.add_argument('--warmup_epochs', type=int, default=20, help='cosine+warmup lr schedule (AgileFormer=20); 0=constant lr')
parser.add_argument('--weak_boost', type=float, default=1.0, help='class-imbalance: CE weight multiplier for weak organs (c4 gallbladder, c5 pancreas); 1.0 = off')
parser.add_argument('--eval_interval', type=int, default=10, help='run full-volume eval every K epochs')
parser.add_argument('--eval_dense_last', type=int, default=0, help='also eval every epoch in the last M epochs')
parser.add_argument('--init_ckpt', type=str, default='', help='optional checkpoint to initialize from; loaded strict=False')
parser.add_argument('--train_only_c', type=int, default=0, help='freeze all params except decoder.C* skip adapters')
parser.add_argument('--train_only_c_gate', type=int, default=0, help='freeze all params except decoder.C*.gate scalars')

args = parser.parse_args()

if __name__ == "__main__":
    if getattr(args, "gpu", -1) >= 0:
        torch.cuda.set_device(args.gpu)
        print("set_device -> physical GPU", args.gpu, "| name", torch.cuda.get_device_name(args.gpu), flush=True)
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # warn_only=True: deterministic where available; ops w/o det impl (grid_sample in
        # deformable models) stay non-det but only warn instead of crashing.
        torch.use_deterministic_algorithms(True, warn_only=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    # ---- experiment dir: <arch>_ug<alpha>_<dataset><img>_<tag>/seed<seed>_<time> ----
    exp = '{}_ug{}_{}{}'.format(args.arch, args.ug_alpha, dataset_name, args.img_size)
    if args.tag:
        exp = exp + '_' + args.tag
    args.exp = exp
    current_time = time.strftime("%H%M%S")
    snapshot_path = "model_pth/{}/seed{}_ep{}_bs{}_lr{}_{}".format(
        exp, args.seed, args.max_epochs, args.batch_size, args.base_lr, current_time)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = build_model(args.arch, n_class=args.num_classes, img_size=args.img_size,
                      model_scale=args.model_scale).cuda()

    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location='cpu')
        if state and all(k.startswith('module.') for k in state.keys()):
            state = {k[len('module.'):]: v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        print('Initialized from %s | missing=%d unexpected=%d' % (
            args.init_ckpt, len(missing), len(unexpected)))
        if missing:
            print('  missing sample:', missing[:12])
        if unexpected:
            print('  unexpected sample:', unexpected[:12])

    if args.train_only_c_gate:
        for name, p in net.named_parameters():
            p.requires_grad = name in ('decoder.C1.gate', 'decoder.C2.gate', 'decoder.C3.gate')
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('train_only_c_gate enabled, trainable params: %d' % trainable)
    elif args.train_only_c:
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith(('decoder.C1', 'decoder.C2', 'decoder.C3'))
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('train_only_c enabled, trainable params: %.3fM' % (trainable / 1e6))

    n_params = sum(m.numel() for m in net.parameters())
    print('Model [%s] created, param count: %.2fM' % (args.arch, n_params / 1e6))

    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path)
