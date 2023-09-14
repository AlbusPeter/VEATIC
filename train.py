import os
import math
import tempfile
import argparse

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import torch.nn as nn

from dataset import VEATIC

from model import VEATIC_baseline

from multi_train_utils.distributed_utils import dist, cleanup
from multi_train_utils.train_utils import train_one_epoch, evaluate


def main_fun(rank, world_size, args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    args.rank = rank
    args.world_size = world_size
    args.gpu = rank

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    save_path = args.save
    data_path = args.data_path
    args.lr *= args.world_size

    if rank == 0: 
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists(save_path) is False and not args.test:
            os.makedirs(save_path)

    train_data_set = VEATIC(character_dir=data_path + 'frames',
                            csv_path=data_path + 'rating_averaged',
                            split=0.7, 
                            mode='train')
    val_data_set = VEATIC(character_dir=data_path + 'frames',
                          csv_path=data_path + 'rating_averaged',
                          split=0.7, 
                          mode='test')


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 0
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             shuffle=False)

    model = VEATIC_baseline().to(device)

    if weights_path != ' ' and os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "feature" in name:
                para.requires_grad_(False)
    else:
        if args.syncBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    # print(model)
    # for name, para in model.named_parameters():
    #     if para.requires_grad:
    #         print(name)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf 
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    eval_best_score = 0
    
    if args.test:
        val_ccc, aro_ccc, val_pcc, \
        aro_pcc, val_rmse, aro_rmse, val_sagr, aro_sagr = evaluate(model=model,
                                                                    data_loader=val_loader,
                                                                    device=device)
        
        if rank == 0:
            print("[test] val_ccc: {} aro_ccc: {}".format(val_ccc, aro_ccc))
            print("[test] val_pcc: {} aro_pcc: {}".format(val_pcc, aro_pcc))
            print("[test] val_rmse: {} aro_rmse: {}".format(val_rmse, aro_rmse))
            print("[test] val_sagr: {} aro_sagr: {}".format(val_sagr, aro_sagr))
        
    else:
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            mean_loss, train_val_ccc, \
            train_aro_ccc, train_val_pcc, \
            train_aro_pcc, train_val_rmse, \
            train_aro_rmse, train_val_sagr, train_aro_sagr = train_one_epoch(model=model,
                                                                            optimizer=optimizer,
                                                                            data_loader=train_loader,
                                                                            device=device,
                                                                            epoch=epoch)
            
            scheduler.step()

            val_ccc, aro_ccc, val_pcc, \
            aro_pcc, val_rmse, aro_rmse, val_sagr, aro_sagr = evaluate(model=model,
                                                                        data_loader=val_loader,
                                                                        device=device)

            if rank == 0:
                print("[epoch {}] mean_loss: {}".format(epoch, mean_loss))
                print("[epoch {}] train_val_ccc: {} train_aro_ccc: {}".format(epoch, train_val_ccc, train_aro_ccc))
                print("[epoch {}] train_val_pcc: {} train_aro_pcc: {}".format(epoch, train_val_pcc, train_aro_pcc))
                print("[epoch {}] train_val_rmse: {} train_aro_rmse: {}".format(epoch, train_val_rmse, train_aro_rmse))
                print("[epoch {}] train_val_sagr: {} train_aro_sagr: {}".format(epoch, train_val_sagr, train_aro_sagr))

                
                print("[test] val_ccc: {} aro_ccc: {}".format(val_ccc, aro_ccc))
                print("[test] val_pcc: {} aro_pcc: {}".format(val_pcc, aro_pcc))
                print("[test] val_rmse: {} aro_rmse: {}".format(val_rmse, aro_rmse))
                print("[test] val_sagr: {} aro_sagr: {}".format(val_sagr, aro_sagr))

                tags = ["loss", "val_ccc", "aro_ccc", "learning_rate"]
                tb_writer.add_scalar(tags[0], mean_loss, epoch)
                tb_writer.add_scalar(tags[1], train_val_ccc, epoch)
                tb_writer.add_scalar(tags[2], train_aro_ccc, epoch)
                tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
                
                torch.save(model.module.state_dict(), save_path + "veatic.pth") # change the name by yourself
                
                if (epoch + 1) % 5 == 0:
                    torch.save(model.module.state_dict(), save_path + "veatic_{}.pth".format(epoch + 1))

                if( val_ccc + aro_ccc) / 2 > eval_best_score:
                    eval_best_score = (val_ccc + aro_ccc) / 2
                    torch.save(model.module.state_dict(), save_path + "veatic_eval_best_{}.pth".format(epoch + 1))

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--syncBN', type=bool, default=False)

    parser.add_argument('--weights', type=str, default=' ', help='initial weights path')
    parser.add_argument('--save', type=str, default='./ckpt/', help='save weights path')
    parser.add_argument('--data_path', default='./data/', help='data path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    # when using mp.spawn, if I set number of works greater 1,
    # before each epoch training and validation will wait about 10 seconds

    # mp.spawn(main_fun,
    #          args=(opt.world_size, opt),
    #          nprocs=opt.world_size,
    #          join=True)

    world_size = opt.world_size
    processes = []
    for rank in range(world_size):
        p = Process(target=main_fun, args=(rank, world_size, opt))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

