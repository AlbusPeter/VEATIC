import sys

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_train_utils.distributed_utils import reduce_value, is_main_process

def ccc(input, target):
    input_mean = input.mean()
    target_mean = target.mean()
    cross_corr = torch.cov(torch.stack([input, target]))[0, 1]
    ccc = 2 * cross_corr / ((target.std()**2 + input.std()**2 + (target_mean - input_mean)**2) + (10**-5))
    return 1 - ccc

def total_ccc(v_input, a_input, v_target, a_target):
    rho_v = ccc(v_input, v_target)
    rho_a = ccc(a_input, a_target)
    return (rho_v + rho_a) / 2

def score(input, target):
    input_mean = input.mean()
    target_mean = target.mean()
    cross_corr = torch.cov(torch.stack([input, target]))[0, 1]
    ccc = 2 * cross_corr / ((target.std()**2 + input.std()**2 + (target_mean - input_mean)**2) + (10**-5))
    pcc = torch.corrcoef(torch.stack([input, target]))[0, 1]
    rmse = torch.sqrt(F.mse_loss(input, target, reduction='mean'))
    sagr = (torch.mul(input, target) > 0).sum() / input.shape[0]
    return (ccc, pcc, rmse, sagr)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = nn.MSELoss()

    mean_loss = torch.zeros(1).to(device)
    val_ccc = torch.zeros(1).to(device)
    aro_ccc = torch.zeros(1).to(device)
    val_pcc = torch.zeros(1).to(device)
    aro_pcc = torch.zeros(1).to(device)
    val_rmse = torch.zeros(1).to(device)
    aro_rmse = torch.zeros(1).to(device)
    val_sagr = torch.zeros(1).to(device)
    aro_sagr = torch.zeros(1).to(device)
    optimizer.zero_grad()

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        mse_loss = loss_function(pred, labels.to(device))
        loss = total_ccc(pred[:,0], pred[:,1], labels[:,0].to(device), labels[:,1].to(device)) + 0.1 * mse_loss
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        v = score(pred[:,0], labels[:,0].to(device))
        a = score(pred[:,1], labels[:,1].to(device))
        
        val_ccc = (val_ccc * step + v[0].detach()) / (step + 1)
        aro_ccc = (aro_ccc * step + a[0].detach()) / (step + 1)
        val_pcc = (val_pcc * step + v[1].detach()) / (step + 1)
        aro_pcc = (aro_pcc * step + a[1].detach()) / (step + 1)
        val_rmse = (val_rmse * step + v[2].detach()) / (step + 1)
        aro_rmse = (aro_rmse * step + a[2].detach()) / (step + 1)
        val_sagr = (val_sagr * step + v[3].detach()) / (step + 1)
        aro_sagr = (aro_sagr * step + a[3].detach()) / (step + 1)

        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, mean_loss.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item(), val_ccc.item(), aro_ccc.item(), val_pcc.item(), aro_pcc.item(), val_rmse.item(), aro_rmse.item(), val_sagr.item(), aro_sagr.item()



@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    val_ccc = torch.zeros(1).to(device)
    aro_ccc = torch.zeros(1).to(device)
    val_pcc = torch.zeros(1).to(device)
    aro_pcc = torch.zeros(1).to(device)
    val_rmse = torch.zeros(1).to(device)
    aro_rmse = torch.zeros(1).to(device)
    val_sagr = torch.zeros(1).to(device)
    aro_sagr = torch.zeros(1).to(device)

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))
        
        v = score(pred[:,0], labels[:,0].to(device))
        a = score(pred[:,1], labels[:,1].to(device))
        
        val_ccc = (val_ccc * step + v[0].detach()) / (step + 1)
        aro_ccc = (aro_ccc * step + a[0].detach()) / (step + 1)
        val_pcc = (val_pcc * step + v[1].detach()) / (step + 1)
        aro_pcc = (aro_pcc * step + a[1].detach()) / (step + 1)
        val_rmse = (val_rmse * step + v[2].detach()) / (step + 1)
        aro_rmse = (aro_rmse * step + a[2].detach()) / (step + 1)
        val_sagr = (val_sagr * step + v[3].detach()) / (step + 1)
        aro_sagr = (aro_sagr * step + a[3].detach()) / (step + 1)

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return val_ccc.item(), aro_ccc.item(), val_pcc.item(), aro_pcc.item(), val_rmse.item(), aro_rmse.item(), val_sagr.item(), aro_sagr.item()