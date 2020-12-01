import argparse
import os
import torch
import torch.nn as nn

def set_lambda(networks, lambda_):
    for n, l in zip(networks, lambda_):
        if n is None:
            continue
        n.set_lambda(l)
        
def get_optim_and_scheduler(networks, lrs, epochs, lr_steps, gamma):
    
    if not isinstance(networks, list):
        networks = [networks]
    
    params = []
    for network, lr in zip(networks, lrs):
        if network is not None:
            params += network.get_params(lr)
    if not isinstance(lr_steps, list):
        lr_steps = [lr_steps,] 
    optimizer = torch.optim.SGD(params, weight_decay=.0005, momentum=.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=gamma)
    
    return optimizer, scheduler

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def set_mode(model, mode="train"):
    if model is not None:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eva()

def save_options(opt, save_folder):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(save_folder, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
