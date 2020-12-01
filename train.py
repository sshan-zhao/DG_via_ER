import argparse
import torch
from torch import nn
from torch.nn import functional as F

from models import model_factory
from data.dataset import available_datasets, get_train_dataloader, get_val_dataloader, dataset
from utils import save_options, get_optim_and_scheduler, set_mode, set_requires_grad, set_lambda

import os
import random
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Script to train DG_VIA_ER", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["PACS"], help="Mulit-domain dataset", default="PACS")
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target", default="art_painting")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--data_dir", default="./dataset", help="Data directory")
    parser.add_argument("--datalist_dir", default="./datalist", help="Data list directory")

    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")

    parser.add_argument("--lr", type=float, default=.001, help="Learning rate for main model F and T")
    parser.add_argument("--lr_c", type=float, default=.0001, help="Learning rate for classifier T_i")
    parser.add_argument("--lr_cp", type=float, default=.0001, help="Learning rate for classifier, T_i^'")
    parser.add_argument("--lr_d", type=float, default=.001, help="Learning rate for discriminator")
    parser.add_argument("--lbd_c", type=float, default=0.05, help="Weight for classifier T_i")
    parser.add_argument("--lbd_cp", type=float, default=0.001, help="Weight for classifier T_i^' (GRL)")
    parser.add_argument("--lbd_d", type=float, default=0.1, help="Weight for discriminator (GRL)")
    
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr_steps", type=int, default=60, nargs='+', help='Step size of LR decay')  
    parser.add_argument("--lr_gamma", type=float, default=0.1, help='Multiplicative factor of LR decay')  
    parser.add_argument("--warmup_step", type=int, default=10)  
    parser.add_argument("--warmup_weight", type=float, default=0.01)  

    parser.add_argument("--num_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="resnet18", help="Which network to use")
    parser.add_argument("--exp_folder", default="experiments", help="Directory for logs and models")
    
    return parser.parse_args()

class Trainer:
    def __init__(self, args):
        args.source = [d for d in dataset[args.dataset] if d != args.target]
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        main_model, dis_model, c_model, cp_model = model_factory.get_network(args.network)(num_classes=args.num_classes, 
                                        num_domains=len(args.source))
        self.main_model = self._model2device(main_model)
        self.dis_model = self._model2device(dis_model)
        self.c_model = self._model2device(c_model)
        self.cp_model = self._model2device(cp_model)

        self.source_loader_list, self.val_loader, self.img_num_per_domain = get_train_dataloader(args)
        self.target_loader = get_val_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        print("Dataset size: train %d, val %d, test %d" % (sum(self.img_num_per_domain), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler([self.main_model, self.dis_model, self.c_model, self.cp_model], 
                                        [args.lr, args.lr_d, args.lr_c, args.lr_cp], epochs=args.epochs, lr_steps=args.lr_steps, gamma=args.lr_gamma)
        
        self.num_classes = args.num_classes
        self.num_domains = len(args.source)

        self.base_dir = os.path.join(args.exp_folder, args.network, args.dataset)
        self.save_dir = os.path.join(self.base_dir, args.target)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_options(args, self.save_dir)
        self.log_file = os.path.join(self.save_dir, "loss_log.txt")

    def _model2device(self, model):

        if model is None:
            return None
        model.to(self.device)
        return model

    def _compute_dis_loss(self, feature, domains):
        if self.dis_model is not None:
            domain_logit = self.dis_model(feature)
            weight = [1.0 / img_num for img_num in self.img_num_per_domain]
            weight = torch.FloatTensor(weight).to(self.device)
            weight = weight / weight.sum() * self.num_domains
            domain_loss = F.cross_entropy(domain_logit, domains, weight=weight)
        else:
            domain_loss = torch.zeros(1, requires_grad=True).to(self.device)
        
        return domain_loss

    def _compute_cls_loss(self, model, feature, label, domain, mode="self"):
        if model is not None:
            feature_list = []
            label_list = []
            weight_list = []
            for i in range(self.num_domains):
                if mode == "self":
                    feature_list.append(feature[domain==i])
                    label_list.append(label[domain==i])
                else:
                    feature_list.append(feature[domain!=i])
                    label_list.append(label[domain!=i])
                weight = torch.zeros(self.num_classes).to(self.device)
                for j in range(self.num_classes):
                    weight[j] = 0 if (label_list[-1]==j).sum() == 0 else 1.0 / (label_list[-1]==j).sum().float() 
                weight = weight / weight.sum()
                weight_list.append(weight)
            class_logit = model(feature_list)
            loss = 0
            for p, l, w in zip(class_logit, label_list, weight_list):
                if p is None:
                    continue
                loss += F.cross_entropy(p, l, weight=w) / self.num_domains
        else:
            loss = torch.zeros(1, requires_grad=True).to(self.device)
        
        return loss
    
    def _do_epoch(self):
        
        set_mode(self.main_model, "train")
        set_mode(self.dis_model, "train")
        set_mode(self.c_model, "train")
        set_mode(self.cp_model, "train")

        set_lambda([self.dis_model], 
                    [self.args.lbd_d])
        set_lambda([self.c_model, self.cp_model], 
                    [self.args.lbd_c, self.args.lbd_cp])
        loader_iter_list = []
        loader_size_list = []

        if self.current_epoch < self.args.warmup_step:
            aux_weight = self.args.warmup_weight
            main_weight = self.args.warmup_weight
        else:
            aux_weight = 1
            main_weight = 1

        for loader in self.source_loader_list:
            loader_iter_list.append(enumerate(loader))
            loader_size_list.append(len(loader))

        for it in range(max(loader_size_list)):
            data = []
            labels = []
            domains = []
            for idx, iter_ in zip(range(self.num_domains), loader_iter_list):
                try:
                    item = iter_.__next__()
                except StopIteration:
                    loader_iter_list[idx] = enumerate(self.source_loader_list[idx])
                    item = loader_iter_list[idx].__next__()
                data.append(item[1][0])
                labels.append(item[1][1])
                domains.append(torch.ones(labels[-1].size(0)).long()*idx)
            data = torch.cat(data, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)
            domains = torch.cat(domains, dim=0).to(self.device)

            set_requires_grad(self.main_model, False)
            set_requires_grad(self.c_model, True)
            _, feature = self.main_model(data)
            c_loss_self = self._compute_cls_loss(self.c_model, feature.detach(), labels, domains, mode="self") * aux_weight
            self.optimizer.zero_grad()
            c_loss_self.backward()
            self.optimizer.step()

            set_requires_grad([self.main_model, self.dis_model, self.c_model, self.cp_model], True)
            class_logit, feature = self.main_model(data)
            main_loss = F.cross_entropy(class_logit, labels) * main_weight
            
            dis_loss = self._compute_dis_loss(feature, domains) * aux_weight
            
            set_requires_grad(self.c_model, False)
            c_loss_others = self._compute_cls_loss(self.c_model, feature, labels, domains, mode="others") * aux_weight

            cp_loss = self._compute_cls_loss(self.cp_model, feature, labels, domains, mode="self") * aux_weight

            loss = dis_loss + c_loss_others + cp_loss + main_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss += c_loss_self

            message = "epoch %d iter %d: all %.6f main %.6f dis %.6f c_self %.6f c_others %.6f cp %.6f\n" % (self.current_epoch, it,
                            loss.data, main_loss.data, dis_loss.data, c_loss_self.data, c_loss_others.data, cp_loss.data)
            with open(self.log_file, "a") as fid:
                fid.write(message)
            print(message)
            
            del loss, main_loss, dis_loss, c_loss_self, c_loss_others, cp_loss

        self.main_model.eval()
        with torch.no_grad():
            with open(self.log_file, "a") as fid:
                for phase, loader in self.test_loaders.items():
                    class_correct, all_domains = self.do_test(loader)
                    class_correct = class_correct.float()
                    class_acc = class_correct.mean() * 100.0
                    self.results[phase][self.current_epoch] = class_acc
                    if phase == "val":
                        message = "epoch %d: val_all_acc %.5f"%(self.current_epoch, class_acc)
                        for i in range(self.num_domains):
                            cc_i = class_correct[all_domains == i]
                            ca_i = cc_i.mean() * 100.0
                            message += " val_%s_acc %.5f"%(self.args.source[i], ca_i)
                        message += "\n"
                        fid.write(message)
                        print(message)
                    elif phase == "test":
                        message = "epoch %d: test_acc %.5f\n"%(self.current_epoch, class_acc)
                        fid.write(message)
                        print(message)

    def do_test(self, loader):
        class_correct = []
        all_domains = []
        for _, ((data, labels), domains) in enumerate(loader):
            data, labels, domains = data.to(self.device), labels.to(self.device), domains.to(self.device)
            class_logit, _ = self.main_model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct.append(cls_pred == labels.data)
            all_domains.append(domains)
        return torch.cat(class_correct, 0), torch.cat(all_domains, 0)

    def do_training(self):

        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        
        for self.current_epoch in range(self.args.epochs):
            self._do_epoch()
            self.scheduler.step()
            
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        message = "Best val %.5f (epoch: %d), corresponding test %.5f\n" % (val_res.max(), 
                                                                        idx_best, test_res[idx_best])
        print(message)
        with open(self.log_file, "a") as fid:
            fid.write(message)
        
if __name__ == "__main__":
    args = get_args()
    trainer = Trainer(args)
    trainer.do_training()
    
