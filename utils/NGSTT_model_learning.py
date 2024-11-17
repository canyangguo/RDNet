import torch
import time
from torch.distributions import Bernoulli, Normal, Uniform, Exponential, poisson
from utils.evaluation import masked_mae_np, masked_mape_np, masked_mse_np
from utils.logs import log_string
from utils.LRMC import start_imputer
import os
from tqdm import tqdm
from utils.learning_strategy import learning_strategy
import random
import numpy as np
from utils.random_seed import setup_seed
import copy


class model_learning:
    def __init__(self, config_args, args):
        self.model_args, self.task_args, self.data_args = config_args['model'], config_args['task'], config_args['data']
        self.args = args
        self.S2D = args.S2D
        self.EpT = args.EpT
        self.max_norm = self.data_args[args.source_dataset]['max_norm']
        self.criterion1 = torch.nn.HuberLoss(delta=1)
        self.criterion2 = torch.nn.HuberLoss(delta=1)
        self.mae_criterion = torch.nn.L1Loss(reduction='mean')
        self.ls = learning_strategy(args, self.model_args['num_of_layers'], self.task_args['pred_num'])

    def print_predction(self, info, log):
        horizon = 1
        for i in info:
            if horizon <= self.task_args['pred_num']:
                log_string(log, '@t' + str(horizon) + ' {:.3f} {:.3f} {:.3f}'.format(*i), use_info=False)
            else:
                log_string(log, 'avg {:.3f} {:.3f} {:.3f}\n'.format(*i), use_info=False)
            horizon = horizon + 1

    def get_in_out(self, data, label):
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        ti, di, data = data[:, -1, -1, 1].type(torch.LongTensor), \
                       data[:, -1, -1, 2].type(torch.LongTensor), \
                       data[..., :1]
        return data, label, ti, di

    def get_bank_in_out(self, target_memory_bank):
        for _, (mbx, mby) in enumerate(target_memory_bank):
            tid, mbx = mbx[:, -1, -1, 1].type(torch.LongTensor), mbx[..., :1]
            mbx = mbx.cuda(non_blocking=True)
            mby = mby.cuda(non_blocking=True).unsqueeze(-1)
            break
        return mbx, mby, tid


    def get_sparse_mask_train(self, data, mean_std, k):  # k应该好好选择,训练的时候高斯，测试用高斯，均匀，指数
        missing_rate = Uniform(0, 1.0).sample(data[:, 0, 0, 0].shape)
        mask = [Bernoulli(missing_rate[i]).sample(data[0].shape) for i in range(data.shape[0])]  # 伯努利分布制作缺失
        mask = torch.stack(mask, dim=0).cuda()

        generator = Normal(0.0, 1.0)
        delta = generator.sample(data.shape).cuda() * k * mask

        return data, mask, delta


    def get_sparse_mask(self, data, missing_rate, mean_std, k, dis='Normal'):
        missing_rate = Uniform(0, 1.0).sample(data[:, 0, 0, 0].shape)  # B
        mask = [Bernoulli(missing_rate[i]).sample(data[0].shape) for i in range(data.shape[0])]  # 伯努利分布制作缺失
        mask = torch.stack(mask, dim=0).cuda()  # B 0-1 random miss

        if dis == 'Normal':
            generator = Normal(0.0, 1.0)
        if dis == 'Uniform':
            generator = Uniform(-1.0, 1.0)
        if dis == 'Exponential':
            generator = Exponential(2)
        if dis == 'Poisson':
            generator = poisson.Poisson(0.5)

        delta = generator.sample(data.shape).cuda() * k * mask

        return data + delta, mask, delta


    # def get_bound(self, center, alpha=0.01):
    #     #k = np.ceil(1 / alpha**0.5)
    #     k = 1 / alpha ** 0.5
    #     bound = k * center
    #     return bound

    def start_training_stage(self, net, optimizer, loader, ep, epoch, tdx, samples, mean_std, stage, log, center=None, target_memory_bank=None, s2t_id=None):
        '''
        alpha1: decoupling,
        alpha2: smooth
        missing_rate 0-1
        k: 缺失的力度大小 0-1, 在get_sparse_mask_train, train=hyperparameter, val=0, test=0-1
        '''


        train_mae1, train_mae2 = 0, 0
        net.train()
        start_time = time.time()
        #bound = self.get_bound(center)
        with torch.enable_grad():
            for idx, (data, label) in enumerate(tqdm(loader)):
                data, label, ti, di = self.get_in_out(data, label)
                label = label[:, :tdx]
                if self.args.k > 0:
                    data, mask, delta = self.get_sparse_mask_train(data, mean_std, self.args.k)

                if stage == 'source':
                    y1, y2, lossvy = net(stage, data, ti, di, ep, tdx, self.S2D, mask=mask, delta=delta, learning=True)
                    train_mae1 = train_mae1 + self.mae_criterion(y1, label).item() * (y1.shape[0] / samples)
                    loss = self.args.alpha1 * self.criterion1(y1, label) + self.criterion2(y2, label) + self.args.alpha2 * lossvy
                    #loss = self.criterion1(y1, label)

                optimizer.zero_grad()
                loss.backward()
                #if self.max_norm:
                    #torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm, norm_type=2)
                optimizer.step()

        train_time = time.time() - start_time
        log_string(log, 'training: epoch: {}/{},  mae: {:.3f}, mae: {:.3f}, time: {:.3f}s'.format(ep + 1, epoch, train_mae1, train_mae2, train_time), use_info=False)
        return net, train_mae1, train_time

    def start_validation_stage(self, net, loader, ep, tdx, samples, mean_std, lowest_val_loss, stage, log, param_file, center, target_memory_bank=None, s2t_id=None):
        net.eval()
        val_mae = 0
        start_time = time.time()
        #bound = self.get_bound(center)
        with torch.no_grad():
            for idx, (data, label) in enumerate(tqdm(loader)):
                data, label, ti, di = self.get_in_out(data, label)
                label = label[:, :tdx]
                # if self.args.missing_rate > 0:
                #     data, mask, delta = self.get_sparse_mask_train(data, self.args.missing_rate, mean_std, 0)
                delta = 0
                if stage == 'source':
                    y1, y2, _ = net(stage, data, ti, di, ep, tdx, self.S2D, delta=delta)
                    val_mae = val_mae + self.mae_criterion(y1, label).item() * (y1.shape[0] / samples)

                elif stage == 'transition':
                    mbx, mby, tid = self.get_bank_in_out(target_memory_bank)
                    pre, domain_loss = net(stage, data, ti, di, ep, tdx, mbx=mbx, mby=mby, tid=tid)
                    val_mae = val_mae + domain_loss * (pre.shape[0] / samples)
                else:
                    pre = net(stage, data, ti, di, ep, tdx, s2t_id=s2t_id)
                    val_mae = val_mae + self.mae_criterion(pre, label).item() * (pre.shape[0] / samples)
        torch.save(net.state_dict(), param_file + 'epoch=' + str(ep + 1))

        if stage == 'source':
            log_string(log, 'validation: mae: {:.3f}, time: {:.3f}s'.format(val_mae, time.time() - start_time), use_info=False)
            if val_mae < lowest_val_loss:
                log_string(log, 'update best_model...')
                torch.save(net.state_dict(), param_file + 'best_model')
                lowest_val_loss = val_mae
        else:
            log_string(log, 'testing: mae: {:.3f}, time: {:.3f}s'.format(val_mae, time.time() - start_time), use_info=False)
            log_string(log, 'update current_model...', use_info=False)
        log_string(log, '\n', use_info=False)
        return lowest_val_loss, val_mae

    def start_test_stage(self, net, loader, stage, center, k, s2t_id=None, mean_std=None):
        #k = 0.5
        net.eval()
        with torch.no_grad():
            pres = []
            labels = []
            y = []
            iy = []
            vy = []
            for idx, (data, label) in enumerate(tqdm(loader)):
                data, label, ti, di = self.get_in_out(data, label)
                data, mask, delta = self.get_sparse_mask(data, 1, mean_std, k)

                y1, y2, _ = net(stage, data, ti, di, ep=self.task_args['source_epoch'], tdx=self.task_args['pred_num'], s2t_id=s2t_id, mask=None, delta=delta)
                pre = y1

                pres.append(pre.to('cpu'))
                labels.append(label.to('cpu'))
                y.append(y1.to('cpu'))

            #     iy.append(y2.to('cpu'))
            #     vy.append((y1-y2).to('cpu'))
            #
            # y = torch.cat(y, dim=0).detach().numpy()
            # iy = torch.cat(iy, dim=0).detach().numpy()
            # vy = torch.cat(vy, dim=0).detach().numpy()

            pres = torch.cat(pres, dim=0).detach().numpy()
            labels = torch.cat(labels, dim=0).detach().numpy()
            prediction_info = []
            for idx in range(self.task_args['pred_num']):
                y, x = labels[:, idx: idx + 1, :], pres[:, idx: idx + 1, :]
                prediction_info.append((masked_mae_np(y, x, 0), masked_mse_np(y, x, 0) ** 0.5, masked_mape_np(y, x, 0)))

            prediction_info.append((
                masked_mae_np(labels, pres, 0),
                masked_mse_np(labels, pres, 0) ** 0.5,
                masked_mape_np(labels, pres, 0)
                ))
        import pandas as pd

        # pd.DataFrame(y[:, 0, 0]).to_csv('wioRy.csv')
        # pd.DataFrame(iy[:, 0, 0]).to_csv('wioRiy.csv')
        # pd.DataFrame(vy[:, 0, 0]).to_csv('wioRvy.csv')
        #pd.DataFrame(pres[:, 0, 0]).to_csv('pres-k=0.5.csv')
        #pd.DataFrame(pres[:, 0, 0]).to_csv('wio-S-pres-k=0.5.csv')
        #pd.DataFrame(labels[:, 0, 0]).to_csv('labels.csv')
        #print(aa)
        return prediction_info


    def load_current_model(self, stage, net, epoch, tdx, param_file, log, loader, samples, mean_std, optimizer, lowest_val_loss, center, target_memory_bank=None, s2t_id=None):
        for ep_id in range(epoch):
            if stage == 'source':
                if ep_id == 70:
                    optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'] * 0.5)
                if ep_id == 80:
                    optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'] * 0.1)
                if (tdx < self.task_args['pred_num']) and (ep_id % self.args.EpT == 0):
                    tdx = tdx + 1
                #    net, tdx = self.ls.select_learning_strategy(net, 'step_increase', log, tdx, ep_id)
                # if ep_id == self.S2D:
                #     net = self.ls.select_learning_strategy(net, 'sta2dyn', log)

            if os.path.exists(param_file + 'epoch={}'.format(ep_id+1)):
                continue
            else:
                if stage == 'source':
                    net.load_state_dict(torch.load(param_file + 'best_model'))
                    lowest_val_loss, val_mae = self.start_validation_stage(net, loader[1], ep_id, tdx, samples[1], mean_std,
                                                                       lowest_val_loss, stage, log, param_file, center)
                    info = self.start_test_stage(net, loader[2], stage, center)
                    self.print_predction(info, log)

                    net.load_state_dict(torch.load(param_file + 'epoch={}'.format(ep_id)))
                else:
                    net.load_state_dict(torch.load(param_file + 'epoch={}'.format(ep_id)))
                    lowest_val_loss, val_mae = self.start_validation_stage(net, loader, ep_id, tdx, samples, mean_std,
                                                                       lowest_val_loss, stage, log, param_file, target_memory_bank, s2t_id)
                log_string(log, 'continue to training...\n')
                break
        return net, ep_id, lowest_val_loss, optimizer, tdx


    def source_learning(self, net, param_file, loader, samples, log, mean_std, center=None):
        stage = 'source'
        epoch = self.task_args['source_epoch']
        lowest_val_loss = 1e6
        ep_id = 0
        train_maes, val_maes, train_times = [], [], []
        optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'])  # weight_decay=wd
        if self.args.step_increase:
            tdx = 0
        else:
            tdx = self.task_args['pred_num']

        if os.path.exists(param_file + 'epoch={}'.format(ep_id+1)):
            net, ep_id, lowest_val_loss, optimizer, tdx = self.load_current_model(
            stage, net, epoch, tdx, param_file, log, loader, samples, mean_std, optimizer, lowest_val_loss, center)

        for ep in range(ep_id, epoch):
            if ep == 70:
                optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'] * 0.5)
            if ep == 80:
                optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'] * 0.1)
            if (tdx < self.task_args['pred_num']) and (ep % self.args.EpT == 0):
                tdx = tdx + 1
                lowest_val_loss = 1e6
            #    net, tdx = self.ls.select_learning_strategy(net, 'step_increase', log, tdx, ep_id)
            # if ep == self.S2D:
            #     net = self.ls.select_learning_strategy(net, 'sta2dyn', log)

            net, train_mae, train_time = self.start_training_stage(net, optimizer, loader[0], ep, epoch, tdx, samples[0], mean_std, stage, log, center)
            lowest_val_loss, val_mae = self.start_validation_stage(net, loader[1], ep, tdx, samples[1], mean_std, lowest_val_loss, stage, log, param_file, center)

            train_maes.append(train_mae)
            val_maes.append(val_mae)
            train_times.append(train_time)
        log_string(log, 'total training time: {}'.format(sum(train_times)))
        log_string(log, 'average training time: {}\n'.format(sum(train_times) / epoch))
        return net

    def transition_learning(self, net, param_file, loader, target_memory_bank, samples, log):
        stage = 'transition'
        epoch = self.task_args['transition_epoch']
        lowest_val_loss = 1e6
        ep_id = 0
        train_maes, val_maes, train_times = [], [], []
        optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['transition_lr'])  # weight_decay=wd
        tdx = self.task_args['pred_num']
        if os.path.exists(param_file + 'epoch={}'.format(ep_id+1)):
            net, ep_id, lowest_val_loss, optimizer, tdx = self.load_current_model(
            stage, net, epoch, tdx, param_file, log, loader[0], samples[0], optimizer, lowest_val_loss, target_memory_bank)

        for ep in range(ep_id, epoch):
            net, train_mae, train_time = self.start_training_stage(net, optimizer, loader[0], ep, epoch, tdx, samples[0], stage, log, target_memory_bank)
            lowest_val_loss, val_mae = self.start_validation_stage(net, loader[2], ep, tdx, samples[2], lowest_val_loss, stage, log, param_file, target_memory_bank)

            train_maes.append(train_mae)
            val_maes.append(val_mae)
            train_times.append(train_time)
        return net

    def finetuning_learning(self, net, param_file, loader, samples, log, s2t_id):
        stage = 'finetuning'
        epoch = self.task_args['finetuning_epoch']
        lowest_val_loss = 1e6
        ep_id = 0
        train_maes, val_maes, train_times = [], [], []
        optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['finetuning_lr'])  # weight_decay=wd
        tdx = self.task_args['pred_num']

        if os.path.exists(param_file + 'epoch={}'.format(ep_id+1)):
            net, ep_id, lowest_val_loss, optimizer, tdx = self.load_current_model(
            stage, net, epoch, tdx, param_file, log, loader[0], samples[0], optimizer, lowest_val_loss, s2t_id=s2t_id)

        for ep in range(ep_id, epoch):
            net, train_mae, train_time = self.start_training_stage(net, optimizer, loader[0], ep, epoch, tdx, samples[0], stage, log, s2t_id=s2t_id)
            lowest_val_loss, val_mae = self.start_validation_stage(net, loader[2], ep, tdx, samples[2], lowest_val_loss, stage, log, param_file, s2t_id=s2t_id)
            train_maes.append(train_mae)
            val_maes.append(val_mae)
            train_times.append(train_time)

        return net

    def svt_denoising(self, data, center, bound, mean_std, rho=1e-2, epsilon=1e-4, maxiter=10):
        data = (data * mean_std[1]) + mean_std[0]


        pos_missing = torch.where(torch.abs(data - center) > bound)
        last_X = copy.deepcopy(data)
        snorm = torch.linalg.norm(data, 'fro')
        T = torch.zeros(data.shape).cuda()
        #Z = copy.deepcopy(data)
        Z = torch.where(torch.abs(data - center) > bound, center, data)
        it = 0

        while True:
            rho = min(rho * 1.05, 1e5)
            X = self.svt(Z - T / rho, 1 / rho)
            Z[pos_missing] = (X + T / rho)[pos_missing]
            T = T + rho * (X - Z)
            tol = torch.linalg.norm((X - last_X), 'fro') / snorm
            last_X = copy.deepcopy(X)
            it += 1
            # if it % 1 == 0:
            #     print('Iter: {}'.format(it))
            #     print('Tolerance: {:.6}'.format(tol))
            #     print()
            if (tol < epsilon) or (it >= maxiter):
                break

        data = (X - mean_std[0]) / mean_std[1]
        return data

    def svt(self, mat, tau):
        u, s, v = torch.linalg.svd(mat, full_matrices=False)
        idx = torch.sum(s > tau)
        return u[:, : idx] @ torch.diag(s[: idx] - tau) @ v[: idx, :]