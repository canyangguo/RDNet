import torch
import argparse
import yaml
import os

from utils.data_process import data_loader
from utils.logs import log_string
from utils.path_generate import start_generating_path
from model.basic_module import CMD
from utils.random_seed import setup_seed
from thop import profile
from utils.NGSTT_model_learning import model_learning

parser = argparse.ArgumentParser(description='description')
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--task', type=str)
parser.add_argument('--source_dataset', type=str)
parser.add_argument('--target_dataset', type=str)
parser.add_argument('--target_days', default=3, type=int)
parser.add_argument('--num_of_layer', default=4, type=int)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--num_of_latent', default=32, type=int)
parser.add_argument('--S2D', default=80, type=int)
parser.add_argument('--EpT', default=5, type=int)
parser.add_argument("--gpu", type=str, default='1', help="gpu ID")

parser.add_argument('--alpha1', default=0.01, type=float, help='decoupling')
parser.add_argument('--alpha2', default=0.0001, type=float, help='smooth')
parser.add_argument('--k', default=1, type=float, help='perturbation')

parser.add_argument('--beta', type=float)
parser.add_argument('--seed', default=0, type=str)
parser.add_argument('--remark1', type=str)
parser.add_argument('--remark2', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--step_increase', default=True, type=bool)
parser.add_argument('--transfer', type=bool)
parser.add_argument('--long_term_node', default=None, type=int)
parser.add_argument('--long_term_his', default=None, type=int)
parser.add_argument('--missing_rate', default=0.0, type=float)
parser.add_argument("--stars", type=str, default='*************************/{}/*************************')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class starting:
    def __init__(self, args, config_args):
        self.mean = None
        self.std = None
        self.args = args
        self.config_args = config_args
        self.data_args, self.task_args, self.model_args = config_args['data'], config_args['task'], config_args['model']
        self.data_loader = data_loader(args, config_args)
        self.model_learner = model_learning(self.config_args, args)


    def print_model(self, net):
        log_string(log, 'model structure')
        param_num = 0
        for name, params in net.named_parameters():
            param_num = param_num + params.nelement()
            log_string(log, str(name) + '--' + str(params.shape), use_info=False)
        log_string(log, 'total num of parameters: {}'.format(param_num))

    def print_FLOPs(self, net, stage, loader):
        net.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(loader):
                data, label, ti, di = self.model_learner.get_in_out(data, label)
                break
            FLOPs, params = profile(net, inputs=(stage, data, ti, di, self.task_args['source_epoch'], self.task_args['pred_num']))  # macs
        log_string(log, 'total num of FLOPs: {:.4f}G'.format(FLOPs / 1000000000))

    def print_predction(self, info):
        horizon = 1
        log_string(log, args.stars.format('prediction result'), use_info=False)
        for i in info:

            if horizon <= self.config_args['task']['pred_num']:
                log_string(log, '@t' + str(horizon) + ' {:.3f} {:.3f} {:.3f}'.format(*i), use_info=False)
            else:
                log_string(log, 'avg {:.3f} {:.3f} {:.3f}\n'.format(*i), use_info=False)
            horizon = horizon + 1

    def start_load_data(self, stage, log):
        # load source data
        if stage == 'source':
            log_string(log, args.stars.format('loading source data: {}...'.format(args.source_dataset)), use_info=False)
            self.source_loaders, self.source_samples, self.source_centers, self.mean_std = self.data_loader.starting(log, args.source_dataset, stage)
        else:
            log_string(log, args.stars.format('loading target data: {}...'.format(args.target_dataset)), use_info=False)
            self.target_loaders, self.target_samples, target_centers = self.data_loader.starting(log, args.target_dataset, stage='target', target_day=3)
            _, _, self.s2t_id = CMD(target_centers[0], self.source_centers[0])
            self.target_memory_bank = self.target_loaders[0]

    def start_construct_model(self):
        # initial model

        if self.args.model == 'CGSTT':
            from model.make_CGSTT import start_make_model
        elif self.args.model == 'DSTSFN':
            from model.make_DSTSFN import start_make_model
        elif self.args.model == 'VLTDL':
            from model.make_VLTDL import start_make_model
        elif self.args.model == 'RDNet':
            from model.make_RDNet import start_make_model
        elif self.args.model == 'D2STGNN':
            from model.make_D2STGNN import start_make_model
        elif self.args.model == 'GRU':
            from model.make_GRU import start_make_model

        else:
            raise SyntaxError('model does not exist.')

        net = start_make_model(self.config_args, args, self.mean_std).cuda()
        return net

    def get_source_model(self, net, param_file, log):
        #starting_main.print_FLOPs(net, stage, self.source_loaders[0])
        if os.path.exists(param_file + 'epoch={}'.format(self.task_args['source_epoch'])):
            log_string(log, 'loading best source model...')
        else:
            log_string(log, 'starting training source model...')
            self.model_learner.source_learning(net, param_file, self.source_loaders, self.source_samples, log, self.mean_std, self.source_centers)
        net.load_state_dict(torch.load(param_file + 'best_model'))
        return net

    def get_transition_model(self, net, param_file, log):
        if os.path.exists(param_file + 'epoch={}'.format(self.task_args['transition_epoch'])):
            log_string(log, 'loading transition model...')
            net.load_state_dict(torch.load(param_file + 'epoch={}'.format(self.task_args['transition_epoch'])))
        else:
            log_string(log, 'starting training transition model...')
            net = self.model_learner.transition_learning(net, param_file, self.source_loaders, self.target_memory_bank, self.source_samples, log)
        return net

    def get_finetuning_model(self, net, param_file, log):
        if os.path.exists(param_file + 'epoch={}'.format(self.task_args['finetuning_epoch'])):
            log_string(log, 'loading finetuning model...')
            net.load_state_dict(torch.load(param_file + 'epoch={}'.format(self.task_args['finetuning_epoch'])))
        else:
            log_string(log, 'starting finetuning model...')
            net = self.model_learner.finetuning_learning(net, param_file, self.target_loaders, self.target_samples, log, self.s2t_id)
        return net

    def testing_source_model(self, net):
        for k in range(0, 11):
            k = k / 10.0
            setup_seed(args.seed)
            log_string(log, 'starting source testing...')
            log_string(log, 'k={}'.format(k))
            prediction_info = self.model_learner.start_test_stage(net, self.source_loaders[2], stage, self.source_centers, k, mean_std=self.mean_std)
            self.print_predction(prediction_info)
        log.close()

    def testing_finetuning_model(self, net):
        log_string(log, 'starting finetuning test...')
        prediction_info = self.model_learner.start_test_stage(net, self.target_loaders[2], stage, s2t_id=self.s2t_id)
        self.print_predction(prediction_info)
        log.close()


if __name__ == '__main__':

    setup_seed(args.seed)
    with open(args.config_filename) as f:
        config_args = yaml.load(f, Loader=yaml.FullLoader)
    data_args = config_args['data']

    starting_main = starting(args, config_args)
    stage = 'source'
    log, param_file = start_generating_path(args, data_args, args.task, stage)
    starting_main.start_load_data(stage, log)
    net = starting_main.start_construct_model()
    starting_main.print_model(net)
    net = starting_main.get_source_model(net, param_file, log)

    starting_main.testing_source_model(net)
    #torch.set_printoptions(threshold=float('inf'))
    #print(net.state_dict()['st_models.3.t_module.adjs.weight'][-1], flush=True)


    ''' **********transfer learning********** '''
    if args.transfer:
        stage = 'transition'
        log, param_file = start_generating_path(args, data_args, args.task, stage)
        starting_main.start_load_data(stage, log)
        net = starting_main.get_transition_model(net, param_file, log)

        stage = 'finetuning'
        log, param_file = start_generating_path(args, data_args, args.task, stage)
        net = starting_main.get_finetuning_model(net, param_file, log)
        starting_main.testing_finetuning_model(net)