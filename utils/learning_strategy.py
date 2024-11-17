from utils.logs import log_string

class learning_strategy:
    def __init__(self, args, num_of_layers, pred_num):
        self.args = args
        self.num_of_layers = num_of_layers
        self.pred_num = pred_num

    def step_increase_4_tem_adj(self, net, ep, tdx, log, param_list):
        for t in param_list:
            output_length = net.state_dict()[t].shape[-1]
            for i in range(output_length // self.pred_num):
                net.state_dict()[t][:, :, self.pred_num * i + tdx - 1: self.pred_num * i + tdx] = \
                    net.state_dict()[t][:, :, self.pred_num * i + tdx - 2: self.pred_num * i + tdx - 1]
        return net, tdx

    def sta_2_dyn_4_ada_adj(self, net, param_list1):
        for t in param_list1:
            net.state_dict()[t][:] = net.state_dict()[t][-1]
        return net

    def sta_2_dyn_4_learnable_adj(self, net, param_list):
        for t in param_list:
            net.state_dict()[t][:] = net.state_dict()[t][-1]
        return net

    def select_learning_strategy(self, net, type, log, tdx=None, ep=None):

        '''
        if type == 'step_increase':
            log_string(log, '\n', use_info=False)
            log_string(log, 'step increase from {} to {}...'.format(tdx, tdx + 1))
            tdx = tdx + 1
            if ep == 0:
                return net, tdx

            if self.args.model == 'DSTSFN' or 'NGSTT':
                param_list = []
                for layer in range(self.num_of_layers):
                    param_list.append('st_models.' + str(layer) + '.t_module.adjs.weight')
                net, tdx = self.step_increase_4_tem_adj(net, ep, tdx, log, param_list)

                return net, tdx

            if self.args.model == 'CGSTT':
                param_list = []
                for layer in range(self.num_of_layers):
                    param_list.append('st_models.' + str(layer) + '.dtgnn.adjs.weight')
                    net, tdx = self.step_increase_4_tem_adj(net, ep, tdx, log, param_list)
                    return net, tdx
            if self.args.model == 'VLTDL':
                pass
        '''

        if type == 'sta2dyn':
            log_string(log, 'switch from static to dynamic...')
            if self.args.model == 'DSTSFN':
                param_list1 = []
                param_list2 = []
                for layer in range(self.num_of_layers):
                    param_list1.append('st_models.' + str(layer) + '.dsgnn.adjs.emb1.weight')
                    param_list2.append('st_models.' + str(layer) + '.dtgnn.adjs.weight')
                net = self.sta_2_dyn_4_ada_adj(net, param_list1)
                net = self.sta_2_dyn_4_learnable_adj(net, param_list2)


            if self.args.model == 'VLTDL':
                param_list1 = []
                param_list2 = []
                for layer in range(self.num_of_layers):
                    param_list1.append('st_models.' + str(layer) + '.s_module.adjs.emb1.weight')
                    param_list2.append('st_models.' + str(layer) + '.t_module.adjs.emb1.weight')
                net = self.sta_2_dyn_4_ada_adj(net, param_list1)
                net = self.sta_2_dyn_4_learnable_adj(net, param_list2)

            if self.args.model == 'CGSTT':
                param_list1 = []
                param_list2 = []

                for layer in range(self.num_of_layers):
                    param_list1.append('st_models.' + str(layer) + '.dtgnn.adjs.weight')
                    param_list2.append('st_models.' + str(layer) + '.dsgnn.adjs.weights.weight')
                net = self.sta_2_dyn_4_ada_adj(net, param_list1)
                net = self.sta_2_dyn_4_learnable_adj(net, param_list2)
            if self.args.model == 'NGSTT':
                param_list1 = []
                param_list2 = []
                param_list3 = []
                for layer in range(self.num_of_layers):
                    param_list1.append('decouple_modules.' + str(layer) + '.variable_block.dsgnn.adjs.emb1.weight')
                    param_list2.append('decouple_modules.' + str(layer) + '.variable_block.dtgnn1.adjs.weight')
                    param_list3.append('decouple_modules.' + str(layer) + '.variable_block.dtgnn2.adjs.weight')

                net = self.sta_2_dyn_4_ada_adj(net, param_list1)
                net = self.sta_2_dyn_4_learnable_adj(net, param_list2)
                net = self.sta_2_dyn_4_learnable_adj(net, param_list3)
            else:
                pass
            return net





