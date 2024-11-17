from utils.logs import log_string
import os

def start_generating_path(args, data_args, task, stage):
    # config list
    config_list = task + '/' + args.model + '/' + args.remark1 + \
        '/alpha1={}--alpha2={}--k={}--num_of_layer={}--d_model={}--seed={}--num_of_latent={}/' \
        .format(args.alpha1, args.alpha2, args.k, args.num_of_layer, args.d_model, args.seed, args.num_of_latent)

    if task == 'CRGTF' and stage == 'transition':
        config_list = config_list + args.remark2 + '/transition/'
    elif task == 'CRGTF' and stage == 'finetuning':
        config_list = config_list + args.remark2 + '/finetuning/'
    else:
        pass
    # if task == 'GTFMV':
    #     config_list = config_list + 'missing_rate={}'.format(args.missing_rate)

    # generating log path
    log_dir = data_args[args.source_dataset]['log_path'] + config_list
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # generating param path
    param_dir = data_args[args.source_dataset]['param_path'] + config_list
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    # merge final path
    if task == 'GTFMV':
        log_dir = log_dir + 'missing_rate={}'.format(args.missing_rate) + '_'
    id = 0
    log_file = log_dir + str(id)
    while os.path.exists(log_file + '.txt'):
        id = id + 1
        log_file = log_dir + str(id)
    log = open(log_file + '.txt', 'w')


    log_string(log, 'log_file: {}'.format(log_file))
    log_string(log, 'param_file: {}\n'.format(param_dir))
    return log, param_dir