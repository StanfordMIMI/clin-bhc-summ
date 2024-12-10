import argparse
import os
import constants

def get_parser():
    ''' parse arguments '''

    parser = argparse.ArgumentParser()

    ### args agnostic to discrete or soft prompting
    parser.add_argument(
        "--model",
        help="model name"
        #required=True,
    )
    parser.add_argument(
        "--case_id",
        type=int,
        help="case id number (integer) per constants.py"
        #required=True,
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='id of gpu to use'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default="0",
        help='id of gpu to use'
    )
    
    args = parser.parse_args()
    args = set_args(args)
    args.device=f'cuda:{args.gpu_id}'

    return args


def set_args(args):
    ''' set args based on parser, constants.py 
        written separately to be modular w generate_table.py '''
    
    # define directories based on expmt params
    args.expmt_name = f'{args.model}_case{args.case_id}'
    args = set_args_dir_out(args)
    args = set_args_dir_model(args)

    args.max_new_tokens = constants.cases[args.case_id]['max_new_tokens']
    if args.case_id >= 100:
        args.batch_size = constants.cases[args.case_id]['batch_size']
        args.trn_epochs = constants.cases[args.case_id]['trn_epochs']
        args.grad_accum_steps = constants.cases[args.case_id]['grad_accum_steps']
        args.lr_n_warmup_steps = constants.cases[args.case_id]['lr_n_warmup_steps']

    return args


def set_args_dir_out(args):
    ''' create directory for output data 
        separate dir from output data for ood cases '''

    args.dir_out = os.path.join(
        constants.DIR_PROJECT,
        'output',
        args.expmt_name + '/'
    )

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    return args


def set_args_dir_model(args):
    ''' create directory for tuned models 
        separate dir from output data for ood cases '''

    args.dir_models_tuned = os.path.join(
        constants.DIR_PROJECT,
        'models_tuned',
        args.expmt_name + '/'
    )
   
    if not os.path.exists(args.dir_models_tuned):
        os.makedirs(args.dir_models_tuned)

    return args