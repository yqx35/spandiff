import argparse

from args import train_argparser, eval_argparser
from config_reader import process_configs
from spandiff import input_reader
# from diffusionner.diffusionner_trainer import DiffusionNERTrainer
from spandiff.spandiff_trainer import SpanDiffTrainer
import warnings

warnings.filterwarnings("ignore")

## os设置环境变量
import os
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  ## enable the deterministic method
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
def __train(run_args):
    trainer = SpanDiffTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):

    print('test data',run_args.dataset_path)
    trainer = SpanDiffTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python diffusionner.py train ...'")
    # arg_parser = argparse.ArgumentParser(add_help=False)
    # arg_parser.add_argument('--mode', type=str, default='train', help="Mode: 'train' or 'eval'")
    # args, _ = arg_parser.parse_known_args()
    # # _train()
    # # print(args.mode)
    #
    # if args.mode == 'train':
    #     _train()
    # elif args.mode == 'eval':
    #     # print('---test---')
    #     _eval()
    # else:
    #     raise Exception("Mode not in ['train', 'eval'], e.g. 'python diffusionner.py train ...'")
