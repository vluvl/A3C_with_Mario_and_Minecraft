import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env_mine
from src.model import ActorCritic
from src.optimizer import GlobalAdam
from src.process import local_train, local_test
import torch.multiprocessing as _mp
import shutil
import logging


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex") # was complex
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=100)
    parser.add_argument("--num_global_steps", type=int, default=100 * 25 * 20) # was 5e6, 100 * 25 * 15
    parser.add_argument("--num_process_restarts", type=int, default=60)
    parser.add_argument("--num_processes", type=int, default=1) # was 6
    parser.add_argument("--save_interval", type=int, default=200, help="Number of episodes between savings")
    parser.add_argument("--max_actions", type=int, default=1000, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/MineRLFINAL")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=True,
                        help="Load weight from previous trained stage")
    parser.add_argument("--checkpoint", type=str, default="15")
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--envName", type=str, default="MineRLObtainDiamondShovel-v0")
    args = parser.parse_args()
    return args


def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    env, num_states, num_actions = create_train_env_mine(opt.envName)
    global_model = ActorCritic(num_states, num_actions)
    if opt.use_gpu:
        global_model.cuda()
    global_model.share_memory()
    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    if True: #opt.load_from_previous_stage:
        file_ = "{}/MineRLParams_check{}".format(opt.saved_path, opt.checkpoint)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_, weights_only=True)['model_state_dict'])
            optimizer.load_state_dict(torch.load(file_, weights_only=True)['optimizer_state_dict'])


    processes = []
    resets = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, 0))
        resets.append(0)
        process.start()
        processes.append(process)
    # process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    # process.start()
    # processes.append(process)
    # resets = opt.num_process_restarts
    while resets[0] < opt.num_process_restarts:
        for index, process in enumerate(processes):
            if not process.is_alive():
                resets[index] += 1
                logging.error("//////////////////Restarting process %s ... //////////////////", str(index))
                if index == 0:
                    process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, resets[index], True))
                else:
                    process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, resets[index]))
                process.start()
                processes[index] = process

    for process in processes:
        process.join()





if __name__ == "__main__":
    opt = get_args()
    train(opt)
