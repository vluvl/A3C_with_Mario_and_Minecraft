import torch
#import torchvision
#import torchvision.transforms.functional as transforms
#import matplotlib.pyplot as plt
from .env import create_train_env_mine
from .model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import logging
from matplotlib import animation
import matplotlib.pyplot as plt

def local_train(index, opt, global_model, optimizer, reset, save=False):
    torch.manual_seed(123 + index)
    
    if save:
        start_time = timeit.default_timer()
        logging.error("Process %s Episode %s has started", str(index), str(reset * int(opt.num_global_steps / opt.num_local_steps)))
    writer = SummaryWriter(opt.log_path+str(index))
    env, num_states, num_actions = create_train_env_mine(opt.envName)
    local_model = ActorCritic(num_states, num_actions)
    local_model.load_state_dict(global_model.state_dict())
    
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    state = env.reset()

    state = torch.from_numpy(state.astype("float32"))

    if opt.use_gpu:
        state = state.cuda()
        
    done = True
    curr_step = 0
    curr_episode = -1 + reset * int(opt.num_global_steps / opt.num_local_steps)
    
    while True:
        curr_episode += 1
        ep_start = timeit.default_timer()
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save({
                    'model_state_dict': global_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), #add loss
                }, "{}/MineRLParams_check{}".format(opt.saved_path, reset))
            if curr_episode % 1000 == 0:
                logging.error("Process %s. Episode %s", str(index), str(curr_episode))

        
        local_model.load_state_dict(global_model.state_dict())
        
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []
        frames = []
        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)  #from 1 to -1
            entropy = -(policy * log_policy).sum(-1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()
            try:
                state, reward, done, _ = env.step(action)
                frames.append(env.render(mode="rgb_array"))
            except:
                print("\n//////////////////////////////////////////////////////\n")
                print("Something went wrong when taking a step on process " + str(
                    index) + ". Attempting to restart MineRL\n")
                env.close()
                env, _, _ = create_train_env_mine(opt.envName)
                state = env.reset()
                done = 0
                curr_step = 0
                #state = torch.from_numpy(state.astype("float32"))

                if opt.use_gpu:
                    state = state.cuda()


            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True

            if done:

                curr_step = 0
                try:
                    state = torch.from_numpy(env.reset())
                except:
                    print("\n//////////////////////////////////////////////////////\n")
                    print("Something went wrong when resetting on done on process " + str(
                        index) + ". Attempting to restart MineRL\n")
                    env.close()
                    env, _, _ = create_train_env_mine(opt.envName)
                    state = env.reset()
                    done = 0
                    state = torch.from_numpy(state.astype("float32"))

                if opt.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break
        save_frames_as_gif(frames,filename=opt.checkpoint + 'mine' + str(curr_episode) + '.gif')
        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        totalReward = 0

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            # delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            # gae = gae * args.gamma * args.tau + delta_t

            next_value = value

            actor_loss = actor_loss - log_policy * gae  # - entropy * opt.beta
            # policy_loss = policy_loss - log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]
            R = R * opt.gamma + reward
            # R = args.gamma * R + rewards[i]
            critic_loss = critic_loss + (R - value) ** 2 / 2
            # advantage = R - values[i]
            # value_loss = value_loss + 0.5 * advantage.pow(2)

            entropy_loss = entropy_loss + entropy
            totalReward = totalReward + reward

            # old calculations \/
            # gae = gae * opt.gamma * opt.tau
            # gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            # next_value = value
            # actor_loss = actor_loss + log_policy * gae
            # R = R * opt.gamma + reward
            # critic_loss = critic_loss + (R - value) ** 2 / 2
            # entropy_loss = entropy_loss + entropy
            # totalReward = totalReward + reward

        total_loss = actor_loss + critic_loss  # - opt.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        writer.add_scalar("Train_{}/Reward".format(index), totalReward, curr_episode)
        writer.add_scalar("Train_{}/Actor_Loss".format(index), actor_loss, curr_episode)
        writer.add_scalar("Train_{}/Critic_Loss".format(index), critic_loss, curr_episode)
        writer.add_scalar("Train_{}/Resets".format(index), reset, curr_episode)
        writer.flush()
        # optimizer.zero_grad()
        # total_loss.backward()

        # freeze all layers
        for param in local_model.parameters():
            param.requires_grad = False

        # activate actor layer and backprop
        local_model.actor_linear.requires_grad = True
        optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        # freeze actor and enable critic layer and backprop
        local_model.actor_linear.requires_grad = False
        local_model.critic_linear.requires_grad = True
        optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)

        # freeze actor and critic and enable all other parameters and backprop total loss
        for param in local_model.parameters():
            param.requires_grad = True
        local_model.actor_linear.requires_grad = False
        local_model.critic_linear.requires_grad = False
        optimizer.zero_grad()
        total_loss.backward()

        # reactivate all
        for param in local_model.parameters():
            param.requires_grad = True

        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 250)
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()
        try:
            if curr_episode % 25 == 0:
                env.reset()
        except:
            print("\n//////////////////////////////////////////////////////\n")
            print("Something went wrong when resetting the env on process " + str(index) + ". Attempting to restart MineRL\n")
            env.close()
            env, _, _ = create_train_env_mine(opt.envName)
            state = env.reset()
            done = 0
            state = torch.from_numpy(state.astype("float32"))
            if opt.use_gpu:
                    state = state.cuda()
        ep_end = timeit.default_timer()
        
        writer.add_scalar("Train_{}/EP_time".format(index), ep_end - ep_start, curr_episode)
        writer.flush()
        if curr_episode == int(opt.num_global_steps / opt.num_local_steps) * (reset + 1):
            logging.error("Process %s. Finished after %s episodes. It should restart soon...", str(index), str(curr_episode))
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                torch.save({
                    'model_state_dict': global_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), #add loss
                }, "{}/MineRLParams_check{}".format(opt.saved_path, reset))
                print('The code runs for %.2f s ' % (end_time - start_time))
            return

# Source: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1]/10, frames[0].shape[0]/10), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=12)


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env_mine(opt.envName)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    file_ = "{}/MineRLParams_check15".format(opt.saved_path)
    local_model.load_state_dict(torch.load(file_, weights_only=True)['model_state_dict'])

    state = env.reset()

    state = torch.from_numpy(state.astype("float32"))
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    #actions = deque(maxlen=opt.max_actions)
    total = 0
    while True:
        curr_step += 1
        #if done:
            #local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            if opt.use_gpu:
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        entropy = -(policy * log_policy).sum(-1, keepdim=True)
        m = Categorical(policy)
        action = m.sample().item()
        state, reward, done, _ = env.step(action)
        total = total + reward
        env.render()

        if curr_step > opt.num_global_steps:
            done = True
        if done:
            #print("TEST reward: " + str(total))
            total = 0
            curr_step = 0
            state = env.reset()
        state = torch.from_numpy(state)
