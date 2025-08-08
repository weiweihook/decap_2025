import os
import random
import time
import logging
from typing import List, Tuple

import numpy as np
import torch

from arguments import get_args
from storage import RolloutStorage
import model1 as model
from env import DecapPlaceParallel
from ppo import PPO

torch.set_num_threads(1)

if __name__ == '__main__':
    args = get_args()
    now_time = time.strftime("%Y%m%d-%H%M", time.localtime(time.time()))
    t1 = ''.join([x for x in now_time if x.isdigit()])
    path = 'runs/case%s/' % (args.case_idx) + str(t1) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # args.learning_rate = 1e-4
    num_updates = 600

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # logging
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(path + 'log.txt')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # load env
    vec_env = DecapPlaceParallel([args.case_idx]*args.num_envs)
    actor_critic = model.PPONetwork(vec_env).to(device)
    # actor_critic.load_state_dict(torch.load('runs/case1/202409101544/vec_agent_params600.pth'))
    agent = PPO(actor_critic, args)
    logging.info("===========================================")
    logging.info("Environment: {}, Parallel Number: {}" .format(args.case_idx, args.num_envs))
    logging.info(f'Using GPU: {device}, {args.GPU}')
    logging.info("================ Training =================")


    # ALGO Logic: Storage setup
    rollouts = RolloutStorage(args.num_steps,
                              args.num_envs,
                              vec_env.SINGLE_OBSERVATION_SPACE_SHAPE,
                              vec_env.ACTION_SPACE_SHAPE,
                              vec_env.ACTION_SPACE)
    rollouts.to(device)

    loss = np.zeros(num_updates)
    pg_loss = np.zeros(num_updates)
    entropy_loss = np.zeros(num_updates)
    v_loss = np.zeros(num_updates)
    rewards = np.zeros((num_updates, args.num_steps * args.num_envs))
    BEST = [-50, 0, 0]  # reward, updates, steps
    BEST_Allocation = np.zeros([])

    start_time = time.time()
    for update in range(1, num_updates + 1):
        vec_obs, vec_imped = vec_env.reset()
        next_obs, next_imped = torch.Tensor(vec_obs).to(device), torch.Tensor(vec_imped).to(device)
        vec_action_mask = torch.Tensor(np.stack(vec_env.vec_action_mask())).to(device)
        rollouts.obs[0].copy_(next_obs)
        rollouts.imped[0].copy_(next_imped)
        rollouts.action_masks[0].copy_(vec_action_mask)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            agent.optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):

            #############################################################
            ###### Obs + Action Masks => Agent => Actions################
            ###### Actions => Env => Rewards, log_probs, New Action Maks#

            with torch.no_grad():
                vec_action, vec_logprob, _, vec_value = actor_critic.get_action_and_value(next_obs, next_imped, vec_action_mask)

            # Cal rewards for given actions
            vec_obs, vec_imped, vec_reward, vec_done, info = vec_env.step(vec_action.cpu().numpy())

            #############################################################
            #############################################################

            next_done = torch.Tensor(vec_done).to(device)
            next_obs = torch.Tensor(vec_obs).to(device)
            next_imped = torch.Tensor(vec_imped).to(device)

            if max(info["reward_now"]) > BEST[0]:
                BEST = max(info["reward_now"]), update, step
                index = max(enumerate(info["reward_now"]), key=lambda x: x[1])[0]
                BEST_Allocation = vec_env.vec_cur_params_idx[index]

            # When the target impedance is satisfied or there is no valid positions, Done is True
            # The final reward is propagated to the before states.
            for idx in range(vec_env.env_count):
                if info["reward_now"][idx] > 0 or sum(vec_action_mask[idx]) == 0:
                    next_done[idx] = True
                    obs_done, imped_done = vec_env.reset_idx(idx)
                    next_obs[idx], next_imped[idx] = torch.Tensor(obs_done), torch.Tensor(imped_done)
                    # vec_reward[idx] = info["reward_now"][idx]
                    # indices = (rollouts.dones[:step-1][idx]==1).nonzero(as_tuple=True)[0]
                    # last_step = indices[-1].item() if indices.numel() > 0 else step-1
                    # rollouts.rewards[:last_step, :] += info["reward_now"][idx]

            vec_action_mask = torch.Tensor(np.stack(vec_env.vec_action_mask())).to(device)

            rollouts.insert(step, next_obs, next_imped, vec_action.reshape([-1, vec_env.ACTION_SPACE_SHAPE[0]]),
                            vec_logprob, torch.tensor(vec_reward).view(-1),next_done, vec_value.flatten(),
                            vec_action_mask)

        with torch.no_grad():
            next_value = actor_critic.get_value(next_obs, next_imped).reshape(1, -1)
            rollouts.advantages, rollouts.returns = rollouts.compute_returns(args.num_steps, args.gae, next_value, next_done,
                                                             args.gamma, args.gae_lambda,
                                                             rollouts.values, rollouts.rewards, rollouts.dones)

        v_loss[update - 1], pg_loss[update - 1], entropy_loss[update - 1], loss[update - 1] = agent.update(rollouts,
                                                                                                           args.num_steps,
                                                                                                           vec_env.SINGLE_OBSERVATION_SPACE_SHAPE,
                                                                                                           vec_env.ACTION_SPACE_SHAPE)
        rewards[update - 1] = rollouts.rewards.cpu().numpy().reshape(-1)

        logging.info(f"-------------------- Update {update} --------------------")
        logging.info('Best Update :{}, Best Step :{}, Best Reward: {}'.format( BEST[1], BEST[2], BEST[0]))
        np.savetxt(path + 'allocation.txt', BEST_Allocation)

    end_time = time.time()
    logging.info('time cost: {} s'.format(end_time - start_time))

    # save network parameters
    torch.save(actor_critic, path + 'vec_agent.pth')
    torch.save(actor_critic.state_dict(), path + 'vec_agent_params.pth')

    # save data
    np.savetxt(path + 'reward.txt', rewards)
    np.savetxt(path + 'loss.txt', loss)
    np.savetxt(path + 'pgloss.txt', pg_loss)
    np.savetxt(path + 'entloss.txt', entropy_loss)
    np.savetxt(path + 'vloss.txt', v_loss)



