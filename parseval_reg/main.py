###
# 
#
#
###


import torch 
import numpy as np

import argparse
import time
import sys, os

class ConfigDictConverter:
    def __init__(self, config_dict):
        '''
        This class takes a config_dict which contains all the variables needed to do one run
        and converts it into the variables needed to run the RL experiments
        We assume that the config file has certain variables and is organized in the proper way
        For the env and agent parameters, we will pass config_dict on to them and assume that they handle it properly
        Note that we *cannot* use the same variable names for env and agent parameters. If the agent or env expect the
        same name, we will need to write it down different in the config file and then convert here

        Attributes:
        agent_dict:
        env_dict:
        '''
        # Improvement: possible to split agent and env parameters here. That is this class contains
        # two dicts for agent_parameters and env_parameters.
        # This would help with passing only the required arguments for envs that are already created (e.g gym)
        # Also, it could help with dealing with parameters that have the same name but are different for agent and env

        self.config_dict = config_dict.copy()

        # training shouldn't need these variables
        if 'num_repeats' in self.config_dict.keys():
            del self.config_dict['num_repeats']
        if 'num_runs_per_group' in self.config_dict.keys():
            del self.config_dict['num_runs_per_group']

        self.agent_dict = self.config_dict.copy()
        self.env_dict = self.config_dict.copy()

        # TODO remove maybe?
        self.repeat_idx = config_dict['repeat_idx']  

        # algorithm
        # for RL
        if config_dict['base_algorithm'].lower() in ('ppo_agent',):
            agent = config_dict['base_algorithm'].lower()

            if agent == 'ppo_agent':
                from agent import PPO_Agent

                self.agent_class = PPO_Agent
                agent_key_lst = ['base_algorithm', 'device',
                                 'learning_rate', 'num_envs', 'rollout_num_steps', 'anneal_lr', 'gamma',
                                 'gae_lambda', 'num_minibatches', 'minibatch_size', 'update_epochs', 'norm_adv',
                                 'clip_coef', 'clip_vloss', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl',
                                 'weight_decay', 'adam_eps', 'adam_beta2', 'tuned_adam',
                                 'layer_norm', 'layer_norm_no_params',
                                 'parseval_reg', 'parseval_num_groups',
                                 'tsallis_entropy', 'l2_init', 'group_sort', 'weight_init', 'init_gain',
                                 'perturb', 'perturb_dist',
                                 'net_width', 'net_activation']
                
            elif agent == 'ppo_agent_metaworld':
                from minimal.agent import PPO_Agent
                self.agent_class = PPO_Agent
                agent_key_lst = ['base_algorithm', 'device', 'seed',
                                 'learning_rate', 'num_envs', 'rollout_num_steps', 'gamma',
                                 'gae_lambda', 'num_minibatches', 'minibatch_size', 'update_epochs', 'norm_adv',
                                 'clip_coef', 'clip_vloss', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl',
                                 'weight_decay', 'adam_eps', 'adam_beta2', 'tuned_adam',
                                 'layer_norm','layer_norm_no_params',
                                 'network_type', 'add_diag_layer',
                                 'parseval_reg', 'parseval_norm', 'parseval_num_groups',
                                 'l2_init',
                                 "parseval_last_layer", "rpo_alpha", 'weight_init', 'init_gain',
                                 'perturb', 'perturb_dist', 'perturb_rotate', 'regen', 'regen_wasserstein',
                                 'net_width', 'net_activation',
                                 'input_scale', 'learnable_input_scale'
                                 ] 
        else:
            raise AssertionError('Invalid algorithm', config_dict['base_algorithm'])

        self.agent_dict = {k: v for k, v in self.agent_dict.items() if k in agent_key_lst}
        print('agent_dict keys', self.agent_dict.keys())

        # environment
        # reinforcement learning
        if 'gridworld' in config_dict['env'].lower():
            env = config_dict['env'].lower()
            env_key_lst = ['env', 'seed', 'change_freq', 'reward_type', 'include_goal_obs',
                           'image_obs', 'time_limit', 'flatten_obs', 'non_sparse_obs', 'hard_exploration',
                           'two_hot_state', 'capture_video', 'save_name']
            import envs.gridworld_env

            if env == 'gridworld_oneroom':
                self.env_class = envs.gridworld_env.GridWorldEnv 
            elif env == 'gridworld_ninerooms':
                self.env_class = envs.gridworld_env.NineRoomsEnvWrapper

            self.env_dict = {k: v for k, v in self.env_dict.items() if k in env_key_lst}
            self.env_dict['env_type'] = 'rl'

        # elif 'gym' in config_dict['env'].lower():
        #     # we only need to give env_id since state and action spaces are inferred from the environment after
        #     # when the env is passed to the agent at init
        #     self.env_class = EnvsRL.GymPPOContinuousControl

        #     env_key_lst = ['env', 'env_id', 'seed', 'capture_video', 'save_name']
        #     env = config_dict['env'].lower()
        #     if env == 'gym_hopper':
        #         self.env_dict['env_id'] = "Hopper-v4"
        #     elif env == 'gym_ant':
        #         self.env_dict['env_id'] = "Ant-v4"
        #     elif env == 'gym_halfcheetah':
        #         self.env_dict['env_id'] = "HalfCheetah-v4"
        #     elif env == 'gym_humanoidstandup':
        #         self.env_dict['env_id'] = "HumanoidStandup-v4"
        #     elif env == 'gym_humanoid':
        #         self.env_dict['env_id'] = "Humanoid-v4"
        #     elif env == 'gym_inverteddoublependulum':
        #         self.env_dict['env_id'] = "InvertedDoublePendulum-v4"
        #     elif env == 'gym_invertedpendulum':
        #         self.env_dict['env_id'] = "InvertedPendulum-v4"
        #     elif env == 'gym_reacher':
        #         self.env_dict['env_id'] = "Reacher-v4"
        #     elif env == 'gym_pusher':
        #         self.env_dict['env_id'] = "Pusher-v4"
        #     elif env == 'gym_swimmer':
        #         self.env_dict['env_id'] = "Swimmer-v4"
        #     elif env == 'gym_walker2d':
        #         self.env_dict['env_id'] = "Walker2d-v4"

        #     self.env_dict = {k: v for k, v in self.env_dict.items() if k in env_key_lst}
        #     self.env_dict['env_type'] = 'rl'


        elif 'carl_' in config_dict['env'].lower():
            import envs.carl_env
            # we expect the 'env' to be e.g. "carl_sequence_dmcwalker_3" (for sequences of tasks)
            #  or "carl_dmcwalker" (for single task)
            # if config_dict['env'].lower()[:13] == 'carl_sequence_': 
            if "sequence" in config_dict['env'].lower():
                self.env_class = envs.carl_env.CARLSequence
                base_task_name = config_dict['env'].lower()[14:]  # e.g. could be dmcwalker_3
            else:
                self.env_class = envs.carl_env.CARLSingleEnv
                base_task_name = config_dict['env'].lower()[5:]

            env_key_lst = ['env', 'base_task_name', 'seed', 'goal_hidden', 'normalize_obs', 'normalize_rewards',
                           'capture_video', 'save_name', 'change_freq', 'env_sequence',
                           'obs_drift_mean', 'obs_drift_std', 'obs_scale_drift', 'obs_noise_std', 'normalize_avg_coef',
                           'reset_obs_stats', 'change_when_solved']

            self.env_dict['base_task_name'] = base_task_name  # e.g. "carl_sequence_walker" or "carl_acrobot"

            temp_dict = {k: v for k, v in self.env_dict.items() if k in env_key_lst}
            temp_dict.update({k:v for k,v in self.env_dict.items() if k[:5] == "carl_"})  # also add context vars like "carl_gravity"
            self.env_dict = temp_dict
            self.env_dict['env_type'] = 'rl'

        elif 'metaworld' in config_dict['env'].lower():
            import envs.metaworld_env
            self.env_class = envs.metaworld_env.MetaWorldSingleEnvSequence

            env_key_lst = ['env', 'base_task_name', 'seed', 'goal_hidden', 'normalize_obs', 'normalize_rewards',
                           'capture_video', 'save_name', 'change_freq', 'env_sequence',
                           'obs_drift_mean', 'obs_drift_std', 'obs_scale_drift', 'obs_noise_std', 'normalize_avg_coef',
                           'reset_obs_stats', 'change_when_solved']
            env = config_dict['env'].lower()

            if env[0:19] == 'metaworld_sequence_':  # e.g. 'metaworld_sequence_reach'
                if env[19:22] == 'set': # e.g. "metaworld_sequence_set1"
                    self.env_dict['env_sequence'] = env[19:]  # e.g. "set1"
                else:
                    self.env_dict['base_task_name'] = f'{env[19:]}-v2'

            self.env_dict = {k: v for k, v in self.env_dict.items() if k in env_key_lst}
            self.env_dict['env_type'] = 'rl'

        else:
            raise AssertionError("config dict converter: env doesn't match" + config_dict['env'])

        print('env_dict keys', self.env_dict.keys())

        # Other params
        self.agent_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Adjust the seed based on repeat
        self.env_dict['seed'] += self.repeat_idx*1


        # local run
        if 'local_run' in self.config_dict and self.config_dict['local_run']:
            self.env_dict['local_run'] = True
        else:
            self.env_dict['local_run'] = False


import pickle
from collections import defaultdict
class RLLogger:
    def __init__(self, save_freq, save_model_freq=None, config_idx=0):
        self.metrics = defaultdict(list)
        self.save_freq = save_freq
        self.save_model_freq = save_model_freq  # how often to save the model checkpoints (less frequent usually)

    def save_metrics(self, option, agent=None, loss=None, episode_return=None, save_path="", save_tag="", *args,
                     **kwargs):
        # save model
        if option == 'standard':
            self.metrics['loss'].append(loss)
            for k, v in kwargs.items():
                self.metrics[k].append(v)
        elif option == 'episode':
            self.metrics['return'].append(episode_return)
            for k, v in kwargs.items():
                self.metrics[k].append(v)
        elif option == 'eval':
            # eval_return
            for k, v in kwargs.items():
                self.metrics[k].append(v)
        elif option == 'model':
            agent.save_model(save_path + f'models/model_{save_tag}_{kwargs.get("total_num_steps")}.pyt')
            print('saved agent')
        elif option == 'states':
            states = kwargs['states']
            import torch
            torch.save(states, save_path + f'models/states_{save_tag}_{kwargs.get("total_num_steps")}.pt')
            print('saved states')

    def reset(self):
        self.metrics = defaultdict(list)

    def save_to_file(self, save_path, save_tag):
        os.makedirs(save_path, exist_ok=True)

        # np.save(save_path + f'data_{save_tag}.npy', self.metrics)
        with open(save_path + f'data_{save_tag}.pkl', 'wb') as file:
            pickle.dump(self.metrics, file, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_summary(self):
        ''' Prints averages of the metrics '''
        ...




def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    parser.add_argument('--test_run', action='store_true', help='Run a test run with only 10k steps and eval/save every 2k steps')

    # General experiment arguments
    parser.add_argument('--algorithm', type=str, default='parseval', help='Learning algorithm to run for the experiment')
    parser.add_argument('--base_algorithm', type=str, default='ppo_agent', help='Base algorithm for the agent')
    parser.add_argument('--repeat_idx', type=int, default=0, help='Index of the repeat (for multiple runs)')
    parser.add_argument('--env', type=str, default='metaworld_sequence_set2', help='Environment to run')
    parser.add_argument('--change_freq', type=int, default=1e6, help='Frequency to change tasks in the environment')  # note this is overriden below per environment
    parser.add_argument('--num_steps', type=int, default=10000, help='Num steps to run' )  # note this is overriden below per-enviornment
    parser.add_argument('--seed', type=int, default=123, help='Num steps to run')
    parser.add_argument('--save_path', type=str, default='results/', help='Path to the folder to be saved in')
    parser.add_argument('--save_freq', type=int, default=25000, help='Number steps between recording metrics')
    parser.add_argument('--save_model_freq', type=int, default=-1, help='Number of steps between saving the model. Set to -1 for never. ')
    

    # Arguments for PPO/RPO
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments')
    parser.add_argument('--rollout_num_steps', type=int, default=2048, help='Number of steps per rollout')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--num_minibatches', type=int, default=32, help='Number of minibatches')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--minibatch_size', type=int, default=64, help='Size of each minibatch')
    parser.add_argument('--norm_adv', type=bool, default=True, help='Normalize advantages')
    parser.add_argument('--clip_coef', type=float, default=0.2, help='Clip coefficient')
    parser.add_argument('--clip_vloss', type=bool, default=True, help='Clip value loss')
    parser.add_argument('--ent_coef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--rpo_alpha', type=float, default=0.5, help='RPO alpha parameter')

    # Parseval regularization
    parser.add_argument('--parseval_reg', type=float, default=0, help='Parseval regularization coefficient')

    # Adding additional learnable parameters
    parser.add_argument('--input_scale', type=float, default=1, help='Input scaling factor')
    parser.add_argument('--learnable_input_scale', type=bool, default=False, help='Make input scale learnable')
    parser.add_argument('--add_diag_layer', type=bool, default=False, help='Add diagonal layers to the network')

    # Other loss of plasticity methods
    parser.add_argument('--layer_norm', type=bool, default=False, help='Use layer normalization')
    parser.add_argument('--layer_norm_no_params', type=bool, default=False, help='Layer norm without additional parameters')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay coefficient')
    parser.add_argument('--tuned_adam', type=bool, default=False, help='Use tuned Adam optimizer')
    # Perturb (for Shrink-and-Perturb, combine with weight decay)
    parser.add_argument('--perturb', type=float, default=0.0, help='Perturbation factor')
    parser.add_argument('--perturb_dist', type=str, default='xavier', help='Distribution for perturbations (e.g., xavier)')
    # Regenerative regularization
    parser.add_argument('--regen', type=float, default=0.0, help='Regeneration factor')
    parser.add_argument('--regen_wasserstein', type=bool, default=False, help='Use Wasserstein loss for regeneration')

    # Network architecture arguments
    parser.add_argument('--weight_init', type=str, default='orthogonal', help='Weight initialization method')
    parser.add_argument('--net_width', type=int, default=64, help='Width of the network layers')
    parser.add_argument('--net_activation', type=str, default='tanh', help='Activation function for the network')
    parser.add_argument('--init_gain', type=float, default=None, help='Gain for weight initialization (if applicable)')

    # Ablations
    parser.add_argument('--parseval_norm', type=bool, default=False, help='Normalize row weight vectors before applying Parseval reg')
    parser.add_argument('--parseval_last_layer', type=bool, default=False, help='Apply Parseval norm to the last layer')
    parser.add_argument('--parseval_num_groups', type=int, default=1, help='Number of groups for Parseval regularization')


    args = parser.parse_args()
    
    # Set different defaults depending on the env and algorithm
    if "gridworld" in args.env: 
        args.change_freq = 40000
        args.num_steps = 800000
        args.save_freq = 5000

        args.learning_rate = 0.00025  
        args.rollout_num_steps = 128  
        args.num_minibatches = 4  
        args.update_epochs = 4 
        args.minibatch_size = 32  
        args.ent_coef = 0.01 
        args.rpo_alpha = 0  

        if args.algorithm == 'base':
            ...
        elif args.algorithm == 'parseval':
            args.parseval_reg = 0.001

        elif args.algorithm == 'snp':  # shrink and perturb
            args.perturb = 0.001
            args.weight_decay = 0.001

        elif args.algorithm == 'layer_norm':  # layer normalization
            args.layer_norm = True
            
        elif args.algorithm == 'regen':  # regenerative regularization (l2-init)
            args.regen = 0.001
            
        elif args.algorithm == 'w-regen':  # wasserstein version
            args.regen = 0.001
            args.regen_wasserstein = True

    elif "lunarlander" in args.env: 
        args.num_steps = 1e7
        args.change_freq = 500000

        args.rollout_num_steps = 128  
        args.num_minibatches = 4  
        args.update_epochs = 4 
        args.minibatch_size = 32  
        args.ent_coef = 0.0001 
        args.rpo_alpha = 0  

        if args.algorithm == 'base':
            ...
        elif args.algorithm == 'parseval':
            args.parseval_reg = 0.0001

        elif args.algorithm == 'snp':  # shrink and perturb
            args.perturb = 0.001
            args.weight_decay = 0.001

        elif args.algorithm == 'layer_norm':  # layer normalization
            args.layer_norm = True
            
        elif args.algorithm == 'regen':  # regenerative regularization (l2-init)
            args.regen = 0.001
            
        elif args.algorithm == 'w-regen':  # wasserstein version
            args.regen = 0.001
            args.regen_wasserstein = True

    elif "quadruped" in args.env: 
        args.num_steps = 12000000
        args.change_freq = 1500000

        args.entropy_coef = 0.0001
        args.learnable_input_scale = True

        if args.algorithm == 'base':
            ...
        elif args.algorithm == 'parseval':
            args.parseval_reg = 0.0001

        elif args.algorithm == 'snp':  # shrink and perturb
            args.perturb = 0.001
            args.weight_decay = 0.001

        elif args.algorithm == 'layer_norm':  # layer normalization
            args.layer_norm = True
            
        elif args.algorithm == 'regen':  # regenerative regularization (l2-init)
            args.regen = 0.01
            
        elif args.algorithm == 'w-regen':  # wasserstein version
            args.regen = 0.001
            args.regen_wasserstein = True


    elif "metaworld" in args.env: 
        args.num_steps = 1e7
        args.change_freq = 1e6

        if args.algorithm == 'base':
            ...
        elif args.algorithm == 'parseval':
            args.parseval_reg = 0.001
            args.add_diag_layer = True

        elif args.algorithm == 'snp':  # shrink and perturb
            args.perturb = 0.001
            args.weight_decay = 0.001

        elif args.algorithm == 'layer_norm':  # layer normalization
            args.layer_norm = True
            
        elif args.algorithm == 'regen':  # regenerative regularization (l2-init)
            args.regen = 0.01
            
        elif args.algorithm == 'w-regen':  # wasserstein version
            args.regen = 0.001
            args.regen_wasserstein = True

    else:
        raise AssertionError("Invalid env", args.env)


    if args.test_run:
        args.num_steps = 10000
        args.save_freq = 2000

    # initialize config
    save_tag = f"{args.env}_{args.algorithm}_{args.repeat_idx}"
    config_obj = ConfigDictConverter(vars(args))
    agent_parameters = config_obj.agent_dict
    env_parameters = config_obj.env_dict
    num_steps_per_run = args.num_steps
    save_path = args.save_path

    print(f"Start RL run!")
    print(agent_parameters)
    print(env_parameters)
    print("Running on gpu:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # TODO check this
    # if env_parameters['local_run']:
    #     print("LOCAL RUN")

    start_time = time.perf_counter()

    # initialize logger, agent, env
    metric_logger = RLLogger(args.save_freq, args.save_model_freq)
    metric_logger.reset()  # reset for run


    env = config_obj.env_class(**env_parameters)

    agent_parameters.update({"env": env})  # use 'env' to pass to agent
    agent_parameters.update({"device": device})
    agent = config_obj.agent_class(**agent_parameters)

    obs, _ = env.reset()  # match the gymnasium interface

    i_step = 0

    if metric_logger.save_model_freq > 0:  # there's an option not to save models
        metric_logger.save_metrics(option='model',
                                        agent=agent,
                                        total_num_steps=i_step,
                                        save_path=save_path)

    while i_step < num_steps_per_run:
        actions = agent.act(obs)
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # TRY NOT TO MODIFY: save data to replay buffer; handle `terminal_observation`
        # also updates parameters
        agent.update(obs, next_obs, actions, rewards, terminateds, truncateds, infos)

        obs = next_obs


        ## This is for the VectorEnv
        if "final_info" in infos:  # "final_info" should only be present if it's a VectorEnv
            for info in infos["final_info"]:
                # Skip the envs that are not done
                # print(infos['final_info'])
                if "episode" not in info:
                    continue

                # print(f"global_step={i_step}, episodic_return={info['episode']['r']}")
                # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                episode_metrics = {}

                metric_logger.save_metrics('episode',
                                                episode_return=info['episode']['r'],
                                                episode_length=info['episode']['l'])
                break

        ## For single (non-vector) envs
        if isinstance(terminateds, bool):  # check that it is a single env
            if "episode" in infos:
                if env_parameters['local_run']:
                    print(f"global_step={i_step}, episodic_return={infos['episode']['r']}")

                temp_online_return = infos['episode']['r']
                metric_logger.save_metrics('episode', episode_return=temp_online_return, step=i_step)

            if terminateds or truncateds:
                obs, _ = env.reset()

        if i_step == 0 or (i_step + 1) % metric_logger.save_freq == 0:
            print('time: ', round((time.perf_counter() - start_time)/60,3), "SPS:", int(i_step / (time.perf_counter() - start_time)))
            # print(f"global_step={i_step}, episodic_return={infos['episode']['r']}")

            save_metrics = {}

            if agent_parameters['base_algorithm'] == 'ppo_agent':
                if agent.explained_var is not None:
                    save_metrics['explained_var'] = agent.explained_var
                    save_metrics['entropy'] = agent.entropy
                    logged_values = agent.get_log_quantities()
                    save_metrics.update(logged_values)
                    metrics = ['loss', 'actor_matrix_stable_rank', 'critic_matrix_stable_rank', 'actor_weight_norms', 'critic_weight_norms']
                    # print({k:v for k,v in save_metrics.items() if k in metrics})
                #
                metric_logger.save_metrics('standard', **save_metrics)


        if i_step == 0 or (i_step + 1) % metric_logger.save_freq == 0:  # eval step
            save_metrics = {}
            # eval
            num_eval_runs = 10
            eval_results = env.evaluate_agent(agent, num_eval_runs)
            eval_episode_returns = eval_results['episodic_returns']
            save_metrics['runtime'] = (time.perf_counter() - start_time) / 60  # time in mins

            save_metrics['mean_eval_return'] = np.mean(eval_episode_returns)
            save_metrics['std_eval_return'] = np.std(eval_episode_returns, ddof=1)
            save_metrics['min_eval_return'] = np.min(eval_episode_returns)
            save_metrics['max_eval_return'] = np.max(eval_episode_returns)
            if 'metaworld' in env_parameters['env'] or 'gridworld' in env_parameters['env']:
                eval_successes = eval_results['successes']
                save_metrics['mean_eval_success'] = np.mean(eval_successes)
                save_metrics['std_eval_success'] = np.std(eval_successes, ddof=1)
                # save_metrics['task_counter'] = env.task_counter

                # save the obs normalization stats
                if hasattr(env, 'obs_mean'):
                    save_metrics['obs_running_mean'] = env.obs_mean
                    save_metrics['obs_running_var'] = env.obs_var

                print(f"{i_step} success {round(save_metrics['mean_eval_success'],3)} +/- {round(save_metrics['std_eval_success']/np.sqrt(num_eval_runs),3)}")

            elif 'dmc' in env_parameters['env']:
                # save_metrics['task_counter'] = env.task_counter
                save_metrics['obs_running_mean'] = env.obs_mean
                save_metrics['obs_running_var'] = env.obs_var

            if 'gridworld' in env_parameters['env']:
                # print(obs, obs.shape)
                save_metrics['goal_location'] = env.goal_states[0][0]

                optimal_length_one_room = eval_results['optimal_length']
                eval_episode_steps = eval_results['episode_steps']
                save_metrics['mean_eval_steps'] = np.mean(eval_episode_steps)
                save_metrics['best_eval_steps'] = np.min(eval_episode_steps)
                save_metrics['optimal_eval_steps_one_room'] = optimal_length_one_room
                print(f"{i_step} eval steps mean {save_metrics['mean_eval_steps']} best {save_metrics['best_eval_steps']} optimal {optimal_length_one_room}")

            print(f"{i_step} eval return {round(save_metrics['mean_eval_return'],3)} +/- {round(save_metrics['std_eval_return']/np.sqrt(num_eval_runs),3)}")
            metric_logger.save_metrics('eval', **save_metrics)

        if metric_logger.save_model_freq > 0:  # there's an option not to save models
            if (i_step+1) % metric_logger.save_model_freq == 0:
                # extra_log = {}

                metric_logger.save_metrics(option='model',
                                                agent=agent,
                                                total_num_steps=(i_step+1),
                                                # **extra_log,
                                                save_path=save_path,
                                                save_tag=save_tag)
                if agent_parameters['algorithm'] == 'ppo_agent':  # save states for checking later
                    metric_logger.save_metrics(option='states', agent=agent,
                                                    total_num_steps=(i_step+1),
                                                    states=agent.obs.clone().detach(),
                                                    save_path=save_path, save_tag=save_tag)

        sys.stdout.flush()
        sys.stderr.flush()

        i_step += 1


    # save data
    metric_logger.save_to_file(save_path, save_tag=save_tag)


    print('done RL run. Time {} min'.format((time.perf_counter() - start_time)/60))

    sys.stdout.flush()
    sys.stderr.flush()
    return metric_logger



if __name__ == "__main__":
    main()