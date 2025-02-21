######
# Used for sequences of CARL environments
#
#
######
import numpy as np

from carl.envs import CARLCartPole, CARLAcrobot, CARLPendulum, CARLDmcFingerEnv, CARLDmcWalkerEnv, CARLDmcQuadrupedEnv, CARLBipedalWalker, CARLLunarLander, CARLVehicleRacing
import gymnasium as gym
from gym.wrappers import TimeLimit, RecordEpisodeStatistics, RecordVideo, TransformObservation, FlattenObservation
from carl.context.selection import StaticSelector

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# env = CARLCartPole(contexts=contexts, hide_context=False, dict_observation_space=True)

# c

class CARLSingleEnv:
    def __init__(self, base_task_name=None, normalize_obs="straight",
                 normalize_avg_coef=0.0001,
                 normalize_rewards=False, reset_obs_stats=False,
                 capture_video=False, seed=None, *args, **kwargs):
        '''
        This class is used to generate a stream of tasks in one of the
        Each task is generated with a new random seed. (I think you can go on forever?)
        base_task_name: name of CARL env e.g. "dmcwalker", "acrobot"

        '''
        self.base_task_name = base_task_name

        # consider making the goal hidden
        self.context_vars = {}
        for arg, val in kwargs.items():
            if arg[:5] == 'carl_':  # check if it denotes a context variable
                self.context_vars[arg[5:]] = val
        print("CONTEXT", self.context_vars)
        self.env = None
        self.base_seed = seed
        self.current_seed = seed
        self.normalize_obs = normalize_obs
        self.normalize_avg_coef = normalize_avg_coef
        self.normalize_rewards = normalize_rewards
        self.reset_obs_stats = reset_obs_stats

        self.timestep_counter = 0
        self.task_counter = 0
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 1e-4  # initialize it to a small value (as per the gym.NormalizeObservation wrapper)
        self.bias_correction = False

        self.rng = np.random.RandomState(seed=self.current_seed)
        # the default magnitude of the observations are about ~1 or smaller I think

        self.make_task()



    def reset(self):
        obs, info = self.env.reset()

        if self.normalize_obs is not None:
            obs = self._normalize_obs(obs)
            # self._update_obs_statistics(obs)
        # obs = obs.astype('float32')
        return obs, info

    def make_task(self):
        self.current_seed += 100

        if self.base_task_name == 'dmcfinger':
            self.base_task_class = CARLDmcFingerEnv
        elif self.base_task_name == 'dmcwalker':
            self.base_task_class = CARLDmcWalkerEnv
        elif self.base_task_name == 'dmcquadruped':
            self.base_task_class = CARLDmcQuadrupedEnv
        elif self.base_task_name == 'lunarlander':
            self.base_task_class = CARLLunarLander
        elif self.base_task_name == 'bipedalwalker':
            self.base_task_class = CARLBipedalWalker
        elif self.base_task_name == 'vehicleracing':
            self.base_task_class = CARLBipedalWalker
        elif self.base_task_name == 'pendulum':
            self.base_task_class = CARLPendulum
        elif self.base_task_name == 'acrobot':
            self.base_task_class = CARLAcrobot
        elif self.base_task_name == 'cartpole':
            self.base_task_class = CARLCartPole
        else:
            raise AssertionError("Invalid task name", self.base_task_name)

        custom_context = self.base_task_class.get_default_context()
        print("DEFAULT CONTEXT", custom_context)


        for context_var, context_scale in self.context_vars.items():
            custom_context[context_var] *= context_scale  # we multiply the default by the factor specified

        self.task_context = custom_context

        # self.env = gym.make("carl/CARLF")
        self.env = self.base_task_class(context_selector=StaticSelector({0: custom_context}), hide_context=True,
                                 dict_observation_space=False)
        self.discrete_action_space = isinstance(self.env.action_space, gym.spaces.Discrete)  # check if discrete or not

        self.env = self._wrap_env(self.env)

        self.env.observation_space = self.env.observation_space['obs']  # override so it matches the gym interface

        # we only pass one context at a time
        # The default behaviour in CARL is to cycle through different contexts if we pass more than 1, which we don't want
        # I guess StaticSelector let's the context be fixed actually

        # if self.env_sequence is not None:
        #     if self.env_sequence[-2:].isnumeric():# we expect something like "set15" or "set3"
        #         env_set_id = int(self.env_sequence[-2:])
        #     else:
        #         env_set_id = int(self.env_sequence[-1])
        #     self.base_task_name = RPO10_SEQ[env_set_id-1][(self.task_counter-1) % len(RPO10_SEQ[0])]
        #     # self.base_task_name = SEQUENCE_ENVS[(self.task_counter-1) % len(SMALL_SEQUENCE_ENVS)]  # for testing
        #     self.base_task_class = self.metaworld_envs[self.base_task_name + self.goal_str]

        # self.env = self._wrap_env(self.base_task_class(seed=self.current_seed))

        # self.env.action_space.seed(self.current_seed)
        # self.env.observation_space.seed(self.current_seed)

        if self.obs_mean is None or self.reset_obs_stats:
            self.obs_mean = np.zeros(self.env.observation_space.shape)

            if self.bias_correction:
                self.obs_var = np.zeros(self.env.observation_space.shape)
            else:
                self.obs_var = np.ones(self.env.observation_space.shape)

        print(f'TASK {self.task_counter}  {self.current_seed} {self.base_task_name}')
        return

    def _wrap_env(self, env, eval_mode=False):
        # env = RecordVideo(env)  # do this later
        # env = FlattenObservation(env)
        env = TransformObservation(env, lambda obs: obs['obs'])  # extracts obs from the dict (not context)
        env = TransformObservation(env, lambda obs: obs.astype('float32'))
        if not self.discrete_action_space:
            env = gym.wrappers.ClipAction(env)
        # if not eval_mode and self.normalize_rewards:
        #     # env = gym.wrappers.NormalizeReward(env)  # note: this resets whenever the task switches
        #     env = gym.wrappers.TransformReward(env, lambda r: r / 500)  # todo get a good reward normalization
        # env = TimeLimit(env, max_episode_steps=200)
        env = RecordEpisodeStatistics(env)

        # don't use normalize observations wrapper here because we need to use the same one for the test env
        # in evaluate_agent()
        # don't use any wrapper that depends on maintaining statistics
        return env

    def step(self, action):
        self.timestep_counter += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.normalize_obs is not None:
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)

        # reward = reward / 1000 # tried rescaling. consider if I should be doing this. Reward scale does matter to optimization

        return obs, reward, terminated, truncated, info

    def evaluate_agent(self, agent, num_eval_episodes=10):
        ''' Runs and evaluation of the agent
        It runs on the current seed i.e. the current task '''

        test_env = self._wrap_env(self.base_task_class(context_selector=StaticSelector({0: self.task_context}), hide_context=True,
                                 dict_observation_space=False), eval_mode=True)
        obs, _ = test_env.reset()

        if self.normalize_obs is not None:
            obs = self._normalize_obs(obs)

        eval_results = {}
        episodic_returns = []
        agent.eval()

        while len(episodic_returns) < num_eval_episodes:
            action = agent.act(obs)

            next_obs, _, terminated, truncated, info = test_env.step(action)

            if self.normalize_obs is not None:
                next_obs = self._normalize_obs(next_obs)

            if "episode" in info:
                episodic_returns.append(info['episode']['r'])

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if "episode" not in info:
            #             continue
            #         # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            #         episodic_returns.append(info["episode"]["r"][0])
            obs = next_obs

            if terminated or truncated:
                obs, _ = test_env.reset()

        agent.train()
        eval_results['episodic_returns'] = episodic_returns
        # eval_results['successes'] = test_env.pop_successes()

        return eval_results

    def _update_obs_statistics(self, obs):
        ''' Update mean and variance statistics '''

        if self.normalize_obs.lower() == 'ema':
            # print("Using EMA")
            # print(self.timestep_counter)
            # print('obs', obs[:4])
            # print('obs mean', self.obs_mean[:4])
            self.obs_mean = (1 - self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs
            # print('obs mean 2', self.obs_mean[:4])
            self.obs_var = (1 - self.normalize_avg_coef) * self.obs_var + self.normalize_avg_coef * (
                        obs - self.obs_mean) ** 2
            # print("NORM", np.linalg.norm(self.obs_var))
            # print('obs var', self.obs_var)
            # print(self.normalize_avg_coef)
            # print(obs[0:5], self.obs_mean[0:5])#, (1-self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs )

        elif self.normalize_obs.lower() == 'straight':
            mean = self.obs_mean
            var = self.obs_var
            count = self.obs_count

            delta = obs - mean
            tot_count = count + 1

            new_mean = mean + delta / tot_count
            new_var = var * count / tot_count + delta ** 2 * count / tot_count ** 2
            new_count = tot_count

            self.obs_mean = new_mean
            self.obs_var = new_var
            self.obs_count = new_count

    def _normalize_obs(self, obs):
        if self.timestep_counter == 0:
            return obs  # don't normalize when there's no data

        if self.bias_correction:
            bias_correction = 1 - (1 - self.normalize_avg_coef) ** self.timestep_counter
            obs_mean = self.obs_mean / bias_correction
            obs_var = self.obs_var / bias_correction
        else:
            obs_mean = self.obs_mean
            obs_var = self.obs_var
        normalized_obs = (obs - obs_mean) / (np.sqrt(obs_var) + 1e-8)
        normalized_obs = np.clip(normalized_obs, -10, 10)
        return normalized_obs



from .carl_env_sequences import *

class CARLSequence:
    def __init__(self,  change_freq, base_task_name=None, env_sequence=None, normalize_obs="straight", normalize_avg_coef=0.0001,
                 normalize_rewards=False, reset_obs_stats=False,
                 capture_video=False, seed=None, *args, **kwargs):
        '''
        This class is used to generate a stream of tasks in one of the
        Each task is generated with a new random seed. (I think you can go on forever?)
        base_task_name: name of CARL env e.g. "dmcwalker", "acrobot"
        change_freq: number of steps until a task change
        goal_hidden: If True, the agent does not observe the goal location

        '''

        self.base_task_name = base_task_name
        self.env_sequence = env_sequence

        # consider making the goal hidden

        self.env = None
        self.base_seed = seed
        self.current_seed = seed
        self.normalize_obs = normalize_obs
        self.normalize_avg_coef = normalize_avg_coef
        self.normalize_rewards = normalize_rewards
        self.reset_obs_stats = reset_obs_stats

        self.change_freq = change_freq
        self.timestep_counter = 0
        self.task_counter = 0
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 1e-4  # initialize it to a small value (as per the gym.NormalizeObservation wrapper)
        self.bias_correction = False

        self.rng = np.random.RandomState(seed=self.current_seed)
        # the default magnitude of the observations are about ~1 or smaller I think

        self.make_task()

    def reset(self):
        obs, info = self.env.reset()

        if self.normalize_obs is not None:
            obs = self._normalize_obs(obs)
            # self._update_obs_statistics(obs)
        # obs = obs.astype('float32')
        return obs, info

    def make_task(self):
        self.current_seed += 100
        self.task_counter += 1

        # pendulum_context_list = [{"gravity": 0.8}, {"gravity": 1.0}, {"gravity": 1.2}]
        # dmcfinger_context_list = [{"gravity": 0.8}, {"gravity": 1.0}, {"gravity": 1.2}]
        if self.base_task_name[-2:].isnumeric():
            seq_number = int(self.base_task_name[-2:])
            task_name = self.base_task_name[:-3]  # we expect something like "dmcfinger_12"
        else:
            seq_number = int(self.base_task_name[-1])
            task_name = self.base_task_name[:-2]  # we expect something like "dmcfinger_3"

        if task_name == 'dmcfinger':
            self.base_task_class = CARLDmcFingerEnv
            context_fn = generate_dmcfinger_seq
        elif task_name == 'dmcwalker':
            self.base_task_class = CARLDmcWalkerEnv
            context_fn = generate_dmcwalker_seq
        elif task_name == 'dmcquadruped':
            self.base_task_class = CARLDmcQuadrupedEnv
            context_fn = generate_dmcquadruped_seq
        elif task_name == 'lunarlander':
            self.base_task_class = CARLLunarLander
            context_fn = generate_lunarlander_seq

        elif task_name == 'bipedalwalker':
            self.base_task_class = CARLBipedalWalker
            context_fn = generate_bipedalwalker_seq
        elif task_name == 'pendulum':
            self.base_task_class = CARLPendulum

        elif task_name == 'acrobot':
            self.base_task_class = CARLAcrobot
            context_fn = generate_acrobot_seq
        elif task_name == 'cartpole':
            self.base_task_class = CARLCartPole
            context_fn = generate_cartpole_seq
        else:
            raise AssertionError('invalid task name', task_name)
        custom_context = self.base_task_class.get_default_context()

        # seq_number = 1  # indexes which sequence we want to use
        context_dict = context_fn(self.task_counter, seq_number)
        print(context_dict)

        for context_var, context_scale in context_dict.items():
            if context_var == 'GRAVITY_X':
                custom_context[context_var] = context_scale  # we set it to the value specified (default is 0)
            else:
                custom_context[context_var] *= context_scale  # we multiply the default by the factor specified

        self.task_context = custom_context  # save this so we can use for evaluate_agent()

        # self.env = gym.make("carl/CARLF")
        self.env = self.base_task_class(context_selector=StaticSelector({0: custom_context}), hide_context=True,
                                 dict_observation_space=False)
        self.discrete_action_space = isinstance(self.env.action_space, gym.spaces.Discrete)  # check if discrete or not

        self.env = self._wrap_env(self.env)

        self.env.observation_space = self.env.observation_space['obs']  # override so it matches the gym interface

        # we only pass one context at a
        # we only pass one context at a time
        # The default behaviour in CARL is to cycle through different contexts if we pass more than 1, which we don't want
        # I guess StaticSelector let's the context be fixed actually


        # if self.env_sequence is not None:
        #     if self.env_sequence[-2:].isnumeric():# we expect something like "set15" or "set3"
        #         env_set_id = int(self.env_sequence[-2:])
        #     else:
        #         env_set_id = int(self.env_sequence[-1])
        #     self.base_task_name = RPO10_SEQ[env_set_id-1][(self.task_counter-1) % len(RPO10_SEQ[0])]
        #     # self.base_task_name = SEQUENCE_ENVS[(self.task_counter-1) % len(SMALL_SEQUENCE_ENVS)]  # for testing
        #     self.base_task_class = self.metaworld_envs[self.base_task_name + self.goal_str]

        # self.env = self._wrap_env(self.base_task_class(seed=self.current_seed))

        # self.env.action_space.seed(self.current_seed)
        # self.env.observation_space.seed(self.current_seed)


        if self.obs_mean is None or self.reset_obs_stats:
            self.obs_mean = np.zeros(self.env.observation_space.shape)

            if self.bias_correction:
                self.obs_var = np.zeros(self.env.observation_space.shape)
            else:
                self.obs_var = np.ones(self.env.observation_space.shape)

        print(f'TASK {self.task_counter}  {self.current_seed} {self.base_task_name}')
        return

    def _wrap_env(self, env, eval_mode=False):
        # env = RecordVideo(env)  # do this later
        # env = FlattenObservation(env)
        env = TransformObservation(env, lambda obs: obs['obs'])  # extracts obs from the dict (not context)

        env = TransformObservation(env, lambda obs: obs.astype('float32'))
        if not self.discrete_action_space:
            env = gym.wrappers.ClipAction(env)
        # if not eval_mode and self.normalize_rewards:
        #     # env = gym.wrappers.NormalizeReward(env)  # note: this resets whenever the task switches
        #     env = gym.wrappers.TransformReward(env, lambda r: r / 500)  # todo get a good reward normalization
        # env = TimeLimit(env, max_episode_steps=200)
        env = RecordEpisodeStatistics(env)

        # don't use normalize observations wrapper here because we need to use the same one for the test env
        # in evaluate_agent()
        # don't use any wrapper that depends on maintaining statistics
        return env

    def step(self, action):
        self.timestep_counter += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.normalize_obs is not None:
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)
            # print(obs[:8])
            # np.set_printoptions(suppress=True)
        # obs = obs.astype('float32')

        # reward = reward / 1000 # tried rescaling. consider if I should be doing this. Reward scale does matter to optimization

        if (self.timestep_counter+1) % self.change_freq == 0:
                self.make_task()
                truncated = True
                # return obs, reward, terminated, True, info   # truncate current episode, get reset called

        return obs, reward, terminated, truncated, info

    def evaluate_agent(self, agent, num_eval_episodes=10):
        ''' Runs and evaluation of the agent
        It runs on the current seed i.e. the current task '''

        test_env = self._wrap_env(self.base_task_class(context_selector=StaticSelector({0: self.task_context}), hide_context=True,
                                 dict_observation_space=False), eval_mode=True)

        obs, _ = test_env.reset()

        if self.normalize_obs is not None:
            obs = self._normalize_obs(obs)

        eval_results = {}
        episodic_returns = []
        agent.eval()

        while len(episodic_returns) < num_eval_episodes:
            action = agent.act(obs)

            next_obs, _, terminated, truncated, info = test_env.step(action)

            if self.normalize_obs is not None:
                next_obs = self._normalize_obs(next_obs)

            if "episode" in info:
                episodic_returns.append(info['episode']['r'])

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if "episode" not in info:
            #             continue
            #         # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            #         episodic_returns.append(info["episode"]["r"][0])
            obs = next_obs

            if terminated or truncated:
                obs, _ = test_env.reset()

        agent.train()
        eval_results['episodic_returns'] = episodic_returns
        # eval_results['successes'] = test_env.pop_successes()

        return eval_results

    def _update_obs_statistics(self, obs):
        ''' Update mean and variance statistics '''

        if self.normalize_obs.lower() == 'ema':
            # print("Using EMA")
            # print(self.timestep_counter)
            # print('obs', obs[:4])
            # print('obs mean', self.obs_mean[:4])
            self.obs_mean = (1-self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs
            # print('obs mean 2', self.obs_mean[:4])
            self.obs_var = (1-self.normalize_avg_coef) * self.obs_var + self.normalize_avg_coef * (obs - self.obs_mean)**2
            # print("NORM", np.linalg.norm(self.obs_var))
            # print('obs var', self.obs_var)
            # print(self.normalize_avg_coef)
            # print(obs[0:5], self.obs_mean[0:5])#, (1-self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs )

        elif self.normalize_obs.lower() == 'straight':
            mean = self.obs_mean
            var = self.obs_var
            count = self.obs_count

            delta = obs - mean
            tot_count = count + 1

            new_mean = mean + delta / tot_count
            new_var = var * count / tot_count + delta ** 2 * count / tot_count ** 2
            new_count = tot_count

            self.obs_mean = new_mean
            self.obs_var = new_var
            self.obs_count = new_count

    def _normalize_obs(self, obs):
        if self.timestep_counter == 0:
            return obs  # don't normalize when there's no data

        if self.bias_correction:
            bias_correction = 1 - (1 - self.normalize_avg_coef) ** self.timestep_counter
            obs_mean = self.obs_mean / bias_correction
            obs_var = self.obs_var / bias_correction
        else:
            obs_mean = self.obs_mean
            obs_var = self.obs_var
        normalized_obs = (obs - obs_mean) / (np.sqrt(obs_var) + 1e-8)
        normalized_obs = np.clip(normalized_obs, -10, 10)
        return normalized_obs


if __name__ == "__main__":


    test_env = CARLSingleEnv("dmcfinger", seed=5, carl_gravity=1.2)
    print(test_env.env.action_space)
    action = test_env.env.action_space.sample()
    print(action)
    obs, info = test_env.reset()
    print("OBS", obs)
    print("INFO", info)
    # print(test_env.env.observation_space)

    for i in range(10):
        state, reward, terminated, truncated, info = test_env.step(action)

        print(state, reward, terminated, truncated)
    #
    # # state, reward, terminated, truncated, info = env.step(action=action)
    #
    quit()