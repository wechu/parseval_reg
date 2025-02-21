#####
# Gridworlds to demonstrate maladaptive representations
#
#
######
import numpy as np
from collections import defaultdict
from itertools import product
import gymnasium as gym


class GridWorldEnv:
    def __init__(self, image_obs, include_goal_obs, reward_type='opt', change_freq=-1, start_state=(4,4), goal_states=None, walls=None,
                 gridsize=None, rescale_state=True, seed=None, time_limit=100, flatten_obs=False, non_sparse_obs=False,
                 test_set=False, *args, **kwargs):
        '''
        image_obs: Gives the state as a stack of grids.
        include_goal_obs: Gives the goal location to the agent as part of the obs
        reward_type: "sparse", "dist", "constant", "opt"
        change_freq: Frequency to change the goal location
        random_goal: If True, randomizes the goal at each episode
        walls: List of walls as indicated by pairs of adjacent cells for which there is a wall in between.
        rescale_state: If True, divides the state position by the gridsize so it's between 0 and 1
        start_state: Int. Specifies the starting location at cell (start_state, start_state)
        non_sparse_obs: If True and image_obs=True, then 0s are replaced by -1s in the obs.
        '''
        self.name = 'gridworld'
        # the x-position goes from [0, gridsize[0]-1] and y-position goes from [0, gridsize[1]-1]
        self.num_actions = 5
        self.rng = np.random.default_rng(seed)

        self.discount = 1.0  # for value iteration, policy eval
        # could be 1 if we have negative rewards per step.

        self.include_goal_obs = include_goal_obs
        self.change_freq = change_freq
        self.rescale_state = rescale_state
        self.reward_type = reward_type
        self.time_limit = time_limit
        self.flatten_obs = flatten_obs
        self.non_sparse_obs = non_sparse_obs
        self.test_set = test_set

        if gridsize is None:
            gridsize = [9,9]
        self.gridsize = np.array(gridsize)
        self.image_obs = image_obs

        # observation and action spaces
        if self.image_obs:
            num_channels = 1
            if self.include_goal_obs:
                num_channels += 1
            self.observation_space = gym.spaces.Box(0, 1,
                                                    shape=(num_channels, self.gridsize[0], self.gridsize[1]),
                                                    dtype=np.float32)
            if flatten_obs:
                self.observation_space = gym.spaces.Box(0, 1,
                                                        shape=[num_channels*self.gridsize[0]*self.gridsize[1]],
                                                        dtype=np.float32)

        else:
            obs_length = 2
            if self.include_goal_obs:
                obs_length += 2
            self.observation_space = gym.spaces.Box(0, 1, shape=[obs_length], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

        self.steps = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right, stay ([0, 0])

        self.start_state = np.array(start_state)
        self.goal_states = goal_states
        # self.goal_states = [[np.array([self.gridsize[0]-1, self.gridsize[1]-1]), 1]]  # [goal, reward]

        if walls is None:
            walls = []
        self.walls = np.array(walls)  # list of pairs of locations. Walls are between each pair of locations. E.g. ((0,0), (1,0))
        # self.walls is indexed by (wall_index, cell_index (0 or 1), cell_coordinate (0 or 1) )

        self.pos = self.start_state.copy()
        self.total_timestep = 0
        self.episode_timestep = 0
        self.min_steps_array = None

        self._make_goal()
        self.reset()

    def reset(self):
        self.pos = self.start_state.copy()
        self.episode_timestep = 0

        # obs = self.pos.copy()

        if self.image_obs:
            obs = self.state_to_image(self.pos)
        else:
            obs = np.array(self.pos.copy())
            if self.rescale_state:
                obs = obs / self.gridsize

        return obs, {}  # no info

    def copy(self):
        copy_env = GridWorldEnv(self.image_obs, self.include_goal_obs,
                                self.reward_type, change_freq=-1, start_state=self.start_state,
                                goal_states=self.goal_states, walls=self.walls, gridsize=self.gridsize,
                                rescale_state=self.rescale_state, seed=None, time_limit=self.time_limit)
        copy_env.pos = self.pos.copy()
        copy_env.goal_states = self.goal_states

        return copy_env

    def _make_goal(self):
        ''' Sets a random goal '''
        goal_state = self.start_state.copy()
        while np.all(goal_state == self.start_state):
            goal_state = self.rng.integers(self.gridsize)

        if self.reward_type == 'opt':
            self.solve_steps_to_goal()  # initialize min steps array
        self.goal_states = [[goal_state, 1]]


    def _check_valid_pos(self, state):
        return 0 <= state[0] < self.gridsize[0] and 0 <= state[1] < self.gridsize[1]

    def _check_goal(self, state):
        for goal_state, goal_reward in self.goal_states:
            if np.all(goal_state == state):
                return True, goal_reward
        return False, 0.0

    def _shaped_reward(self, state):
        ''' Add reward to get a dense one '''
        if self.reward_type == 'dist':
            # l1 dist
            return -0.01 * np.sum(np.abs(state - self.goal_states[0][0]))
        elif self.reward_type == 'constant':
            return -0.01
        elif self.reward_type == 'opt':  # shortest path distance
            return -0.01 * self.min_steps_array[tuple(state)]
        else:
            return 0

    def step(self, action):
        if not np.ndim(action) == 0:  # https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
            raise AssertionError("Action needs to be an int, instead it is:", action)

        # if not (isinstance(action, int) or (isinstance(action, np.ndarray) and np.isscalar(action))):
        #     raise AssertionError("Action needs to be an int, instead it is:", action)

        self.total_timestep += 1
        self.episode_timestep += 1
        query_state = self.pos
        reward, next_states, next_state_probs, done = self.transition_dist(query_state, action)
        truncate = False

        # print(next_states)
        if len(next_states) > 1:
            idx = np.random.choice(np.arange(len(next_states)), p=next_state_probs)
            next_state = next_states[idx]
        else:
            next_state = next_states[0]

        self.pos = np.array(next_state)

        if self.image_obs:
            obs = self.state_to_image(self.pos)
        else:
            obs = np.array(next_state)
            if self.include_goal_obs:
                obs = np.append(obs, self.goal_states[0][0])
            if self.rescale_state:
                obs = obs / self.gridsize[0]

        if self.change_freq > 0 and self.total_timestep % self.change_freq == 0:
            print(self.total_timestep)

            self._make_goal()
            print("GOAL", self.goal_states[0][0])

            truncate = True

        if self.episode_timestep >= self.time_limit:
            # print("TIME LIMIT")
            truncate = True
            self.episode_timestep = 0

        return obs, reward, done, truncate, {}  # truncated, info  are last two

    def _check_wall(self, state, next_state):
        ''' Returns true if there is a collision with a wall.
        Assumes state, next_state and wall are numpy arrays '''
        for wall in self.walls:
            if np.all(state == wall[0]) and np.all(next_state == wall[1]):
                return True
            elif np.all(state == wall[1]) and np.all(next_state == wall[0]):
                return True
        return False

    def transition_dist(self, state, action):
        # returns a list of reward, next_states, next_states_prob and done
        # each returned value is a list
        # next_states and next_states_prob are a list of possible next states and their probabilities (nonzero)
        # done is a boolean which is true if state is terminal
        # Note that in this env, done is only true when the agent takes an action _after_ having reached a terminal state
        state = np.array(state)
        next_states_and_prob = defaultdict(lambda: 0.0)  # keys are next states, values are probabilities

        # # done and reward only depend on the current state
        done, reward = self._check_goal(state)
        reward += self._shaped_reward(state)

        # for movement
        # first, consider the case of not slipping
        temp_state = state + self.steps[action]

        if self._check_valid_pos(temp_state) and not self._check_wall(state, temp_state):
            next_state = temp_state
        else:
            next_state = state
        next_states_and_prob[tuple(next_state)] = 1.0

        # check this toggle
        # done and reward only depend on the next state
        # this is only correct if next_state is deterministic
        # done, reward = self._check_goal(next_state)

        return reward, list(next_states_and_prob.keys()), list(next_states_and_prob.values()), done

    def state_to_image(self, position):
        # Returns a 10x10 grid
        # position: the location of interest
        # we add a channel for other information like goal states
        num_channels = 1
        if self.include_goal_obs:
            num_channels += 1
        if self.non_sparse_obs:
            img = -np.ones(shape=(num_channels, self.gridsize[0], self.gridsize[1]), dtype='int8')
        else:
            img = np.zeros(shape=(num_channels, self.gridsize[0], self.gridsize[1]), dtype='int8')

        img[0, position[0], position[1]] = 1  # agent position
        if self.include_goal_obs:
            img[1, self.goal_states[0][0][0] , self.goal_states[0][0][1]] = 1  # goal position

        if self.flatten_obs:
            img = img.flatten()
        return img

    def preprocess_state(self, state):
        # rescales the state between 0 and 1
        return np.array(state) / self.gridsize

    def value_iteration(self, init_values=None):
        # to be fixed
        # returns the optimal values and optimal policy
        tolerance = 1e-6
        max_error = 99999
        iter = 0

        if init_values:
            q_values = init_values
        else:
            q_values = 1/(1-self.discount) /2 * np.ones((self.num_states, self.num_actions))  # initialize a guess to the values
            # q_values = np.zeros((self.num_states, self.num_actions))
        while max_error > tolerance:
            iter += 1
            copy_q_values = q_values.copy()
            for s,a in product(range(self.num_states), range(self.num_actions)):
                max_q_values = np.max(q_values, axis=1)

                q_values[s,a] = self.reward(s,a) + self.discount * np.sum(self.transition(s,a) * max_q_values)

            max_error = np.max(np.abs(copy_q_values - q_values))
        # print("iter", iter)
        return q_values, np.argmax(q_values, axis=1)

    def policy_evaluation(self, policy):
        ''' Returns a list of all state-actions and their associated values
        Q-values and V-values
        policy: this is a numpy array with shape (gridsize[0], gridsize[1], num_actions) containing the policy
            probabilities in each entry '''
        tolerance = 1e-6
        max_error = 99999
        iter = 0

        q_values = 0.5 * np.ones((self.gridsize[0], self.gridsize[1], self.num_actions))

        while max_error > tolerance:
            iter += 1
            copy_q_values = q_values.copy()
            q_value_targets = np.sum(q_values * policy, axis=2)

            for i,j,a in product(range(self.gridsize[0]), range(self.gridsize[1]), range(self.num_actions)):
                reward, next_states, next_states_prob, done = self.transition_dist(np.array([i,j]), a)
                updated_q = reward
                # if i == 5 and j == 0:
                #     print("here", reward, done)
                if not done:
                    for idx in range(len(next_states)):
                        updated_q += self.discount * next_states_prob[idx] * q_value_targets[next_states[idx]]  # uses tuple indexing
                q_values[i,j,a] = updated_q
                # print(updated_q)
            #
            max_error = np.max(np.abs(copy_q_values - q_values))
            print(iter, max_error)

        state_values = np.sum(q_values * policy, axis=2)

        return q_values, state_values
    #
    # def print_visual(self):
    #     for j in range(self.gridsize[1]):
    #         print('.', sep='')
    #
    #         for
    #         print('')

    def solve_steps_to_goal(self):
        ''' Finds the minimum number of steps to reach the goal from all states '''
        # print('SOLVING')
        num_steps_array = -np.ones(self.gridsize)
        goal_state = self.goal_states[0][0]  # assume there's only one goal state

        # proceed backwards from the goal state
        # use BFS
        queue = [[goal_state, 0]]  # (state, num_steps) pairs

        while len(queue) > 0:
            # print(len(queue))
            # pop from queue
            query_state, query_num_steps = queue[0]
            # query_num_steps is the number of steps it could take to reach query_state
            del queue[0]

            # update min number of steps
            if num_steps_array[tuple(query_state)] == -1:  # not visited yet
                num_steps_array[tuple(query_state)] = query_num_steps
                # if visited already, we have must've assigned a lower (or equal) number of steps previously

                # append neighbor states
                for step in self.steps:
                    temp_state = query_state + step
                    if not self._check_wall(query_state, temp_state) and self._check_valid_pos(temp_state):
                        queue.append([query_state + step, query_num_steps+1])

        # all states have been traversed and we have the min number of steps
        self.min_steps_array = num_steps_array

        return num_steps_array.copy()


    def evaluate_agent(self, agent, num_eval_episodes=10):
        ''' Runs and evaluation of the agent
        On the current task '''


        # test_env = self._wrap_env(self.base_task_class(seed=self.current_seed))
        test_env = self.copy()

        obs, _ = test_env.reset()

        eval_results = {}
        episodic_returns = []
        successes = []
        episode_steps = []
        agent.eval()
        r = 0
        steps = 0
        while len(episodic_returns) < num_eval_episodes:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = test_env.step(action)
            r += reward * self.discount**steps
            #
            # if "episode" in info:
            #     episodic_returns.append(info['episode']['r'])
            obs = next_obs
            if terminated or truncated:
                test_env = self.copy()  # just to make sure the goal stays the same
                # print(steps)
                obs, _ = test_env.reset()
                episodic_returns.append(r)
                successes.append(terminated)  # check if we reached the goal
                episode_steps.append(steps)

                r = 0
                steps = 0

            steps += 1

        agent.train()
        eval_results['episodic_returns'] = episodic_returns
        eval_results['successes'] = successes
        eval_results['episode_steps'] = episode_steps
        eval_results['optimal_length'] = np.sum(np.abs(self.start_state - self.goal_states[0][0]))  # for the one room env
        # print(np.round(episodic_returns,3))
        return eval_results

    def get_all_states(self):
        """ Returns an array which consists of a batch of all states in the gridworld """
        from itertools import product
        state_lst = []
        for x,y in product(range(self.gridsize[0]), range(self.gridsize[1])):
            state_lst.append(self.state_to_image([x,y]))

        return np.array(state_lst)



class NineRoomsEnv(GridWorldEnv):
    def __init__(self, hard_exploration=2, reward_type='opt', include_goal_obs=False, image_obs=True,
                 change_freq=-1, seed=None, *args, **kwargs):
        ''' A nine room env
        looks like 9 rooms
        A B C
        D E F
        G H I
        All rooms are connected to adjacent ones except D-E and E-F
        Agent starts at the middle of room E
        hard_exploration: If True, removes the door from E-H. So the agent can only leave the first room
            by going E -> B.

        The default arguments correspond to those in the paper.
            '''
        walls = [((4, i), (5, i)) for i in range(15)]
        walls.extend([((i, 4), (i, 5)) for i in range(15)])
        walls.extend([((i, 9), (i, 10)) for i in range(15)])
        walls.extend([((9, i), (10, i)) for i in range(15)])
        # doorways
        # vertical passages
        walls.remove(((4,2), (5,2)))
        walls.remove(((4,7), (5,7)))
        walls.remove(((4,12), (5,12)))
        walls.remove(((9,2), (10,2)))
        if hard_exploration is not None:  # then we keep one more wall
            walls.remove(((9,7), (10,7)))
        walls.remove(((9,12), (10,12)))
        # horizontal passages
        walls.remove(((2,4), (2,5)))
        # walls.remove(((7,4), (7,5)))
        walls.remove(((12,4), (12,5)))
        walls.remove(((2,9), (2,10)))
        # walls.remove(((7,9), (7,10)))
        walls.remove(((12,9), (12,10)))

        self.hard_exploration = hard_exploration

        self.goal_room = None  # we generate a goal in a different room than the current one

        super().__init__(start_state=(7,7),
                         image_obs=image_obs,
                         include_goal_obs=include_goal_obs, change_freq=change_freq,
                         walls=walls, reward_type=reward_type,
                         gridsize=(15,15), seed=seed)

    def _make_goal(self):
        ''' Sets a random goal.
        '''
        # goal_state = self.rng.integers(0, 15, size=2)
        goal_state = self.start_state

        def _in_start_room(state):
            return 5 <= state[0] <= 9 and 5 <= state[1] <= 9
        def _in_top_rooms(state):
            return state[0] <= 4

        invalid_location = True
        while invalid_location:  # we sample a goal location outside of the starting room
            goal_state = self.rng.integers(0, 15, size=2)
            if self.hard_exploration == 2:
                invalid_location = _in_start_room(goal_state) or _in_top_rooms(goal_state)
            else:
                invalid_location = _in_start_room(goal_state)

        self.goal_states = [[goal_state, 3]]
        # reward is 3 so that positive returns are possible with the optimal policy
        self.solve_steps_to_goal()


class NineRoomsEnvWrapper:
    def __init__(self, *args, **kwargs):
        """ This is a dummy wrapper just so that the env respects the interface used by the metaworld_ppo.py agent
        The agent uses env.env so assumes there is a wrapper around a base env """
        self.env = NineRoomsEnv(*args, **kwargs)

        self.goal_states = self.env.goal_states
        print('Gridworld args', args, kwargs)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def evaluate_agent(self, agent, num_eval_episodes=10):
        return self.env.evaluate_agent(agent, num_eval_episodes)




if __name__ == "__main__":
    # test four rooms

    cfg_dict = {
        "num_steps": 800000,
        "seed": 123,
        "change_freq": 40000,
        "include_goal_obs": False,
        "image_obs": True,
        "non_sparse_obs": False,
        "reward_type": "opt",
        "hard_exploration": 2
    }
    env = NineRoomsEnvWrapper(**cfg_dict)



    # quit()
    # env = NineRoomsEnv(change_freq=5)
    # print(env.reward_type)
    # env.reset()
    # print(env.min_steps_array)

    # actions = [1, 3,3,3,3,3,3,3,3, 1,1,1,1,1,1,1,1]
    # for i in range(len(actions)):
    #     # act = np.random.randint(5)
    #     # act = 0
    #     act = actions[i]
    #     obs, r, terminated, truncated, _ = env.step(act)

    #     print(i, act, r, terminated)
    #     # print(obs[0])
    #     if terminated or truncated:
    #         obs, _ = env.reset()
