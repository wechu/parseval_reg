###
# Sequences of context variables for the CARL envs
#
#
###
import numpy as np

##### Discrete Action envs
### Acrobot

def generate_acrobot_seq(task_idx, seed):
    ''' Generates the seq of tasks for Acrobot
    Should generate the same sequence every time due to fixed seed
    Change seed to get different sequences
    500k steps is enough'''
    seed_offset = 2
    seed = seed + seed_offset
    rng = np.random.RandomState(seed)

    link_low = -1  # in logspace
    link_high = 0.5

    # torque_noise_low = -1
    # torque_noise_high = 1.0

    i = seed % 3

    def sample_context():
        global i
        if i % 3 == 0:
            torque_noise = 0
        elif i % 3 == 1:
            torque_noise = -1
        elif i % 3 == 2:
            torque_noise = 1
        i += 1

        links = np.power(10, np.array([rng.uniform(link_low, link_high), rng.uniform(link_low, link_high),
                         rng.uniform(link_low, link_high), rng.uniform(link_low, link_high)]))
        return np.concatenate([links, np.array([torque_noise])])

    ## ensure that subsequent draws are far enough from each other
    min_change = 0.0  # in l_1 distance

    context_list = [sample_context()]
    while len(context_list) < task_idx:
        context_vars = sample_context()
        if np.sum(np.abs(context_vars - context_list[-1])) > min_change:
            context_list.append(context_vars)

    context_var_names = ["LINK_LENGTH_1", "LINK_LENGTH_2", "LINK_MASS_1", "LINK_MASS_2", "TORQUE_NOISE_MAX"]
    return {var:value for var,value in zip(context_var_names, context_list[task_idx-1])}



### CartPole
def generate_cartpole_seq(task_idx, seed):
    ''' Generates the seq of tasks for CartPole
    Should generate the same sequence every time due to fixed seed
    Change seed to get different sequences
    500k steps is enough'''
    seed_offset = 2
    seed = seed + seed_offset
    rng = np.random.RandomState(seed)

    var_low = -0.5 # in logspace
    var_high = 0.5

    def sample_context():
        return np.power(10, np.array([rng.uniform(var_low, var_high), rng.uniform(var_low, var_high),
                         rng.uniform(var_low, var_high)]))

    ## ensure that subsequent draws are far enough from each other
    min_change = 1.5  # in l_1 distance

    context_list = [sample_context()]
    while len(context_list) < task_idx:
        context_vars = sample_context()
        if np.sum(np.abs(context_vars - context_list[-1])) > min_change:
            context_list.append(context_vars)

    context_var_names = ["gravity", "length", "force_mag"]
    return {var:value for var,value in zip(context_var_names, context_list[task_idx-1])}


### LunarLander

def generate_lunarlander_seq(task_idx, seed):
    ''' Generates the seq of tasks for lunar lander.
    Should generate the same sequence every time due to fixed seed
    Change seed to get different sequences'''
    rng = np.random.RandomState(seed)

    gravity_low = 0.2
    gravity_high = 1.5

    engine_low = 1.0
    engine_high = 1.5

    initial_low = 0.5
    initial_high = 1.5

    def sample_grav_x(i):
        # we make GRAVITY_X toggle between 3 settings: around +2, 0, -2
        if i % 3 == 0:
            grav_x = -2
        elif i % 3 == 1:
            grav_x = 0
        elif i % 3 == 2:
            grav_x = 2
        return grav_x

    def sample_context():
        return np.array([rng.uniform(gravity_low, gravity_high), rng.uniform(engine_low, engine_high), rng.uniform(engine_low, engine_high),
                         rng.uniform(initial_low, initial_high)])

    ## ensure that subsequent draws are far enough from each other
    min_change = 1.0  # in l_1 distance

    i = 0
    context_list = [np.concatenate([sample_context(), np.array([sample_grav_x(i)])])]
    while len(context_list) < task_idx:
        context_vars_random = sample_context()
        if np.sum(np.abs(context_vars_random - context_list[-1][:len(context_vars_random)])) > min_change:
            i += 1

            context_vars = np.concatenate([context_vars_random, np.array([sample_grav_x(i)])])
            context_list.append(context_vars)

    context_var_names = ['GRAVITY_Y', 'MAIN_ENGINE_POWER', 'SIDE_ENGINE_POWER', 'INITIAL_RANDOM', 'GRAVITY_X']
    return {var:value for var,value in zip(context_var_names, context_list[task_idx-1])}




##### Continuous Action envs (without DMC)

### Pendulum


### BipedalWalker
def generate_bipedalwalker_seq(task_idx, seed):
    ''' Generates the seq of tasks for BipedalWalker
    Should generate the same sequence every time due to fixed seed
    Change seed to get different sequences'''
    seed_offset = 2
    seed = seed + seed_offset
    rng = np.random.RandomState(seed)

    friction_low = 1.0  # can't be much less than 1, cannot learn
    friction_high = 2.0

    # scale_low_log = 0  # can't be too samll e.g <0.4 (regular scale) cannot learn
    # scale_high_log = 0.5

    motors_low = 0.7
    motors_high = 1.5

    gravity_low = 0.5
    gravity_high = 1.2


    def sample_grav_x(i):
        # we make GRAVITY_X toggle between 3 settings: around +2, 0, -2
        if i % 3 == 0:
            grav_x = -2 #+ rng.uniform(gravity_x_low, gravity_x_high)
        elif i % 3 == 1:
            grav_x = 0
        elif i % 3 == 2:
            grav_x = 2
        return grav_x


    def sample_context():
        return np.array([rng.uniform(friction_low, friction_high), rng.uniform(motors_low, motors_high),
                         rng.uniform(gravity_low, gravity_high)])

    ## ensure that subsequent draws are far enough from each other
    min_change = 1.0  # in l_1 distance

    i = 0
    context_list = [np.concatenate([sample_context(), np.array([sample_grav_x(i)])])]
    while len(context_list) < task_idx:
        context_vars_random = sample_context()
        if np.sum(np.abs(context_vars_random - context_list[-1][:len(context_vars_random)])) > min_change:
            i += 1

            context_vars = np.concatenate([context_vars_random, np.array([sample_grav_x(i)])])
            context_list.append(context_vars)


    context_var_names = ['FRICTION', 'SCALE', 'MOTORS_TORQUE', 'GRAVITY_Y', 'GRAVITY_X']
    return {var:value for var,value in zip(context_var_names, context_list[task_idx-1])}

##### DeepMind Control (DMC)
#### todo check results again for single envs
### Finger
def generate_dmcfinger_seq(task_idx, seed):
    ''' Generates the seq of tasks for DMCFinger
    Should generate the same sequence every time due to fixed seed
    Change seed to get different sequences'''
    seed_offset = 2
    seed = seed + seed_offset
    rng = np.random.RandomState(seed)

    limb_length_low = 0.9
    limb_length_high = 1.1

    def sample_context():
        return np.array([rng.uniform(limb_length_low, limb_length_high), rng.uniform(limb_length_low, limb_length_high)])

    ## ensure that subsequent draws are far enough from each other
    min_change = 0.1  # in l_1 distance

    context_list = [sample_context()]
    while len(context_list) < task_idx:
        context_vars = sample_context()
        if np.sum(np.abs(context_vars - context_list[-1])) > min_change:
            context_list.append(context_vars)
    return {'limb_length_0': context_list[task_idx-1][0], 'limb_length_1': context_list[task_idx-1][1]}

### Walker
def generate_dmcwalker_seq(task_idx, seed):
    ''' Generates the seq of tasks for DMCWalker
    Should generate the same sequence every time due to fixed seed
    Change seed to get different sequences'''
    seed_offset = 2
    seed = seed + seed_offset
    rng = np.random.RandomState(seed)

    gravity_list = [0.3, 0.8, 1.2]
    gravity_low = 0.3
    gravity_high = 1.2


    actuator_list = [0.8, 1.0, 1.2]
    actuator_strength_low = 0.8
    actuator_strength_high = 1.2

    # damping_low = 0.8
    # damping_high = 1.2
    stiffness_list = [0.8, 1.3, 2.0]
    stiffness_low = 0.8
    stiffness_high = 2.0

    viscosity_list = [0.8, 1.0, 1.2]
    viscosity_low = 0.8
    viscosity_high = 1.2

    def sample_wind(i):
        # we make GRAVITY_X toggle between 3 settings: around +2, 0, -2
        if i % 3 == 0:
            wind = -2 #+ rng.uniform(gravity_x_low, gravity_x_high)
        elif i % 3 == 1:
            wind = 0
        elif i % 3 == 2:
            wind = 2
        return wind, wind

    def sample_context():
        return np.array([rng.choice(gravity_list), rng.choice(actuator_list),
                         rng.choice(stiffness_list), rng.choice(viscosity_list)])

    # def sample_context():
    #     return np.array([rng.uniform(gravity_low, gravity_high), rng.uniform(actuator_strength_low, actuator_strength_high),
    #                      rng.uniform(stiffness_low, stiffness_high),rng.uniform(viscosity_low, viscosity_high)])

    ## ensure that subsequent draws are far enough from each other
    min_change = 0.1 # in l_1 distance
    #
    i = 0
    context_list = [np.concatenate([sample_context(), np.array(sample_wind(i))])]
    while len(context_list) < task_idx:
        context_vars_random = sample_context()
        if np.sum(np.abs(context_vars_random - context_list[-1][:len(context_vars_random)])) > min_change:
            i += 1

            context_vars = np.concatenate([context_vars_random, np.array(sample_wind(i))])
            context_list.append(context_vars)


    context_var_names = ['gravity', 'actuator_strength', 'joint_stiffness', 'viscosity', 'wind_x', 'wind_y']
    return {var:value for var,value in zip(context_var_names, context_list[task_idx-1])}

### Quadruped
def generate_dmcquadruped_seq(task_idx, seed):
    ''' Generates the seq of tasks for DMCQuadruped
    Should generate the same sequence every time due to fixed seed
    Change seed to get different sequences'''
    seed_offset = 2
    seed = seed + seed_offset
    rng = np.random.RandomState(seed)

    actuator_low = 0.3
    actuator_high = 1.3

    gravity_low = 0.3
    gravity_high = 1.3

    # damping_low = 0.7
    # damping_high = 1.3

    i = seed % 5
    wind_list = [(0, 0), (-1, -1), (1,1), (1, -1), (-1, 1)]
    def sample_context():
        nonlocal i
        wind_x, wind_y = wind_list[i]
        i = (i+1) % 5
        return np.array([rng.uniform(actuator_low, actuator_high), rng.uniform(gravity_low, gravity_high),
                         wind_x, wind_y])

    ## ensure that subsequent draws are far enough from each other
    min_change = 0.0  # in l_1 distance

    context_list = [sample_context()]
    while len(context_list) < task_idx:
        context_vars = sample_context()
        if np.sum(np.abs(context_vars - context_list[-1])) > min_change:
            context_list.append(context_vars)

    context_var_names = ['actuator_strength', 'gravity', 'wind_x', 'wind_y']
    return {var:value for var,value in zip(context_var_names, context_list[task_idx-1])}



########
if __name__ == '__main__':
    seq_number = 10  # increase viscosity

    # rng = np.random.RandomState(seed=3)
    # for i in range(10):
    #     print(rng.choice([1,3,5]))

    for i in range(1,9):
        print(i, generate_dmcwalker_seq(i, seq_number))