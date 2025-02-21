#####
# Load and plot results
#
#
#####



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from itertools import product 
import pickle
import copy

def _generate_default_file_paths(algorithm_used, env_used, num_repeats, num_task_sequences, base_folder='results/'):
    """ Helper function. Generates the default file paths for the results of the experiments which are used in run_many.py """
    if env_used == 'metaworld':
        env_names = [f'metaworld_sequence_set{i}' for i in range(num_task_sequences)]  
    elif env_used == 'carl_dmcquadruped':
        env_names = [f"carl_sequence_dmcquadruped_{i}" for i in range(num_task_sequences)]  
    elif env_used == 'carl_lunarlander':
        env_names = [f"carl_sequence_lunarlander_{i}" for i in range(num_task_sequences)]  
    elif env_used == 'gridworld':
        env_names = ['gridworld_ninerooms']  # gridworld randomizes the tasks for every run, so just use more repeats instead (e.g. 20)

    file_paths = []
    for env, i_repeat in product(env_names, range(num_repeats)):
        file_paths.append(f"{base_folder}data_{env}_{algorithm_used}_{i_repeat}.pkl")

    return file_paths

def load_results_for_one_alg_and_env(algorithm_used, env_used, num_repeats, num_task_sequences, base_folder='results/'):
    """ Load data from the results files for one combination of algorithm and environment
     e.g. algorithm is "base" and environment is "metaworld" 
     This function reorganizes the data so that the output all_data has keys which 
      are the saved metrics and values which are list of lists. First index is over repeats and sequence index
       while the second index is over the timesteps of the experiment. """
    
    file_paths = _generate_default_file_paths(algorithm_used, env_used, num_repeats, num_task_sequences, base_folder=base_folder)

    all_data = []

    for file_path in file_paths:
        with open(file_path, "rb") as file:
            data = pickle.load(file)  # each data is a dictionary of the results
            # the dictionary has keys which are the metrics and the values are lists recorded over time 
        all_data.append(data) 

    # reorganize data to aggregate over repeats and sequences
    all_data = {key: [data[key] for data in all_data] for key in all_data[0].keys()}

    return all_data

def helper_mean_smoothing(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def mean_smoothing(x, N):
    # add back missing entries so the length is the same as the original
    # we avg the starting entries
    start_means = [np.mean(x[:i]) for i in range(1, N)]
    other_means = helper_mean_smoothing(x, N)

    return np.concatenate([start_means, other_means])

def iqm(data, axis=0):
    return scipy.stats.trim_mean(data, 0.25, axis)

def iqm_error_bars(data, axis=0):
    lower_curve, upper_curve = scipy.stats.mstats.trimmed_mean_ci(data, limits=(0.25, 0.25), axis=axis)
    return lower_curve, upper_curve 


def plot_learning_curves(load_path, algorithms_used_list, env, num_task_sequences=None, num_repeats=None, save_freq=None, running_mean_window=0, plot_save_path=None):

    """ This function plots the learning curves
    algorithms_used_list: list of algorithms used in the experiments e.g. ['base', 'parseval']
    load_path: path to load the data
    env: environment used in the experiments e.g. 'metaworld' 
    num_task_sequences: number of task sequences used in the experiments. If None, sets to default.
    num_repeats: number of repeats used in the experiments. If None, sets to default.
    save_path: path to save the plot
    
    --- Plotting arguments
    save_freq: The save/eval frequency of the experiment. Uses this to generate the x-axis of the plot. If None, uses the defaults.
    change_freq: How often the environment changes. If None, uses the defaults.
    running_mean_window: How large of a window to average the curves for smoothing. If 0, no smoothing is done.
    """
    
    default_plot_params = {"alpha": 0.7, 'linewidth': 1.0}  # label is the algorithm
    SHADED_ALPHA = 0.3   # for error region

    # Metric to plot
    if env in ('metaworld', 'gridworld'):
        metric = "mean_eval_success"  # 
    elif env in ('carl_dmcquadruped', 'carl_lunarlander'):
        metric = "mean_eval_return"
    else:
        raise AssertionError('Invalid env', env)

    ### default plot parameters
    if save_freq is None:
        if env in ('metaworld', 'carl_lunarlander', 'carl_dmcquadruped'):
            save_freq = 25000
        elif env == 'gridworld':
            save_freq = 5000
    if num_repeats is None:
        if env in ('metaworld', 'carl_lunarlander', 'carl_dmcquadruped'):
            num_repeats = 3
        elif env == 'gridworld':
            num_repeats = 20
    if num_task_sequences is None:
        if env in ('metaworld', 'carl_lunarlander', 'carl_dmcquadruped'):
            num_task_sequences = 20
        elif env == 'gridworld':
            num_task_sequences = 1  # this doesn't really matter for gridworld since it is ignored
    ###


    all_curve_data = {}
    # load the relevant data
    for algorithm in algorithms_used_list:
        curve_data = load_results_for_one_alg_and_env(algorithm, env, num_repeats=num_repeats, num_task_sequences=num_task_sequences, base_folder=load_path)
        all_curve_data[algorithm] = curve_data[metric]

    for alg in all_curve_data:
        curve_data = all_curve_data[alg]

        mean_curve = iqm(curve_data, axis=0)
        low_curve, high_curve = iqm_error_bars(curve_data, axis=0)


        xs = save_freq * np.arange(len(mean_curve))
        
        if running_mean_window > 0:
            mean_curve = mean_smoothing(mean_curve, running_mean_window)
            low_curve = mean_smoothing(low_curve, running_mean_window)
            high_curve = mean_smoothing(high_curve, running_mean_window)

        plots = plt.plot(xs, mean_curve, label=alg, **default_plot_params)
        plt.fill_between(xs, low_curve, high_curve, color=plots[0].get_color(), alpha=SHADED_ALPHA)

        if metric == 'mean_eval_success':
            plt.ylim(-0.01, 1.01)
        plt.legend()
        plt.title(f"{env}")
        plt.xlabel("Timesteps")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)

    plt.show()
    plt.savefig(plot_save_path + f'{env}_learning_curves.png')
    return 


def plot_performance_profile(load_path, algorithms_used_list, env, save_freq=None, change_freq=None, num_steps=None, plot_save_path=None):
    SHADED_ALPHA = 0.3
    default_plot_params = {"alpha": 0.7, 'linewidth': 1.0}  # label is the algorithm

    ## defaults
    if num_steps is None:
        if env == 'metaworld':
            num_steps = 1e7
            change_freq = 1e6
        elif env == 'carl_dmcquadruped':
            num_steps = 12e6
            change_freq = 1.5e6
        elif env == 'carl_lunarlander':
            num_steps = 1e7
            change_freq = 500000
        elif env == 'gridworld':
            num_steps = 800000
            change_freq = 40000
        else:
            raise AssertionError('Invalid env', env)
    
    num_tasks = int(num_steps // change_freq)  # we assume num_steps is a multiple of change_freq
    
    if env in ('metaworld', 'gridworld'):
        metric = "mean_eval_success"  # 
    elif env in ('carl_dmcquadruped', 'carl_lunarlander'):
        metric = "mean_eval_return"
    else:
        raise AssertionError('Invalid env', env)
    

    # load the relevant data
    all_grouped_data = {}
    for algorithm in algorithms_used_list:
        curve_data = load_results_for_one_alg_and_env(algorithm, env, num_repeats=num_repeats, num_task_sequences=num_task_sequences, base_folder=load_path)
        all_grouped_data[algorithm] = np.array(_group_into_tasks(curve_data[metric], num_tasks)).flatten()  # flatten to group runs

    # plot the performance profile
    for algorithm in algorithms_used_list:
        a = all_grouped_data[algorithm]

        x = np.sort(a)
        y = 1 - np.arange(len(x)) / float(len(x))
        plots = plt.plot(x, y, label=env, **default_plot_params)

        ## try addingconfidence band using DKW
        confidence_level = 0.1
        err = np.sqrt(1 / (2 * len(a)) * np.log(2 / 0.1))
        y_lower = np.clip(y - err, 0, 1)
        y_upper = np.clip(y + err, 0, 1)

        ## pointwise confidence band using binomial proportion with normal approx
        #         err = np.sqrt(y * (1-y)) * 1.96 / np.sqrt(len(a))
        #         y_lower = np.clip(y - err, 0, 1)
        #         y_upper = np.clip(y + err, 0, 1)

        plt.fill_between(x, y_lower, y_upper, color=plots[0].get_color(), alpha=SHADED_ALPHA)

    plt.title("Performance Profile for " + env)
    plt.grid(alpha=0.3)

    plt.legend()
    if metric == 'mean_eval_success':
        plt.xlabel("Average Success Rate")
        plt.ylabel("Pr(Success Rate > x)")
    elif metric == 'mean_eval_return':
        plt.xlabel("Average Return")
        plt.ylabel("Pr(Return > x)")

    plt.show()
    plt.savefig(plot_save_path + f'{env}_performance_profile.png')

    return 

def _segment_averages(float_list, num_segments):
    arr = np.array(float_list)

    n = len(arr)
    segment_size = n // num_segments

    def segment_summary(segment_data):
        # mean on last 50% of each task
        summary = np.mean(segment_data[int(0.5 * len(segment_data)):])
        return summary

    averages = [segment_summary(arr[i * segment_size:(i + 1) * segment_size]) for i in range(num_segments)]
    return averages

def _group_into_tasks(data, num_tasks):
    """ Helper function. Groups the data by tasks.
    data: list of data points as obtained by load_results_for_one_alg_and_env for one metric
    num_group_segments: number of segments to group the data into. If None, uses the default settings. It's equal to total_num_steps / change_freq """
    
    grouped_data = []
    for i in range(len(data)):
        one_curve = data[i]
        one_curve = _segment_averages(one_curve, num_tasks)
        grouped_data.append(one_curve)

    return grouped_data


if __name__ == '__main__':
    test_run = True  # if you want to plot for a test run, set this to true
    if test_run:
        num_task_sequences = 1
        num_repeats = 1
        algorithms_used_list = ['base', 'parseval']
        save_freq = 2000
    
    else:
        algorithms_used_list = ['base', 'parseval']
        env = 'metaworld'
        # use defautls
        num_task_sequences = None
        num_repeats = None
        save_freq = None

    plot_learning_curves('results/', ['base', 'parseval'], env,
                          num_repeats=num_repeats, num_task_sequences=num_task_sequences, save_freq=save_freq,
                          plot_save_path='')
    # plot_performance_profile('results/', ['base', 'parseval'], 'metaworld', 
    #                          change_freq=2e3, num_steps=10e3, 
    #                          plot_save_path='')



    quit()
    