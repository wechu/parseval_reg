##### 
# Example file to run many of the experiments
#
#
#####

import subprocess
from itertools import product
import argparse


def run_many(arguments_list, test_run=False):
    """
    Generates a bash script with multiple lines calling `run.py` with different commands.
    Used to run many experiments in parallel.
    """
    bash_filename = 'run_many.sh'
    bash_script_content = "#!/bin/bash\n\n"
    
    for arg in arguments_list:
        bash_script_content += f"python minimal/main.py {arg}{' --test_run' if test_run else ''} &\n"


    with open(bash_filename, "w") as file:
        file.write(bash_script_content)

    # Make the bash script executable
    os.chmod(bash_filename, 0o755)

    try:
        subprocess.call("./" + bash_filename)
    except subprocess.CalledProcessError as e:
        print(f"Error while executing the script: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_run", action='store_true', help="Set to True to do a small run, with 1 task sequence for two algorithms and only 10k steps. Should be done within a minute.")
    parser.add_argument("--env_to_run", type=str, default='metaworld', help="Environment to run. Options: 'metaworld', 'carl_dmcquadruped', 'carl_lunarlander', 'gridworld'")
    parser.add_argument("--number_of_repeats", type=int, default=3, help="Number of repeats for each algorithm per environment sequence")
    
    args = parser.parse_args()

    algorithms_to_run = ['base', 'parseval']  # 'base', 'parseval', 'layer_norm', 'snp', 'regen', 'w-regen'
  
    num_task_sequences = 20 # there are 20 sequences of tasks in total, can run fewer to test
    # number_of_repeats = 3  # how many seeds to run for each algorithm per environment sequence (total_num of runs = number_of_repeats * num_sequences)
    # e.g. metaworld has 20 sequences. So if number_of_repeats = 3, then the total number of runs = 20 * 3 = 60
    # default number of repeats is 3 for 'metaworld', 'carl_dmcquadruped', 'carl_lunarlander', 
    # and 20 for 'gridworld' 

    if args.test_run:
        num_task_sequences = 1
        args.number_of_repeats = 1
        algorithms_to_run = ['base', 'parseval']

    if args.env_to_run == 'metaworld':
        env_names = [f'metaworld_sequence_set{i}' for i in range(num_task_sequences)]  
    elif args.env_to_run == 'carl_dmcquadruped':
        env_names = [f"carl_sequence_dmcquadruped_{i}" for i in range(num_task_sequences)]  
    elif args.env_to_run == 'carl_lunarlander':
        env_names = [f"carl_sequence_lunarlander_{i}" for i in range(num_task_sequences)]  
    elif args.env_to_run == 'gridworld':
        env_names = ['gridworld_ninerooms']  # gridworld randomizes the tasks for every run, so just use more repeats instead (e.g. 20)


    # list of all arguments to run
    arguments_list = []
    for alg, env, i_repeat in product(algorithms_to_run, env_names, range(args.number_of_repeats)):
        arguments_list.append(f"--env {env} --algorithm {alg} --repeat_idx {i_repeat}")
    
    import os
    print("Current working directory:", os.getcwd())
    
    
    print(arguments_list) 
    print("Total number of runs: ", len(arguments_list))

    run_many(arguments_list, test_run=args.test_run)
