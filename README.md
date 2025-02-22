# Parseval Regularization for Continual Reinforcement Learning

Official repository for the paper "Parseval Regularization for Continual Reinforcement Learning" published at NeurIPS2024

Parseval regularization is designed to address loss of plasticity. This is demonstrated in sequences of reinforcement learning tasks.

Experiments were run using Python3.10.
Download the repo and navigate to the `parseval_reg` folder. Unstall the requirements using:
```
pip install -r reqs.txt
```

You can run the experiments using
```
python run_many.py 
```
By default, this will run the MetaWorld environments for the baseline and Parseval regularization.
For the other environments, change the `env_to_run` variable to one of: `'metaworld', 'carl_dmcquadruped', 'carl_lunarlander', 'gridworld'`

You can use the `python run_many.py --test_run` command to run a very small test to check the code works. It should finish in less than 1 minute. 

For running specific configurations, you can use the `main.py` file along with the appropriate arguments.

### Citation
```
@inproceedings{chung2024parseval,
 author = {Chung, Wesley and Cherif, Lynn and Precup, Doina and Meger, David},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {127937--127967},
 title = {Parseval Regularization for Continual Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
