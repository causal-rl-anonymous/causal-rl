# Causal Reinforcement Learning using Observational and Interventional Data

## Requirements

You must have Python 3 and the following packages installed:
```shell
pip install pytorch
pip install gym
pip install matplotlib
```

# Toy problem 1 (Bandit)

To run all experiments execute the following commands:
```shell
GPU = 0  # -1 for CPU

for EXPERT_SCENARIO in noisy_good perfect_good perfect_bad random strong_bad_bias strong_good_bias; do
  for SEED in {0..9}; do
    python experiments/toy1/01_train_models.py $EXPERT_SCENARIO -s $SEED -g $GPU
  done
  for SEED in {0..9}; do
    python experiments/toy1/02_evaluate_models.py $EXPERT_SCENARIO -s $SEED -g $GPU
  done
  python experiments/toy1/03_plots.py $EXPERT_SCENARIO
done
```

# Toy problem 2 (POMDP)

To run all experiments execute the following commands:
```shell
GPU = 0  # -1 for CPU

for EXPERT_SCENARIO in noisy_good very_good very_bad random strong_bad_bias strong_good_bias; do
  for SEED in {0..9}; do
    python experiments/toy1/01_train_models.py $EXPERT_SCENARIO -s $SEED -g $GPU
  done
  for SEED in {0..9}; do
    python experiments/toy1/02_evaluate_models.py $EXPERT_SCENARIO -s $SEED -g $GPU
  done
  python experiments/toy1/03_plots.py $EXPERT_SCENARIO
done
```

Note that for toy problem 2 experiments might run faster on a CPU.
