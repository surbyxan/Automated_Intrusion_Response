# File structure

This repository consist of 4 directories. The _cage-v2_ consist of necesary components for the custom scenario. The _Evaluation_ directory includes the **evaluation.py** script that evaluates all final agents. The **Argus** agents is found in the _Heuristic Agents_ directory, and everything related to training and tuning the reinforcement learning agents is found in the _RL Agens_ directory.

- For files and scripts referenced in the thesis report follow the file tree below

```
|
├── cage-v2/
├── Evaluation/
|	├── evaluation.py
|	├──	TrueStateBlueTableWrapper.py
|	└──	TrueStateChallengeWrapper.py
|
├── Heuristic Agents/
|	└── # All Argus agents V0-XI
|
├── RL Agents/
|	├── Optuna/
|	|	├── optuna_dql.py
|	|	└──	optuna_ppo.py.py
|	|
|	├── dql_test_agent.py
|	├── ppo_test_agent.py
|	├── RL_dql_more_meander.py
|	├── RL_dql.py
|	├── RL_ppo_more_meander.py
|	└──	RL_ppo.py.py
|
├── find_latest_model.py
└── README.md

```

## Imports and requirements

_The code utilises the CybORG environment with the CAGE-2 sceanrio, OpenAI's gymnasium, stable baselines 3, and need these imported to run the code._

[CAGE-2 Challenge Repo](https://github.com/cage-challenge/cage-challenge-2)
