# LaRL

This is [snakeztc/NeuralDialog-LaRL](https://github.com/snakeztc/NeuralDialog-LaRL) commit `6d0320d` ported to ConvLab.

| Spec              | Results                                                       |
| ----------------- | ------------------------------------------------------------- |
| milu_rule_larl-sl | 100 episodes, -18.76 return, 30.00% success rate, 16.76 turns |
| milu_rule_larl-rl | 100 episodes, -17.77 return, 31.00% success rate, 16.97 turns |

## Setup

LaRL uses `norm-multi-woz` dataset to train and stores pretrained models in `sys_config_log_model`. You can prepare them with the `setup.sh` script.

```
bash convlab/modules/word_policy/multiwoz/larl/setup.sh
```

## Training

The training for LaRL is a two-step process: SL (supervised learning) and RL (reinforcement learning).

```bash
python convlab/modules/word_policy/multiwoz/larl/experiments_woz/sl_cat.py
```

The logs and models appear in a timestamped folder inside the `sys_config_log_model` folder. (Note that this folder is ignored by git) the models are named `XX-model`. The number `XX` is the model id.

Afterwards, you can load the SL-trained model and train it with RL:

```bash
python convlab/modules/word_policy/multiwoz/larl/experiments_woz/reinforce_cat.py
```

Note that you will need to change two variables in `reinforce_cat.py` to make this work:

```python
pretrained_folder = '2019-08-20-04-17-55-sl_cat'
pretrained_model_id = 36
```

## Evaluation

For ConvLab's end-to-end evaluation, you can use `larl.json`. The `milu_rule_larl-sl` spec uses MILU NLU, Rule-based DST, and LaRL Word Policy trained with SL. The milu_rule_larl-rl` is the same except the LaRL Word Policy was trained with both SL and RL.

```bash
python run.py larl.json milu_rule_larl-sl eval
python run.py larl.json milu_rule_larl-rl eval
```

Again, note that you will need to change `pretrained_folder` and `model_path` for LaRL in the JSON file:

```
"word_policy": {
    "name": "LaRLWordPolicy",
    "pretrained_folder": "2019-08-20-04-17-55-sl_cat",
    "model_path": "sys_config_log_model/2019-08-20-04-17-55-sl_cat/rl-2019-08-23-04-06-55/reward_best.model"
},
```

## Logging

LaRL allows two additional log levels for debugging:

```
LOG_LEVEL=DELEX python run.py larl.json milu_rule_larl-sl eval
LOG_LEVEL=DB python run.py larl.json milu_rule_larl-sl eval
```

The `DELEX` level logs user and system utterance, and `DB` level also logs DB results.
