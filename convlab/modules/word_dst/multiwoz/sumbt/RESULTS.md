
# Unified-SUMBT(SUMBT*)
- [Unified-SUMBT(SUMBT*)](#unified-sumbtsumbt)
  - [Architecture](#architecture)
  - [Hyper-parameter Serach](#hyper-parameter-serach)
    - [Dataset](#dataset)
    - [Request Classifier](#request-classifier)
      - [Results on the Test Set](#results-on-the-test-set)
    - [Reccurrent Request Classifier](#reccurrent-request-classifier)
      - [Results on the Test Set](#results-on-the-test-set-1)
  - [Hyper-parameter Serach: MultiWOZ 2 .1](#hyper-parameter-serach-multiwoz-2-1)
    - [Dataset](#dataset-1)
    - [Reccurrent Request Classifier](#reccurrent-request-classifier-1)
      - [Results on the Test Set](#results-on-the-test-set-2)
      - [Results on the Test Set (Max sequence length| 128)](#results-on-the-test-set-max-sequence-length-128)
    - [Reccurrent Request Classifier (lowercased)](#reccurrent-request-classifier-lowercased)
      - [Results on the Test Set](#results-on-the-test-set-3)

## Architecture

![Unified SUMBT](SUMBT_unified.001.png)
* Multi-task learning of beleif state tracking, user general act classifier, and user request classifiers.
* $\mathcal{L}=\mathcal{L}_{bf}+\lambda_{r}\mathcal{L}_{r}+\lambda_{g}\mathcal{L}_{g}$

* $\mathcal{L}_{r}=\sum_\mathcal{D}\sum_t\sum_s [\alpha_r r^s_t\log(\hat{r}^s_t) + (1-r^s_t)\log(\hat{r}^s_t)]$
  * $\alpha_r$ : positive weight

* $\mathcal{L}_{g}=\sum_\mathcal{D}\sum_t\sum_{a\in \{greet, bye, thank\}}  [\alpha_g g_t^a\log(\hat{g}^a_t) + (1-g^a_t)\log(\hat{g}^a_t)]$
  * $\alpha_g$ : positive weight


## Hyper-parameter Serach

### Dataset
|         | POS  | NEG    | (NEG/POS) Ratio |
| ------- | ---- | ------ | --------------- |
| REQUEST | 1863 | 270901 | 145.41          |
| GENERAL | 1173 | 20943  | 17.85           |

### Request Classifier
* Experiment dir = ```models-multiwoz-unified-1l/task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=$lambda-rpw=$request_positive_weight```
* lambda ($\lambda_{r}=\lambda_{g}$)= {1, 2, 5}
* request_positive_weight ($\alpha_r$) = {10, 50, 100, 200}
* general_positive_weight ($\alpha_g$) = q
* batchsize = 3
* hidden_size = 300

#### Results on the Test Set
|                                                         | $\lambda_{r}=\lambda_{g}$ | $\alpha_r$ | Belief State Joint Accuracy | Belief State Slot acc | REQ Accuracy | REQ Precision | GEN Accuracy | GEN Precsion |
| ------------------------------------------------------- | ------------------------- | ---------- | ---------------------- | --------------------- | ------- | ------------- | ------- | ------------ |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=10      | 1                         | 10         | **0.49145**            | 0.97453               | 0.99435 | 0.92002       | 0.99679 | 0.96505      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=50      | 1                         | 50         | **0.49417**            | 0.97378               | 0.99474 | 0.94579       | 0.99774 | 0.97698      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=100     | 1                         | 100        | 0.48684                | 0.97445               | 0.99251 | 0.96672       | 0.99774 | 0.97783      |
| **task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=200** | 1                         | 200        | **0.49078**            | 0.97473               | 0.98347 | **0.97155**   | 0.99769 | 0.96846      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=10      | 2                         | 10         | 0.48209                | 0.97425               | 0.99671 | 0.93076       | 0.99806 | 0.97442      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=50      | 2                         | 50         | **0.49376**            | 0.97471               | 0.99320 | 0.96457       | 0.99792 | 0.97613      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=100     | 2                         | 100        | 0.48277                | 0.97357               | 0.99264 | 0.96833       | 0.99783 | 0.97442      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=200     | 2                         | 200        | 0.48521                | 0.97383               | 0.98681 | **0.97424**   | 0.99742 | 0.97187      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=5-rpw=10      | 5                         | 10         | 0.48766                | 0.97458               | 0.99573 | 0.93612       | 0.99774 | 0.97869      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=5-rpw=50      | 5                         | 50         | 0.47789                | 0.97372               | 0.99494 | 0.95706       | 0.99797 | **0.98380**  |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=5-rpw=100     | 5                         | 100        | 0.47735                | 0.97377               | 0.98916 | **0.97692**   | 0.99833 | 0.97783      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=5-rpw=200     | 5                         | 200        | 0.48426                | 0.97483               | 0.99450 | 0.95706       | 0.99815 | **0.98380**  |

![Tensorboard](exp_tensorboard.png)

### Reccurrent Request Classifier
* the above trained model is vulerable when the context does not show domain information
  * System utterance: The phone number is 01223295264 .
  * User utterance: Can i please have the postcode ?
  * Inferred user action: {'Attraction-Request': [['Post', '?']], 'Hospital-Request': [['Post', '?']], 'Hotel-Request': [['Post', '?']], 'Police-Request': [['Post', '?']], 'Restaurant-Request': [['Post', '?']]}
* Therefore, we added a RNN layer on top of the attention layer likewise belief state tracker.

#### Results on the Test Set

|                                                         | $\lambda_{r}=\lambda_{g}$ | $\alpha_r$ | Belief State Joint Accuracy | Belief State Slot acc | REQ Accuracy | REQ Precision | GEN Accuracy | GEN Precsion |
| ------------------------------------------------------- | ------------------------- | ---------- | ---------------------- | --------------------- | ------- | ------------- | ------- | ------------ |
| **task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=50**      | 1                         | 50         |**0.49390**                | 0.97455               | **0.99124** | 0.96887       | 0.99751 | 0.97357      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=100     | 1                         | 100        | 0.48426                | 0.97422               | 0.98363 | 0.97531       | 0.99661 | 0.97101      |
| **task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=200** | 1                         | 200        | 0.48779                | 0.97490               | 0.98157 | **0.99141**       | 0.99783 | 0.97698      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=50      | 2                         | 50         | 0.48521                | 0.97384               | **0.99327** | 0.96994       | 0.99806 | 0.97698      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=100     | 2                         | 100        | 0.48698                | 0.97478               | 0.98696 | 0.97960       | 0.99756 | 0.97101      |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=200     | 2                         | 200        | 0.49186                | **0.97563**               | 0.98976 | 0.97960       | 0.99792 | 0.97869      |


## Hyper-parameter Serach: MultiWOZ 2 .1

### Dataset
|         | POS  | NEG    | (NEG/POS) Ratio |
| ------- | ---- | ------ | --------------- |
| REQUEST | 1863 | 270901 | 145.41          |
| GENERAL | 1173 | 20943  | 17.85           |

### Reccurrent Request Classifier
* Experiment dir = ```models-multiwoz-unified/task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=$lambda-rpw=$request_positive_weight```
* lambda ($\lambda_{r}=\lambda_{g}$)= {1, 2}
* request_positive_weight ($\alpha_r$) = {50, 100, 200}
* general_positive_weight ($\alpha_g$) = q
* batchsize = 3
* hidden_size = 300

#### Results on the Test Set
|                                                         | $\lambda_{r}=\lambda_{g}$ | $\alpha_r$ | Belief State Joint Accuracy | Belief State Slot acc | REQ Accuracy | REQ Precision | GEN Accuracy | GEN Precsion |
| ------------------------------------------------------- | ------------------------- | ---------- | ---------------------- | --------------------- | ------- | ------------- | ------- | ------------ |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=50  | 1                         | 50         | 0.47382 | 0.97410 | 0.99613 | 0.95974 | 0.99751 | 0.97101 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=100     | 1                         | 100        | 0.49390 | 0.96956 | 0.98558 | 0.98551 | 0.99801 | 0.97016|
| **task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=200** | 1                         | 200        | 0.49471 | 0.97509 | 0.98879 | 0.98443 | 0.99810 | 0.97698 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=50      | 2                         | 50         | 0.48101 | 0.97422 | 0.99602 | 0.97692 | 0.99801 | 0.97528 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=100     | 2                         | 100        | 0.47897 | 0.97424 | 0.99135 | 0.97048 | 0.99787 | 0.97101 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=200     | 2                         | 200        | 0.48983 | 0.97486 | 0.99043 | 0.97638 | 0.99810 | 0.97442 |

#### Results on the Test Set (Max sequence length| 128)
* Experiment dir = ```models-multiwoz-unified-len128/task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=$lambda-rpw=$request_positive_weight```

|                                                         | $\lambda_{r}=\lambda_{g}$ | $\alpha_r$ | Belief State Joint Accuracy | Belief State Slot acc | REQ Accuracy | REQ Precision | GEN Accuracy | GEN Precsion |
| ------------------------------------------------------- | ------------------------- | ---------- | ---------------------- | --------------------- | ------- | ------------- | ------- | ------------ |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=50  | 1                         | 50         | 0.47721 | 0.97442 | 0.99541 | 0.97316 | 0.99824 | 0.97272 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=100     | 1                         | 100        | 0.47911 | 0.97473 | 0.99343 | 0.97853 | 0.99815 | 0.97698 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=1-rpw=200 | 1                         | 200        | 
| **task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=50**     | 2                         | 50         | 0.49322 | 0.97515 | 0.99537 | 0.97853 | 0.99806 | 0.96846 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=100     | 2                         | 100        | 0.47897 | 0.97447 | 0.98803 | 0.98604| 0.99828 | 0.97698 |
| task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=2-rpw=200     | 2                         | 200        |


### Reccurrent Request Classifier (lowercased)
* Experiment dir = ```models-multiwoz/task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-l=$lambda-rpw=$request_positive_weight```
* lambda ($\lambda_{r}=\lambda_{g}$)= {1}
* request_positive_weight ($\alpha_r$) = {1, 50, 100}
* general_positive_weight ($\alpha_g$) = q
* batchsize = 3
* hidden_size = 300

#### Results on the Test Set