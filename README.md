
# SUMBT+LaRL

This is PyTorch implementation for [SUMBT+LaRL: Effective Multi-Domain End-to-End Neural Task-Oriented Dialog System](https://ieeexplore.ieee.org/document/9514885),
Hwaran Lee, Seokhwan Jo, Hyungjun Kim, Sangkeun Jung, and Tae-Yoon Kim,
IEEE Access, 2021

## Installation

This code is implemented upon [ConvLab][1]. 
Specifically, it requires Python 3.6, CUDA 10.1, PyTorch 1.0+, and HuggingFace 0.6.1.

For convenience, Build and Run the [Dockerfile](Dockerfile) to get the environment for this implementation.
Note that when you run the docker file, mount this repository to access the data and codes.

## Data preprocessing
1. unzip followings:
   - `data/multiwoz/annotation/annotated_user_da_with_span_full_patchName_convai.json.zip`
   - `data/multiwoz/annotation/MULTIWOZ2.zip`

2. run preprocessing codes:
	```python
    cd data/multiwoz;
	python split_dataset.py;
	python construct_ontology.py --output_dir $output_dir;
	```
	Or, just unzip followings:
   - `data/multiwoz/sumbt_larl.zip`
   - `data/multiwoz/{train, val, test}.json.zip`

## Training

The main command is following:

```bash
python convlab/modules/e2e/multiwoz/SUMBT_LaRL/main.py --do_train \
--data_dir $data_dir --task_name $task_name \
--output_dir $output_dir --delex_data_dir $delex_data_dir
```

For more details, see running scripts in `convlab/modules/e2e/multiwoz/SUMBT_LaRL/scripts` for pretraining [SUMBT][2] and [LaRL][3] modules, [end-to-end fine-tuning][4], and then [RL training][5].

## Evaluation with Simulators in ConvLab
You can run the trained end-to-end models with the simulator in ConvLab with the command.

```bash
$ python run.py {spec file} {spec name} {mode}
```

For example, you can download our trained models from [Google Drive](https://drive.google.com/file/d/1NnhkQFFepxb87DZNiTiI8GG5-N6yHtQ2/view?usp=sharing) and unzip the a model inside `model` directory, then run the command.
```bash
$ python run.py convlab/spec/sumbt_larl.json sumbt_larl eval
```



## Citing
If you use this code in your research, please cite following:
```text
@article{lee2021sumbt+,
  title={SUMBT+ LaRL: Effective Multi-Domain End-to-End Neural Task-Oriented Dialog System},
  author={Lee, Hwaran and Jo, Seokhwan and Kim, Hyungjun and Jung, Sangkeun and Kim, Tae-Yoon},
  journal={IEEE Access},
  volume={9},
  pages={116133--116146},
  year={2021},
  publisher={IEEE}
}
```

[1]: https://github.com/ConvLab/ConvLab
[2]: convlab/modules/e2e/multiwoz/SUMBT_LaRL/scripts/exp_sumbt_larl_ptr_sumbt.pl
[3]: convlab/modules/e2e/multiwoz/SUMBT_LaRL/scripts/exp_sumbt_larl_ptr_larl.pl
[4]: convlab/modules/e2e/multiwoz/SUMBT_LaRL/scripts/exp_sumbt_larl_finetune.pl
[5]: convlab/modules/e2e/multiwoz/SUMBT_LaRL/scripts/exp_sumbt_larl_rl.pl
