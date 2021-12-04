## Multiwoz data

source: https://www.repository.cam.ac.uk/handle/1810/280608

Sampled NLG data: `annotated_user_utts_18k.txt`

Sampled NLU data: `annoted_bio.txt`

# Processing dataset to train SUMBT_LaRL

```python
	python split_dataset.py
	python construct_ontology.py --output_dir $output_dir
```
