## Multiwoz data

source: https://www.repository.cam.ac.uk/handle/1810/280608

Sampled NLG data: `annotated_user_utts_18k.txt`

Sampled NLU data: `annoted_bio.txt`

# Processing dataset to train SUMBT_LaRL
1. unzip followings:
   - `data/multiwoz/annotation/annotated_user_da_with_span_full_patchName_convai.json.zip`
   - `data/multiwoz/annotation/MULTIWOZ2.zip`

2. run preprocessing codes:
	```python
	python split_dataset.py
	python construct_ontology.py --output_dir $output_dir
	```
	Or, just unzip followings:
   - `data/multiwoz/sumbt_larl.zip`
   - `data/multiwoz/{train, val, test}.json.zip`