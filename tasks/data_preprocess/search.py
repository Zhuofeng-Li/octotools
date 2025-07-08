import datasets
from typing import Dict, Any
import argparse
import random
import os

# Set a seed for reproducible results
random.seed(42)

# add a row to each data item that represents a unique id
def make_map_fn():
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        example['question'] = example['question'].strip()
        if example['question'][-1] != '?':
            example['question'] += '?'

        if len(example['golden_answers']) > 1:
            print("===== ERROR =========")
            
        data = {
            "query": example['question'],
            "image": None,
            "answer": example['golden_answers'][0]
        }
        return data

    return process_fn

# set a seed for the random number generator    
def random_sample(dataset: datasets.Dataset, sample_size: int) -> datasets.Dataset:
    if len(dataset) > sample_size:
        indices = random.sample(range(len(dataset)), sample_size)
        return dataset.select(indices)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--sample_size', type=int, default=200, help='Number of samples to randomly extract')

    args = parser.parse_args()

    # Load  dataset
    bamboogle_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'bamboogle')['test']
    musique_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'musique')['dev']
    two_wikimultihopqa_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', "2wikimultihopqa")['dev']
    popqa_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'popqa')['test']

    # process the data
    bamboogle_dataset = bamboogle_dataset.filter(lambda x: len(x['golden_answers']) == 1)
    musique_dataset = musique_dataset.filter(lambda x: len(x['golden_answers']) == 1)
    two_wikimultihopqa_dataset = two_wikimultihopqa_dataset.filter(lambda x: len(x['golden_answers']) == 1)
    popqa_dataset = popqa_dataset.filter(lambda x: len(x['golden_answers']) == 1)

    bamboogle_dataset = bamboogle_dataset.map(make_map_fn(), with_indices=True, remove_columns=bamboogle_dataset.column_names)
    musique_dataset = musique_dataset.map(make_map_fn(), with_indices=True, remove_columns=musique_dataset.column_names)
    two_wikimultihopqa_dataset = two_wikimultihopqa_dataset.map(make_map_fn(), with_indices=True, remove_columns=two_wikimultihopqa_dataset.column_names)
    popqa_dataset = popqa_dataset.map(make_map_fn(), with_indices=True, remove_columns=popqa_dataset.column_names)

    # random sample the data
    bamboogle_dataset = random_sample(bamboogle_dataset, args.sample_size)
    musique_dataset = random_sample(musique_dataset, args.sample_size)
    two_wikimultihopqa_dataset = random_sample(two_wikimultihopqa_dataset, args.sample_size)
    popqa_dataset = random_sample(popqa_dataset, args.sample_size)

    # add a row to each data item that represents a unique id
    bamboogle_dataset = bamboogle_dataset.map(lambda x, idx: {'pid': int(idx)}, with_indices=True)
    musique_dataset = musique_dataset.map(lambda x, idx: {'pid': int(idx)}, with_indices=True)
    two_wikimultihopqa_dataset = two_wikimultihopqa_dataset.map(lambda x, idx: {'pid': int(idx)}, with_indices=True)
    popqa_dataset = popqa_dataset.map(lambda x, idx: {'pid': int(idx)}, with_indices=True)


    os.makedirs(args.local_dir, exist_ok=True)


    # Create subdirectories for each dataset
    bamboogle_dir = os.path.join(args.local_dir, 'bamboogle', 'data')
    musique_dir = os.path.join(args.local_dir, 'musique', 'data')
    wikimultihopqa_dir = os.path.join(args.local_dir, '2wikimultihopqa', 'data')
    popqa_dir = os.path.join(args.local_dir, 'popqa', 'data')
    
    os.makedirs(bamboogle_dir, exist_ok=True)
    os.makedirs(musique_dir, exist_ok=True)
    os.makedirs(wikimultihopqa_dir, exist_ok=True)
    os.makedirs(popqa_dir, exist_ok=True)

    # save the data to the local file
    # bamboogle_dataset.to_json(os.path.join(bamboogle_dir, 'data.json'), orient='records', lines=False, indent=4)
    musique_dataset.to_json(os.path.join(musique_dir, 'data.json'), orient='records', lines=False, indent=4)
    two_wikimultihopqa_dataset.to_json(os.path.join(wikimultihopqa_dir, 'data.json'), orient='records', lines=False, indent=4)
    popqa_dataset.to_json(os.path.join(popqa_dir, 'data.json'), orient='records', lines=False, indent=4)

"""
python tasks/data_preprocess/search.py --local_dir ./tasks --sample_size 200
"""


