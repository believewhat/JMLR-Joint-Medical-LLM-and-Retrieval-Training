from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher
import torch
import ipdb
import os
import json
import re
import csv
import pandas as pd
import numpy as np
import sys


if __name__ ==  '__main__':
    data_file = 'Your Json file'
    collection = './open_guideline/name.tsv'
    num_doc = 20
    with open(data_file, "r") as file:
        data = json.load(file)
    
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
    query_file = sys.argv[3]
    save_file = sys.argv[4]
    
    
    data = pd.DataFrame(data).loc[start_index:end_index]
    data.reset_index(inplace=True, drop=True)

    data['id'] = np.zeros(data.shape[0], int)
    for i in range(data.shape[0]):
        data['id'].loc[i] = i+1
        data['input'].loc[i] = data['input'].loc[i].replace('\n', ' ')
    

    data[['id', 'input']].to_csv(query_file, index=False, header=False, sep='\t')
    
    
    nbits = 2
    checkpoint_path = "your path"
    experiment_root = "./experiments/"
    experiment_name = "openguideline_medmcqa"
    index_name = f'{experiment_name}.{nbits}bits'
    
    with Run().context(RunConfig(nranks=4, root=experiment_root, experiment=experiment_name)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=512, query_maxlen=128, nbits=nbits)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    

    
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        config = ColBERTConfig(
            root="./",
        )
        searcher = Searcher(index=index_name, checkpoint=checkpoint_path, config=config)
        queries = Queries(query_file)
        
        ranking = searcher.search_all(queries, k=num_doc)
        ranks_result = ranking.tolist()
        output = []
        with open(collection, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')
            doc = [row for row in reader]
        doc_dict = {}
        for x in doc:
            doc_dict[int(x[0])] = x[1]
        for i in range(data.shape[0]):
            doc_list = []
            for x in ranks_result[i*num_doc:(i+1)*num_doc]:
                """
                if doc_dict[x[1]] in data['input'].loc[i] or data['input'].loc[i] in doc_dict[x[1]]:
                    continue
                """
                doc_list.append(doc_dict[x[1]])
            print(len(doc_list))
            output.append({'input': data['input'].loc[i], 'output': data['output'].loc[i], 'doc': doc_list})
        with open(save_file, 'w') as file:
            json.dump(output, file, indent=4)

