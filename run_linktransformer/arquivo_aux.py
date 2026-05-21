import pandas as pd
import numpy as np
import os

vetor_base = np.load('../data/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy')
print(vetor_base.shape)

vetor_query = np.load('../data/embeddings_query_sentence-transformers_all-MiniLM-L6-v2.npy')
print(vetor_query.shape)