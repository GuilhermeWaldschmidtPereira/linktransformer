import pandas as pd
import numpy as np
import os

vetor_base = np.load('../data/embeddings_base_intfloat_multilingual-e5-large.npy')
print(vetor_base.shape)

vetor_query = np.load('../data/embeddings_query_neuralmind_bert-large-portuguese-cased.npy')
print(vetor_query.shape)