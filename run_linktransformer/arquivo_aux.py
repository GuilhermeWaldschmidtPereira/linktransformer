import pandas as pd
import numpy as np
import os

vetor_base = np.load('../data/embeddings_base.npy')
print(vetor_base.shape)

vetor_query = np.load('../data/embeddings_query.npy')
print(vetor_query.shape)