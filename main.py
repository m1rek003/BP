from normalizer import remove_comments_and_docstrings, normalize_whitespace, rename_identifiers
from feature_extractor import lexical_features, syntactic_features
import pandas as pd
import numpy as np

def create_dataframe(dataset):
    data = []
    for author, snippets in dataset.items():
        for snippet in snippets:
            data.append({'author': author, 'code': snippet})
    return pd.DataFrame(data)

def build_vocabulary(df, feature_type='lexical', normalizations=None):
    codes = df['code'].copy()
    if normalizations:
        for norm_func in normalizations:
            codes = codes.apply(lambda x: norm_func(x))

    if feature_type == 'lexical':
        vectorizer = lexical_features("")
        vectorizer.fit(codes)
        return vectorizer
    else:
        all_nodes = set()
        for code in codes:
            all_nodes.update(syntactic_features(code).keys())
        return sorted(list(all_nodes))

def process_pipeline(df, vocabulary, feature_type='lexical', normalizations=None):
    df = df.copy()
    if normalizations:
        for norm_func in normalizations:
            df['code'] = df['code'].apply(lambda x: norm_func(x))
    
    if feature_type == 'lexical':
        df['features'] = list(vocabulary.transform(df['code']).toarray())
    else:
        def get_vec(c):
            f = syntactic_features(c)
            return np.array([f.get(n, 0) for n in vocabulary])
        df['features'] = df['code'].apply(get_vec)
    return df