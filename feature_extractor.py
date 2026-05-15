import ast
from collections import Counter
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def lexical_features(source_code: str, max_features=5000):
    return TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=max_features, min_df=2)

def syntactic_features(source_code: str):
    """
    Extrahuje syntaktické príznaky. Ak AST zlyhá, použije regexovú zálohu,
    aby sme v bakalárke nemali nulové výsledky.
    """
    features = Counter()
    
    # 1. Pokus o AST analýzu
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            features[type(node).__name__] += 1
    except Exception:
        # 2. Záložná analýza (štrukturálne prvky), ak je kód syntakticky neplatný
        keywords = ['def', 'class', 'if', 'else', 'for', 'while', 'return', 'import', 'with', 'try']
        for kw in keywords:
            features[f"kw_{kw}"] = len(re.findall(r'\b' + kw + r'\b', source_code))
        
        # Počet operátorov a zátvoriek
        features['op_assign'] = source_code.count('=')
        features['bracket_open'] = source_code.count('(')
        features['indent_count'] = source_code.count('    ')

    # Ak je stále prázdne, pridáme aspoň dĺžku kódu ako príznak
    if not features:
        features['code_length_bucket'] = len(source_code) // 100

    return dict(features)