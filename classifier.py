import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

from main import build_vocabulary, process_pipeline
from normalizer import remove_comments_and_docstrings, normalize_whitespace, rename_identifiers

def train_and_evaluate_detailed(df: pd.DataFrame, model_type='random_forest'):
    if df.empty: return None
    
    X = np.stack(df['features'].values)
    y = df['author'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = time.time()
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = SVC(kernel='rbf', C=10.0, probability=True)

    model.fit(X_train, y_train)
    end_time = time.time()
    
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    # Výpočet Top-3 Accuracy
    top3_acc = 0
    for i in range(len(y_test)):
        top3_indices = np.argsort(probs[i])[-3:]
        if np.where(model.classes_ == y_test[i])[0] in top3_indices:
            top3_acc += 1
    top3_acc /= len(y_test)

    return {
        'acc': accuracy_score(y_test, preds),
        'top3': top3_acc,
        'features': X.shape[1],
        'time': end_time - start_time
    }

def run_full_experiment_suite(code_df):
    results = []
    exps = [
        ('Scenár A (Baseline)', 'lexical', [normalize_whitespace]),
        ('Scenár B (Lexikálna)', 'lexical', [remove_comments_and_docstrings, normalize_whitespace]),
        ('Scenár C (Syntaktická)', 'syntactic', [remove_comments_and_docstrings, normalize_whitespace, rename_identifiers])
    ]
    
    for name, f_type, norms in exps:
        print(f"Spracúvam: {name}...")
        vocab = build_vocabulary(code_df, f_type, norms)
        proc_df = process_pipeline(code_df, vocab, f_type, norms)
        rf = train_and_evaluate_detailed(proc_df, 'random_forest')
        svm = train_and_evaluate_detailed(proc_df, 'svm')
        
        results.append({
            'Scenár': name, 'Atribúty': rf['features'],
            'RF Acc': rf['acc'], 'RF Top-3': rf['top3'],
            'SVM Acc': svm['acc'], 'Čas': rf['time'] + svm['time']
        })

    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f"{'Tabuľka ':^80}")
    print("="*80)
    display_df = res_df.copy()
    for col in ['RF Acc', 'RF Top-3', 'SVM Acc']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
    display_df['Čas'] = display_df['Čas'].apply(lambda x: f"{x:.3f}s")
    print(display_df.to_string(index=False))
    return res_df