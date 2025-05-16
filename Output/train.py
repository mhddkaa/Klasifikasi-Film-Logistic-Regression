import pandas as pd
import numpy as np
import joblib
from model import MulticlassLogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

# Label Encoding
def save_label_encoder():
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['Horor', 'Drama', 'Komedi', 'Laga', 'Romantis'])
    
    joblib.dump(label_encoder, 'label_encoder.joblib')
    print("Label Encoder Sukses Tersimpan")
    return label_encoder

# TF-IDF 
def process_and_save_tfidf():
    # Load data
    datafile1 = pd.read_csv('hasil_labeling_train.csv')
    datafile2 = pd.read_csv('hasil_labeling_validation.csv')
    datafile3 = pd.read_csv('hasil_labeling_test.csv')
    
    X_train_text = datafile1['stemming']
    y_train = datafile1['label']
    X_val_text = datafile2['stemming']
    y_val = datafile2['label']
    X_test_text = datafile3['stemming']
    y_test = datafile3['label']
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        min_df=2,
        max_df=0.8,
        ngram_range=(1,2)
    )
    X_train = tfidf.fit_transform(X_train_text)
    X_val = tfidf.transform(X_val_text)
    X_test = tfidf.transform(X_test_text)
    
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    print("TF-IDF Vectorizer Sukses Tersimpan")
    
    return tfidf, X_train, X_val, X_test, y_train, y_val, y_test

# PCA
def process_and_save_pca(X_train, X_val, X_test, y_train, y_val, y_test):
    fitur_list = [167, 300, 500]
    pca_results = {}  
    
    for n in fitur_list:
        print(f"PCA dengan {n} fitur")
        
        # PCA
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        
        joblib.dump(pca, f'pca_model_{n}.joblib')
        print(f"PCA Model dengan {n} Komponen Sukses Tersimpan")
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"Total varians yang dijelaskan dengan {n} fitur: {explained_var:.4f} atau {explained_var*100:.2f}%")
        
        train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n)])
        val_pca_df = pd.DataFrame(X_val_pca, columns=[f'PC{i+1}' for i in range(n)])
        test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(n)])
        
        train_pca_df['label'] = y_train.values
        val_pca_df['label'] = y_val.values
        test_pca_df['label'] = y_test.values
        
        train_pca_df.to_csv(f'hasil_pca_train_{n}.csv', index=False)
        val_pca_df.to_csv(f'hasil_pca_validation_{n}.csv', index=False)
        test_pca_df.to_csv(f'hasil_pca_test_{n}.csv', index=False)
        
        print(f"Sebelum PCA (train): {X_train.shape} → Setelah PCA: {X_train_pca.shape}")
        print(f"Sebelum PCA (validation): {X_val.shape} → Setelah PCA: {X_val_pca.shape}")
        print(f"Sebelum PCA (test): {X_test.shape} → Setelah PCA: {X_test_pca.shape}")
        
        pca_results[n] = (X_train_pca, X_val_pca, X_test_pca)
    
    return pca_results

# Train dan Save Logistic Regression Model
def train_and_save_models(pca_results, y_train, y_val, y_test):
    for n_components, data in pca_results.items():
        X_train_pca, X_val_pca, X_test_pca = data
        
        print(f"\nTraining Logistic Regression model dengan {n_components} PCA Komponen")
        
        # Inisialisasi dan Train Model
        model = MulticlassLogisticRegression(n_iters=1000)
        model.fit(X_train_pca, y_train, X_val_pca, y_val)
        
        # Evaluasi dengan Data Test
        test_acc = model.evaluate(X_test_pca, y_test)
        print(f"Test accuracy with {n_components} components: {test_acc:.4f}")
        
        joblib.dump(model, f'logistic_regression_model_{n_components}.joblib')
        print(f"Logistic Regression Model dengan {n_components} Komponen Sukses Tersimpan")

# Grid Search
def grid_search_and_save_model(X_train_pca, y_train, n_components):
    param_grid = {
        'learning_rate': [0.01, 0.005, 0.001],
        'batch_size': [32, 64],
        'lambda_reg': [0.1, 0.5, 1.0],
        'n_iters': [1000],
        }
    
    def generate_combinations(param_grid):
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        return [dict(zip(keys, v)) for v in product(*values)]

    param_combinations = generate_combinations(param_grid)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_score = 0
    best_model = None
    best_params = None
    
    print(f"\nGrid Search Logistic Regression dengan {n_components} Komponen PCA")
    
    for params in param_combinations:
        fold_scores = []
        for train_idx, val_idx in skf.split(X_train_pca, y_train):
            X_train_fold, X_val_fold = X_train_pca[train_idx], X_train_pca[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = MulticlassLogisticRegression(**params)
            model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            score = model.evaluate(X_val_fold, y_val_fold)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_params = params

    print(f"\nBest params untuk {n_components} komponen PCA: {best_params}")
    print(f"Best CV accuracy: {best_score:.4f}")

    joblib.dump(best_model, f'gridsearch_model_{n_components}.joblib')
    print(f"GridSearch Logistic Regression Model dengan {n_components} Komponen Sukses Tersimpan")

# Main execution
if __name__ == "__main__":
    print("Start Training")
    
    label_encoder = save_label_encoder()
    print("\nStep 1: Label encoder")
    
    tfidf, X_train, X_val, X_test, y_train, y_val, y_test = process_and_save_tfidf()
    print("\nStep 2: TF-IDF processing")
    
    pca_results = process_and_save_pca(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\nStep 3: PCA processing")
    
    train_and_save_models(pca_results, y_train, y_val, y_test)
    print("\nStep 4: Logistic Regression Model")
    
    for n_components, data in pca_results.items():
        X_train_pca, X_val_pca, X_test_pca = data
        print(f"\nGrid Search dan Training dengan {n_components} Komponen PCA")
        
        best_model, best_params, best_score = grid_search_and_save_model(X_train_pca, y_train, n_components)

    print("\nSukses!")