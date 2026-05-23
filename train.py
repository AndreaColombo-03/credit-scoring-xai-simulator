import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

def train_and_evaluate_model(data_path='credit_risk_clean.csv'):
    """
    Carica i dati puliti, gestisce lo sbilanciamento, addestra XGBoost e salva gli artefatti.
    """
    print("1. Caricamento dati...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Impossibile trovare {data_path}. Esegui prima lo script di preparazione dati.")

    # Separazione feature (X) e target (y)
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    print(f"Dimensioni X: {X.shape}, Dimensioni y: {y.shape}")

    # 2. Train-Test Split
    # Fondamentale l'uso di stratify=y: assicura che l'1.69% di default sia mantenuto 
    # proporzionalmente sia nel train set che nel test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("2. Preprocessing (Standardization)...")
    # 3. Standardizzazione (Scaling)
    # Lo scaler viene "fittato" SOLO sul train set per evitare Data Leakage dal test set
    scaler = StandardScaler()
    
    # Manteniamo i dataframe pandas per non perdere i nomi delle colonne, che ci serviranno per SHAP
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # 4. Calcolo del peso per la classe sbilanciata
    # Calcoliamo quanti Target=0 ci sono per ogni Target=1 nel train set
    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train == 1).sum()
    scale_weight = neg_class_count / pos_class_count
    
    print(f"3. Addestramento XGBoost (scale_pos_weight = {scale_weight:.2f})...")

    # 5. Definizione e Addestramento del Modello XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,             
        learning_rate=0.05,           
        max_depth=4,                  
        scale_pos_weight=scale_weight, 
        base_score=0.49,               
        eval_metric='auc',            
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    print("4. Valutazione del Modello...\n")
    # 6. Valutazione
    # predict_proba restituisce un array con due colonne: [prob_classe_0, prob_classe_1]. Prendiamo la 1.
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # Metrica principale per il credit scoring: ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"📊 ROC-AUC Score: {roc_auc:.4f}")
    
    # Stampiamo la Confusion Matrix e il Classification Report
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Serializzazione e Salvataggio (Export per Streamlit)
    print("\n5. Salvataggio artefatti...")
    joblib.dump(scaler, 'scaler.pkl')
    model.save_model('xgboost_model.json')
    
    # Salviamo anche i nomi delle feature per la UI di Streamlit (utile per l'ordinamento)
    joblib.dump(list(X.columns), 'feature_names.pkl')
    print("✅ Salvataggio completato: 'scaler.pkl', 'xgboost_model.json', 'feature_names.pkl'")

if __name__ == "__main__":
    train_and_evaluate_model()
    