import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import re
import xgboost   
import sklearn   

# --- FUNZIONE PURA AGGIUNTA PER I TEST AUTOMATICI ---
def calculate_risk(model, scaler, expected_features, raw_input_dict):
    """
    Funzione che calcola il rischio. Separata dalla UI per facilitare i test con pytest.
    """
    raw_input = pd.DataFrame([raw_input_dict])
    encoded_input = pd.get_dummies(raw_input)
    encoded_input.columns = [re.sub(r'[<>, ]', '_', col) for col in encoded_input.columns]
    aligned_input = encoded_input.reindex(columns=expected_features, fill_value=0)
    scaled_input = pd.DataFrame(scaler.transform(aligned_input), columns=expected_features)
    
    probability = model.predict_proba(scaled_input)[0, 1]
    return float(probability)
# ----------------------------------------------------

# 1. Configurazione della pagina
st.set_page_config(
    page_title="Credit Risk Simulator",
    page_icon="🏦",
    layout="wide"
)

# 2. Caricamento degli artefatti (In Cache per non ricaricarli ad ogni click)
@st.cache_resource
def load_models():
    model = xgboost.XGBClassifier()
    model.load_model('xgboost_model.json')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, expected_features = load_models()

# 3. Interfaccia Utente - Sidebar per l'inserimento dati
st.sidebar.title("⚙️ Parametri Richiedente")
st.sidebar.markdown("Inserisci i dati del cliente per valutare il rischio di credito.")

# Raccogliamo gli input base dell'utente (quelli prima del One-Hot Encoding)
gender = st.sidebar.selectbox("Genere", ["M", "F"])
own_car = st.sidebar.selectbox("Possiede un'auto?", ["Y", "N"])
income = st.sidebar.number_input("Reddito Annuale (€)", min_value=10000, max_value=1000000, value=45000, step=5000)
income_type = st.sidebar.selectbox("Tipo di Occupazione", ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'])
education = st.sidebar.selectbox("Livello di Istruzione", ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'])
family = st.sidebar.selectbox("Stato Civile", ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'])
age = st.sidebar.slider("Età (Anni)", 18, 90, 35)
employed_years = st.sidebar.slider("Anzianità Lavorativa (Anni)", 0, 50, 5)

st.title("🏦 Interactive Credit Scoring con XAI")
st.write("Questa applicazione calcola la probabilità di default di un prestito e utilizza la teoria dei giochi (SHAP) per spiegare quali variabili hanno influenzato la decisione dell'algoritmo.")

st.divider()

if st.button("Valuta Rischio di Credito", type="primary"):
    
    with st.spinner("Calcolo del rischio e generazione spiegazione SHAP in corso..."):
        
        # 4. Ricostruzione dinamica del DataFrame (Pipeline di Preprocessing)
        raw_input = pd.DataFrame({
            'CODE_GENDER': [gender],
            'FLAG_OWN_CAR': [own_car],
            'AMT_INCOME_TOTAL': [income],
            'NAME_INCOME_TYPE': [income_type],
            'NAME_EDUCATION_TYPE': [education],
            'NAME_FAMILY_STATUS': [family],
            'AGE_YEARS': [age],
            'EMPLOYED_YEARS': [employed_years]
        })
        
        # Applichiamo il One-Hot Encoding come nel training
        encoded_input = pd.get_dummies(raw_input)
        
        # Puliamo i nomi delle colonne per evitare crash di XGBoost
        encoded_input.columns = [re.sub(r'[<>, ]', '_', col) for col in encoded_input.columns]
        
        # Riappacifichiamo l'input dell'utente con le feature originali del training set.
        # Riempiamo con 0 le categorie che l'utente non ha selezionato in questo momento.
        aligned_input = encoded_input.reindex(columns=expected_features, fill_value=0)
        
        # Standardizzazione
        scaled_input = pd.DataFrame(scaler.transform(aligned_input), columns=expected_features)
        
        # 5. Previsione
        # Estraiamo la probabilità della classe 1 (Default)
        probability = model.predict_proba(scaled_input)[0, 1]
        
        # 6. Risultati a schermo
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Esito Modello")
            # Soglia prudenziale abbassata (es. 40%) visto il forte ribilanciamento
            if probability > 0.40:
                st.error(f"🔴 **ALTO RISCHIO**\n\nProbabilità di Default: **{probability:.1%}**")
            else:
                st.success(f"🟢 **APPROVATO**\n\nProbabilità di Default: **{probability:.1%}**")
                
        with col2:
            st.subheader("Metriche Cliente")
            st.write(f"- **Reddito:** €{income:,}")
            st.write(f"- **Anzianità:** {employed_years} anni")
            st.write(f"- **Età:** {age} anni")

        st.divider()
        
        # 7. Explainable AI (SHAP Waterfall Plot)
        st.subheader("🧠 Perché il modello ha preso questa decisione?")
        st.write("Il grafico a cascata (Waterfall) mostra come ogni singola caratteristica del cliente ha spinto il punteggio verso l'approvazione (blu) o verso il rifiuto (rosso), partendo dal rischio medio della banca.")
        
        # Inizializziamo l'Explainer di SHAP per i modelli ad albero
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(scaled_input)
        
        # Creiamo il plot catturando l'oggetto figure di Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Mostriamo il waterfall per la singola previsione (indice 0)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)