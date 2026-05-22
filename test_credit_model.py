import pytest
import joblib
from app import calculate_risk

@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_features = joblib.load('feature_names.pkl')
    return model, scaler, expected_features

def test_long_history_client(artifacts):
    model, scaler, expected_features = artifacts
    
    # Cliente storico (alta esposizione = maggior rischio statistico nel dataset)
    historical_client = {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'Y',
        'AMT_INCOME_TOTAL': 120000,
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Married',
        'AGE_YEARS': 45,
        'EMPLOYED_YEARS': 20
    }
    
    prob = calculate_risk(model, scaler, expected_features, historical_client)
    
    # Il modello assegna circa il 47% a causa dello scale_pos_weight e della lunga esposizione
    assert prob > 0.40, f"Errore: Il cliente storico ha uno score anomalo ({prob:.2%})"

def test_new_client_low_exposure(artifacts):
    model, scaler, expected_features = artifacts
    
    # Cliente nuovo (nessuna storia = non ha ancora avuto modo di fare default)
    new_client = {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'N',
        'AMT_INCOME_TOTAL': 15000,
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Lower secondary',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'AGE_YEARS': 22,
        'EMPLOYED_YEARS': 0
    }
    
    prob = calculate_risk(model, scaler, expected_features, new_client)
    
    # Il modello assegna un rischio bassissimo (circa 1%) a causa del survival bias
    assert prob < 0.10, f"Errore: Il nuovo cliente ha uno score troppo alto ({prob:.2%})"

def test_deterministic_output(artifacts):
    model, scaler, expected_features = artifacts
    
    standard_client = {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'Y',
        'AMT_INCOME_TOTAL': 45000,
        'NAME_INCOME_TYPE': 'Commercial associate',
        'NAME_EDUCATION_TYPE': 'Secondary / secondary special',
        'NAME_FAMILY_STATUS': 'Married',
        'AGE_YEARS': 35,
        'EMPLOYED_YEARS': 5
    }
    
    prob_1 = calculate_risk(model, scaler, expected_features, standard_client)
    prob_2 = calculate_risk(model, scaler, expected_features, standard_client)
    
    assert prob_1 == pytest.approx(prob_2), "Il modello non è deterministico!"
    