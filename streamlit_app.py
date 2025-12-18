# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Pharmacogenetic Warfarin Dosing Assistant",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("⚕️ Pharmacogenetic Warfarin Dosing Assistant")
st.markdown("""
**Evidence-based dosing using top 5 predictors from SHAP analysis:**
1. VKORC1 Sensitivity  
2. CYP2C9 Activity  
3. Ethnicity  
4. Genetic Burden  
5. Age

*Uses exact same feature encoding as training data*
""")

# --- Load Model (with error handling) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("final_warfarin_model_xgboost.pkl")
        return model, None
    except FileNotFoundError:
        return None, "Model file 'final_warfarin_model_xgboost.pkl' not found. Please ensure it's in the same directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

model, model_error = load_model()

if model_error:
    st.error(f"❌ **Critical Error:** {model_error}")
    st.info("""
    **Troubleshooting steps:**
    1. Ensure `final_warfarin_model_xgboost.pkl` is in the same directory as this app
    2. If deploying to Streamlit Cloud, include the model file in your GitHub repository
    3. Check that the model file is not corrupted
    """)
    st.stop()

# --- ALL YOUR ORIGINAL CONSTANTS AND FUNCTIONS (preserved exactly) ---
CYP2C9_ACTIVITY = {"Normal": 2.0, "Intermediate": 1.0, "Poor": 0.5}
VKORC1_SENS = {"GG": 0.0, "AG": 1.0, "AA": 2.0}
CYP4F2_SCORE = {"CC": 0.0, "CT": 0.5, "TT": 2.0}

def calculate_genetic_burden(cyp2c9, vkorc1, cyp4f2):
    cyp2c9_score = CYP2C9_ACTIVITY.get(cyp2c9, 2.0)
    vkorc1_score = VKORC1_SENS.get(vkorc1, 0.0)
    cyp4f2_score = CYP4F2_SCORE.get(cyp4f2, 0.0)
    burden = (
        (2.0 - cyp2c9_score) * 0.4 +
        vkorc1_score * 0.4 +
        cyp4f2_score * 0.2
    )
    return round(burden, 2)

ETHNICITY_OHE = {
    "Asian": {'Ethnicity_Asian': 1, 'Ethnicity_Caucasian': 0, 'Ethnicity_Hispanic': 0, 'Ethnicity_Other': 0},
    "Caucasian": {'Ethnicity_Asian': 0, 'Ethnicity_Caucasian': 1, 'Ethnicity_Hispanic': 0, 'Ethnicity_Other': 0},
    "Hispanic": {'Ethnicity_Asian': 0, 'Ethnicity_Caucasian': 0, 'Ethnicity_Hispanic': 1, 'Ethnicity_Other': 0},
    "Other": {'Ethnicity_Asian': 0, 'Ethnicity_Caucasian': 0, 'Ethnicity_Hispanic': 0, 'Ethnicity_Other': 1},
}

ALCOHOL_OHE = {
    "Heavy": {'Alcohol_Intake_Light': 0, 'Alcohol_Intake_Moderate': 0, 'Alcohol_Intake_Unknown': 0},
    "Light": {'Alcohol_Intake_Light': 1, 'Alcohol_Intake_Moderate': 0, 'Alcohol_Intake_Unknown': 0},
    "Moderate": {'Alcohol_Intake_Light': 0, 'Alcohol_Intake_Moderate': 1, 'Alcohol_Intake_Unknown': 0},
    "Unknown": {'Alcohol_Intake_Light': 0, 'Alcohol_Intake_Moderate': 0, 'Alcohol_Intake_Unknown': 1},
}

SMOKING_OHE = {
    "Former Smoker": {'Smoking_Status_Non-smoker': 1, 'Smoking_Status_Smoker': 0},
    "Non-smoker": {'Smoking_Status_Non-smoker': 1, 'Smoking_Status_Smoker': 0},
    "Smoker": {'Smoking_Status_Non-smoker': 0, 'Smoking_Status_Smoker': 1},
}

DIET_OHE = {
    "High": {'Diet_VitK_Intake_Low': 0, 'Diet_VitK_Intake_Medium': 0},
    "Low": {'Diet_VitK_Intake_Low': 1, 'Diet_VitK_Intake_Medium': 0},
    "Medium": {'Diet_VitK_Intake_Low': 0, 'Diet_VitK_Intake_Medium': 1},
    "Unknown": {'Diet_VitK_Intake_Low': 0, 'Diet_VitK_Intake_Medium': 0},
}

COLUMNS_ORDER = [
    'Age', 'Weight_kg', 'Height_cm', 'CYP2C9_Activity', 'VKORC1_Sensitivity',
    'CYP4F2_Score', 'Genetic_Burden', 'Sex_M',
    'Ethnicity_Asian', 'Ethnicity_Caucasian', 'Ethnicity_Hispanic', 'Ethnicity_Other',
    'Alcohol_Intake_Light', 'Alcohol_Intake_Moderate', 'Alcohol_Intake_Unknown',
    'Smoking_Status_Non-smoker', 'Smoking_Status_Smoker',
    'Diet_VitK_Intake_Low', 'Diet_VitK_Intake_Medium'
]

# --- Your EXACT prediction function (preserved 100%) ---
def predict_warfarin(age, weight, height_cm, cyp2c9, vkorc1, cyp4f2, ethnicity,
                     sex, alcohol, smoking, diet_vitk,
                     amiodarone=False, antibiotics=False, statins=False, aspirin=False):
    try:
        # Convert inputs
        age = int(age)
        weight = float(weight)
        height_cm = float(height_cm)
        
        # Calculate derived metrics
        height_m = height_cm / 100.0
        bmi = weight / (height_m ** 2) if height_m > 0 else 0
        genetic_burden = calculate_genetic_burden(cyp2c9, vkorc1, cyp4f2)
        ethnicity_map = ETHNICITY_OHE.get(ethnicity, ETHNICITY_OHE["Caucasian"])

        # Build patient data row
        patient_data = {
            "Age": age, "Weight_kg": weight, "Height_cm": height_cm,
            "CYP2C9_Activity": CYP2C9_ACTIVITY[cyp2c9],
            "VKORC1_Sensitivity": VKORC1_SENS[vkorc1],
            "CYP4F2_Score": CYP4F2_SCORE[cyp4f2],
            "Genetic_Burden": genetic_burden,
            "Sex_M": 1 if sex == "M" else 0,
            **ethnicity_map,
            **ALCOHOL_OHE[alcohol],
            **SMOKING_OHE[smoking],
            **DIET_OHE[diet_vitk],
        }

        for col in COLUMNS_ORDER:
            patient_data.setdefault(col, 0)

        df = pd.DataFrame([patient_data])[COLUMNS_ORDER]
        prediction = float(model.predict(df)[0])

        # --- Clinical interpretation ---
        genetic_summary = []
        if cyp2c9 != "Normal":
            effect = "↑ dose req" if cyp2c9 == "Poor" else "↗ dose req"
            genetic_summary.append(f"CYP2C9 {cyp2c9} ({effect})")
        if vkorc1 != "GG":
            effect = "↓ dose req" if vkorc1 == "AA" else "↘ dose req"
            genetic_summary.append(f"VKORC1 {vkorc1} ({effect})")
        if cyp4f2 != "CC":
            effect = "↑ dose req" if cyp4f2 == "TT" else "↗ dose req"
            genetic_summary.append(f"CYP4F2 {cyp4f2} ({effect})")

        # Dose classification
        if prediction < 2.0:
            dose_class, dose_icon, clinical_advice = "VERY LOW", "🔴", "Consider alternative anticoagulants or verify genetic testing"
        elif prediction < 3.0:
            dose_class, dose_icon, clinical_advice = "LOW", "🟡", "Start low, monitor INR closely every 2-3 days initially"
        elif prediction <= 7.0:
            dose_class, dose_icon, clinical_advice = "STANDARD", "🟢", "Standard initiation protocol, check INR after 48-72h"
        elif prediction <= 10.0:
            dose_class, dose_icon, clinical_advice = "HIGH", "🟠", "Higher starting dose, confirm patient factors, monitor INR frequently"
        else:
            dose_class, dose_icon, clinical_advice = "VERY HIGH", "🔴", "Unusual dose - review all inputs, consider drug interactions"

        # Drug interactions
        interactions = []
        if amiodarone: interactions.append("• **Amiodarone:** Potent CYP2C9 inhibitor - 30-50% dose reduction typical")
        if antibiotics: interactions.append("• **Antibiotics:** May alter vitamin K synthesis - monitor INR closely")
        if statins: interactions.append("• **Statins:** Variable interaction - check for specific agent")
        if aspirin: interactions.append("• **Aspirin:** Increases bleeding risk independent of INR - assess indication")

        # Return structured results
        return {
            "prediction": prediction,
            "dose_class": dose_class,
            "dose_icon": dose_icon,
            "clinical_advice": clinical_advice,
            "bmi": bmi,
            "genetic_burden": genetic_burden,
            "genetic_summary": genetic_summary,
            "interactions": interactions,
            "error": None
        }

    except Exception as e:
        return {"error": f"❌ **Error:** {type(e).__name__}: {str(e)}\n\nPlease check all inputs are valid."}

# --- STREAMLIT USER INTERFACE ---
with st.sidebar:
    st.header("📋 Patient Parameters")
    
    # Demographics
    st.subheader("Demographics")
    age = st.slider("Age (years)", 18, 100, 65, 
                   help="Older patients generally require lower doses")
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.1,
                           help="Higher weight increases dose requirement")
    height_cm = st.number_input("Height (cm)", 120.0, 220.0, 170.0, 0.1,
                              help="Used for BMI and body surface area calculation")
    sex = st.radio("Sex", ["M", "F"], 
                  help="Male patients often require slightly higher doses")
    ethnicity = st.selectbox("Ethnicity", ["Caucasian", "Asian", "Hispanic", "Other"], 
                           help="#3 most important predictor - Asian ancestry reduces dose")
    
    # Genetics
    st.subheader("Genetics")
    cyp2c9 = st.radio("CYP2C9 Phenotype", ["Normal", "Intermediate", "Poor"], 
                     help="Metabolism rate - #2 most important predictor")
    vkorc1 = st.radio("VKORC1 Genotype", ["AG", "GG", "AA"], 
                     help="Sensitivity - #1 most important predictor")
    cyp4f2 = st.radio("CYP4F2 Genotype", ["CC", "CT", "TT"], 
                     help="Vitamin K metabolism - minor contributor")
    
    # Lifestyle
    st.subheader("Lifestyle")
    alcohol = st.selectbox("Alcohol Intake", ["Unknown", "Light", "Moderate", "Heavy"],
                         help="Heavy alcohol use may increase dose requirements")
    smoking = st.selectbox("Smoking Status", ["Non-smoker", "Former Smoker", "Smoker"],
                         help="Smoking induces CYP enzymes - may increase dose")
    diet_vitk = st.selectbox("Dietary Vitamin K", ["Medium", "Low", "High", "Unknown"],
                           help="High vitamin K intake increases dose requirement")
    
    # Drug Interactions
    st.subheader("🚨 Drug Interactions")
    amiodarone = st.checkbox("Amiodarone", 
                           help="Potent CYP2C9 inhibitor - reduces dose by 30-50%")
    antibiotics = st.checkbox("Recent Antibiotics", 
                            help="May alter gut flora and vitamin K synthesis")
    statins = st.checkbox("Statin Therapy", 
                        help="Variable interaction - monitor closely")
    aspirin = st.checkbox("Aspirin Use", 
                        help="Increases bleeding risk - assess carefully")
    
    # Calculate Button
    calculate_button = st.button("Calculate Warfarin Dose", type="primary", use_container_width=True)

# --- EXAMPLE CASES (from your Gradio app) ---
with st.expander("📚 Example Patient Scenarios (Click to Load)"):
    examples = st.columns(4)
    
    with examples[0]:
        if st.button("Standard Case", use_container_width=True):
            st.session_state.age = 65
            st.session_state.weight = 70.0
            st.session_state.height_cm = 170.0
            st.session_state.sex = "M"
            st.session_state.ethnicity = "Caucasian"
            st.session_state.cyp2c9 = "Normal"
            st.session_state.vkorc1 = "GG"
            st.session_state.cyp4f2 = "CC"
            st.session_state.alcohol = "Moderate"
            st.session_state.smoking = "Non-smoker"
            st.session_state.diet_vitk = "Medium"
            st.rerun()
    
    with examples[1]:
        if st.button("High-Risk Case", use_container_width=True):
            st.session_state.age = 78
            st.session_state.weight = 58.0
            st.session_state.height_cm = 155.0
            st.session_state.sex = "F"
            st.session_state.ethnicity = "Asian"
            st.session_state.cyp2c9 = "Poor"
            st.session_state.vkorc1 = "AA"
            st.session_state.cyp4f2 = "TT"
            st.session_state.alcohol = "Unknown"
            st.session_state.smoking = "Non-smoker"
            st.session_state.diet_vitk = "Low"
            st.rerun()
    
    with examples[2]:
        if st.button("Complex Case", use_container_width=True):
            st.session_state.age = 45
            st.session_state.weight = 90.0
            st.session_state.height_cm = 185.0
            st.session_state.sex = "M"
            st.session_state.ethnicity = "Hispanic"
            st.session_state.cyp2c9 = "Intermediate"
            st.session_state.vkorc1 = "AG"
            st.session_state.cyp4f2 = "CT"
            st.session_state.alcohol = "Heavy"
            st.session_state.smoking = "Smoker"
            st.session_state.diet_vitk = "High"
            st.rerun()

# --- MAIN RESULTS DISPLAY ---
if calculate_button:
    st.header("📄 Clinical Recommendation")
    
    with st.spinner("Calculating dose based on pharmacogenetic profile..."):
        result = predict_warfarin(
            st.session_state.get('age', age),
            st.session_state.get('weight', weight),
            st.session_state.get('height_cm', height_cm),
            st.session_state.get('cyp2c9', cyp2c9),
            st.session_state.get('vkorc1', vkorc1),
            st.session_state.get('cyp4f2', cyp4f2),
            st.session_state.get('ethnicity', ethnicity),
            st.session_state.get('sex', sex),
            st.session_state.get('alcohol', alcohol),
            st.session_state.get('smoking', smoking),
            st.session_state.get('diet_vitk', diet_vitk),
            st.session_state.get('amiodarone', amiodarone),
            st.session_state.get('antibiotics', antibiotics),
            st.session_state.get('statins', statins),
            st.session_state.get('aspirin', aspirin)
        )
    
    if result["error"]:
        st.error(result["error"])
    else:
        # Dose Recommendation
        st.subheader(f"{result['dose_icon']} {result['dose_class']} DOSE RANGE")
        st.markdown(f"### Predicted Maintenance Dose: **{result['prediction']:.2f} mg/day**")
        
        # Patient Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", f"{age}y")
            st.metric("Sex", sex)
        with col2:
            st.metric("Weight", f"{weight} kg")
            st.metric("BMI", f"{result['bmi']:.1f}")
        with col3:
            st.metric("Ethnicity", ethnicity)
            st.metric("Genetic Burden", f"{result['genetic_burden']:.2f}/2.0")
        
        # Genetics Summary
        st.markdown("#### Genetic Profile")
        if result['genetic_summary']:
            for summary in result['genetic_summary']:
                st.write(summary)
        else:
            st.write("Normal variants (no significant genetic dose modifications)")
        
        # Initial Regimen
        st.success("#### Initial Regimen")
        st.markdown(f"""
        • **Start with:** {result['prediction'] * 0.5:.2f} mg daily for 2 days  
        • **Check INR:** After 48-72 hours  
        • **Adjust by:** 0.5-1 mg based on INR response  
        • **Target INR:** 2.0-3.0 (standard) or 2.5-3.5 (mechanical valves)
        """)
        
        # Drug Interactions
        if result['interactions']:
            st.warning("#### Drug Interaction Considerations")
            for interaction in result['interactions']:
                st.markdown(interaction)
        
        # Clinical Guidance
        st.info(f"#### Clinical Guidance\n{result['clinical_advice']}")
        
        # Model Information
        with st.expander("📊 Model Information & Disclaimer"):
            st.markdown("""
            **Model Performance & Context:**
            - Based on SHAP analysis: VKORC1 > CYP2C9 > Ethnicity > Genetic Burden > Age
            - R² = 0.84, RMSE = 0.64 mg (clinically acceptable)
            - Trained on n=30,000 simulated patients
            - Genetic factors explain ~40-50% of dose variability
            - Ethnicity contributes ~15-20% of dose variability
            
            ⚠️ **For Research/Demonstration Purposes Only**  
            This tool is intended for educational and research use. Clinical validation is required before any application in patient care. Always combine algorithmic predictions with clinical judgment and INR monitoring.
            """)
else:
    # Initial welcome state
    st.info("👈 **Please fill in the patient parameters on the sidebar and click 'Calculate Warfarin Dose' to get a personalized recommendation.**")
    
    # Quick info about the app
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy (R²)", "0.84")
        st.caption("Explains 84% of dose variability")
    with col2:
        st.metric("Prediction Error (RMSE)", "0.64 mg")
        st.caption("Clinically acceptable range")

# --- Footer ---
st.markdown("---")
st.caption("Pharmacogenetic Warfarin Dosing Assistant v1.0 | For research and educational use only")
