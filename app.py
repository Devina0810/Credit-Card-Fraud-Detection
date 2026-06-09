import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #1a1a2e, #e94560);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #1e1e2e; border-radius: 12px;
        padding: 1rem 1.2rem; border-left: 4px solid #e94560;
    }
    .fraud-badge {
        background: #e94560; color: white; padding: 6px 16px;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
    }
    .legit-badge {
        background: #2ecc71; color: white; padding: 6px 16px;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & cache everything ───────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop(columns=['Class'])
    y = df['Class']

    amount_scaler = StandardScaler()
    time_scaler   = StandardScaler()
    X = X.copy()
    X['Amount'] = amount_scaler.fit_transform(X[['Amount']])
    X['Time']   = time_scaler.fit_transform(X[['Time']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scale = y_train.value_counts()[0] / y_train.value_counts()[1]

    # LR Baseline
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train, y_train)

    # LR Balanced
    lr_bal = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_bal.fit(X_train, y_train)

    # LR SMOTE
    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_smote, y_train_smote)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # XGBoost Tuned
    xgb_tuned = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        gamma=0.1, scale_pos_weight=scale, random_state=42,
        eval_metric='aucpr', n_jobs=-1
    )
    xgb_tuned.fit(X_train, y_train)

    # Probabilities
    probs = {
        'LR Baseline':        lr_base.predict_proba(X_test)[:, 1],
        'LR + Class Weights': lr_bal.predict_proba(X_test)[:, 1],
        'LR + SMOTE':         lr_smote.predict_proba(X_test)[:, 1],
        'Random Forest':      rf.predict_proba(X_test)[:, 1],
        'XGBoost Tuned':      xgb_tuned.predict_proba(X_test)[:, 1],
    }

    best_thresh = 0.85
    explainer   = shap.TreeExplainer(xgb_tuned)

    return (X_train, X_test, y_train, y_test,
            xgb_tuned, explainer, probs, best_thresh,
            amount_scaler, time_scaler, scale)

# ── Sidebar ───────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-front-side.png", width=72)
st.sidebar.markdown("## 🛡️ Fraud Detection")
st.sidebar.markdown("**Model:** Tuned XGBoost  \n**ROC-AUC:** 0.9800  \n**Threshold:** 0.85")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
284,807 transactions | 492 fraud (0.17%)  
Features: V1–V28 (PCA), Time, Amount
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Devina Saini**  \n[GitHub](https://github.com/Devina0810) · [Kaggle](https://kaggle.com/midnight1708)")

# ── Header ────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown("**End-to-end ML pipeline** | Logistic Regression · Random Forest · XGBoost + SHAP Explainability")
st.markdown("---")

# ── Load data ─────────────────────────────────────────
with st.spinner("Loading dataset..."):
    try:
        df = load_data()
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False
        st.error("⚠️ `creditcard.csv` not found. Please place it in the same folder as `app.py`.")
        st.stop()

with st.spinner("Training models... this takes ~2 minutes on first run (cached after that)"):
    (X_train, X_test, y_train, y_test,
     xgb_tuned, explainer, probs, best_thresh,
     amount_scaler, time_scaler, scale) = train_models(df)

# ── Tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Data Insights",
    "🤖 Model Comparison",
    "🔍 SHAP Explainability",
    "🚨 Live Fraud Predictor"
])

# ═══════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    # Top metrics
    fraud_count = df['Class'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Legitimate", f"{fraud_count[0]:,}", f"{fraud_count[0]/len(df)*100:.2f}%")
    c3.metric("Fraudulent", f"{fraud_count[1]:,}", f"{fraud_count[1]/len(df)*100:.2f}%")
    c4.metric("Imbalance Ratio", f"{fraud_count[0]//fraud_count[1]}:1")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['Legitimate', 'Fraud'], fraud_count.values,
               color=['steelblue', 'crimson'], edgecolor='black', alpha=0.85)
        for i, v in enumerate(fraud_count.values):
            ax.text(i, v + 2000, f'{v:,}', ha='center', fontweight='bold', fontsize=10)
        ax.set_ylabel("Transactions")
        ax.set_title("Class Distribution (Raw Count)")
        st.pyplot(fig)
        plt.close()

        st.info("💡 If we predict ALL as Legitimate: Accuracy = 99.83% — but catches **zero fraud**. This is why accuracy is useless here.")

    with col2:
        st.markdown("#### Transaction Amount by Class")
        fig, ax = plt.subplots(figsize=(5, 4))
        fraud_amt = df[df['Class'] == 1]['Amount']
        legit_amt = df[df['Class'] == 0]['Amount']
        ax.hist(legit_amt, bins=50, alpha=0.6, color='steelblue',
                label=f'Legit (mean=${legit_amt.mean():.0f})', density=True)
        ax.hist(fraud_amt, bins=50, alpha=0.8, color='crimson',
                label=f'Fraud (mean=${fraud_amt.mean():.0f})', density=True)
        ax.set_xlim(0, 800)
        ax.set_xlabel("Amount ($)")
        ax.set_ylabel("Density")
        ax.set_title("Amount Distribution by Class")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### Feature Correlation with Fraud")
    correlations = df.corr()['Class'].drop('Class').sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['crimson' if x < 0 else 'steelblue' for x in correlations]
    correlations.plot(kind='barh', color=colors, edgecolor='black', alpha=0.8, ax=ax)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Correlation with Fraud")
    ax.set_title("Feature Correlation with Fraud (Red = negative, Blue = positive)")
    st.pyplot(fig)
    plt.close()

    st.markdown("**Top fraud signals:** V14, V12, V10 (strong negative correlation) and V4, V11 (positive).")

# ═══════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Model Comparison")

    # Build results table
    model_names = list(probs.keys())
    rows = []
    for name, prob in probs.items():
        y_pred = (prob >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            'Model': name,
            'ROC-AUC': round(roc_auc_score(y_test, prob), 4),
            'Avg Precision': round(average_precision_score(y_test, prob), 4),
            'Fraud Caught': int(tp),
            'Fraud Missed ❌': int(fn),
            'False Alarms ⚠️': int(fp),
            'Recall %': round(tp / (tp + fn) * 100, 1),
        })
    df_res = pd.DataFrame(rows)

    st.dataframe(
        df_res.style.highlight_max(subset=['ROC-AUC', 'Fraud Caught', 'Recall %'], color='#d4edda')
                    .highlight_min(subset=['Fraud Missed ❌', 'False Alarms ⚠️'], color='#d4edda'),
        use_container_width=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_list = ['gray', 'orange', 'gold', 'steelblue', 'crimson']
        for (name, prob), c in zip(probs.items(), colors_list):
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr, label=f'{name} ({auc:.3f})', color=c, linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models")
        ax.legend(fontsize=7)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Precision-Recall Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, prob), c in zip(probs.items(), colors_list):
            precision, recall, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            ax.plot(recall, precision, label=f'{name} (AP={ap:.3f})', color=c, linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves — All Models")
        ax.legend(fontsize=7)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### Confusion Matrix — Tuned XGBoost (Best Model)")
    y_pred_final = (probs['XGBoost Tuned'] >= best_thresh).astype(int)
    cm_final = confusion_matrix(y_test, y_pred_final)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Legit', 'Predicted Fraud'],
                yticklabels=['Actual Legit', 'Actual Fraud'],
                linewidths=2, linecolor='white', annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix (threshold={best_thresh})")
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════
# TAB 3 — SHAP
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 SHAP Feature Explainability")
    st.markdown("SHAP (SHapley Additive exPlanations) explains **why** the model flagged a transaction.")

    with st.spinner("Computing SHAP values on 300 test samples..."):
        X_sample = X_test.iloc[:300]
        shap_values = explainer.shap_values(X_sample)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Feature Importance (Mean |SHAP|)")
        fig, ax = plt.subplots(figsize=(6, 6))
        shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### SHAP Beeswarm — Feature Impact Direction")
        fig, ax = plt.subplots(figsize=(6, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("""
    **How to read this:**
    - **Red dots** = high feature value, **Blue dots** = low value  
    - Dots pushed **right** = pushed prediction toward fraud  
    - Dots pushed **left** = pushed prediction toward legitimate  
    - **V14, V10, V12** are the strongest fraud signals in this dataset
    """)

# ═══════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════
with tab4:
    st.subheader("🚨 Live Fraud Predictor")
    st.markdown("Enter transaction details below to get an instant fraud prediction with SHAP explanation.")

    # Quick fill buttons
    st.markdown("#### Quick Fill")
    qc1, qc2, qc3 = st.columns(3)

    fraud_sample   = X_test[y_test == 1].iloc[0].to_dict()
    legit_sample   = X_test[y_test == 0].iloc[0].to_dict()
    medium_sample  = X_test[y_test == 0].iloc[42].to_dict()

    if qc1.button("🔴 Load Real Fraud Case"):
        st.session_state['sample'] = fraud_sample
    if qc2.button("🟢 Load Real Legit Case"):
        st.session_state['sample'] = legit_sample
    if qc3.button("🟡 Load Random Case"):
        st.session_state['sample'] = medium_sample

    defaults = st.session_state.get('sample', legit_sample)

    st.markdown("---")
    st.markdown("#### Transaction Features")

    col1, col2 = st.columns(2)
    with col1:
        time_val   = st.number_input("Time (seconds since first transaction)", value=float(defaults.get('Time', 0.0)), format="%.4f")
        amount_val = st.number_input("Amount ($)", value=float(defaults.get('Amount', 0.0)), format="%.4f")

    # V1–V28 inputs in two columns
    v_values = {}
    cols = st.columns(4)
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            v_values[f'V{i}'] = st.number_input(
                f"V{i}", value=float(defaults.get(f'V{i}', 0.0)),
                format="%.4f", key=f"v{i}"
            )

    st.markdown("---")
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        transaction = {'Time': time_val, 'Amount': amount_val, **v_values}
        df_input = pd.DataFrame([transaction])
        df_input = df_input[X_train.columns]

        # Scale Amount and Time
        df_input['Amount'] = amount_scaler.transform(df_input[['Amount']])
        df_input['Time']   = time_scaler.transform(df_input[['Time']])

        prob     = xgb_tuned.predict_proba(df_input)[0][1]
        is_fraud = prob >= best_thresh

        if prob < 0.3:            risk = "🟢 LOW"
        elif prob < 0.6:          risk = "🟡 MEDIUM"
        elif prob < best_thresh:  risk = "🟠 HIGH"
        else:                     risk = "🔴 CRITICAL"

        # Display result
        st.markdown("### Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{prob*100:.1f}%")
        r2.metric("Risk Level", risk)
        r3.metric("Threshold", f"{best_thresh}")

        if is_fraud:
            st.markdown('<div style="background:#e94560;color:white;padding:16px;border-radius:10px;font-size:1.4rem;font-weight:800;text-align:center;">🚨 FRAUD DETECTED — Transaction Blocked</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#2ecc71;color:white;padding:16px;border-radius:10px;font-size:1.4rem;font-weight:800;text-align:center;">✅ LEGITIMATE — Transaction Approved</div>', unsafe_allow_html=True)

        # Probability gauge
        st.markdown("#### Fraud Probability Gauge")
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(0, 1, color='#eee', height=0.5)
        ax.barh(0, prob, color='crimson' if is_fraud else 'steelblue', height=0.5)
        ax.axvline(best_thresh, color='black', linewidth=2, linestyle='--', label=f'Threshold ({best_thresh})')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Fraud Probability")
        ax.legend(loc='upper right')
        ax.set_title(f"Fraud Probability: {prob*100:.1f}%")
        st.pyplot(fig)
        plt.close()

        # SHAP waterfall
        st.markdown("#### Why this prediction? (SHAP Explanation)")
        shap_vals_input = explainer.shap_values(df_input)
        expected        = explainer.expected_value

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values        = shap_vals_input[0],
                base_values   = expected,
                data          = df_input.iloc[0],
                feature_names = X_train.columns.tolist()
            ),
            show=False, max_display=12
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top 3 reasons
        feature_names = X_train.columns.tolist()
        top_idx = np.argsort(np.abs(shap_vals_input[0]))[::-1][:3]
        st.markdown("#### Top 3 Factors")
        for rank, idx in enumerate(top_idx, 1):
            fname = feature_names[idx]
            fval  = shap_vals_input[0][idx]
            direction = "↑ pushed toward FRAUD" if fval > 0 else "↓ pushed toward LEGITIMATE"
            st.markdown(f"**{rank}. {fname}** — SHAP value: `{fval:.3f}` — {direction}")
