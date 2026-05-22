import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_curve, auc, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Automasi Evaluasi Kelayakan Pinjaman",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .metric-card h2 { margin: 0; color: #2E7D32; font-size: 2rem; }
    .metric-card p  { margin: 4px 0 0; color: #555; font-size: 0.85rem; }
    .approved-box {
        background: #E8F5E9; border: 2px solid #2E7D32;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .rejected-box {
        background: #FFEBEE; border: 2px solid #C62828;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .section-header {
        color: #1a1a2e; font-size: 1.3rem; font-weight: 700;
        border-bottom: 3px solid #2E7D32; padding-bottom: 6px; margin: 20px 0 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading & preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    # Load dataset asli
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")
    
    # Pakai train.csv untuk modeling
    df = train_df.copy()
    
    # Drop Loan_ID kalau ada
    if 'Loan_ID' in df.columns:
        df.drop(columns=['Loan_ID'], inplace=True)
        
    # Encode target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Handle missing values
    categorical_cols = [
        'Gender', 'Married', 'Dependents',
        'Self_Employed', 'Credit_History'
    ]
    numerical_cols = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term'
    ]
    
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
        
    # Encode kategorikal
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3}
    }
    
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
    # Feature Engineering
    df['Log_ApplicantIncome']   = np.log1p(df['ApplicantIncome'])
    df['Log_CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
    df['Log_LoanAmount']        = np.log1p(df['LoanAmount'])
    df['Total_Income']          = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Log_Total_Income']      = np.log1p(df['Total_Income'])
    df['Income_Per_Dependent']  = df['Total_Income'] / (df['Dependents'] + 1)
    df['Loan_to_Income'] = (
        df['LoanAmount'] / (df['Total_Income']/1000 + 1)
    )
    df['Area_Income_Interaction'] = (
        df['Property_Area'] * df['Log_Total_Income']
    )
    df['Family_Size'] = (
        df['Dependents'] + df['Married'] + 1
    )
    
    return df

@st.cache_resource
def train_models(df):
    """Train all 4 models exactly as in notebook."""
    drop_cols = ['Gender', 'Dependents', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome',
                 'LoanAmount', 'Total_Income', 'Loan_Amount_Term']
                 
    df_reduced = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = df_reduced.drop(columns=['Loan_Status'])
    y = df_reduced['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    trained, results = {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        trained[name] = m
        results[name] = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'report':    classification_report(y_test, y_pred, output_dict=True),
            'y_pred':    y_pred,
            'y_proba':   m.predict_proba(X_test)[:, 1],
            'cm':        confusion_matrix(y_test, y_pred),
        }
        
    return trained, results, X_train, X_test, y_train, y_test, X.columns.tolist()

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_and_preprocess()
trained_models, results, X_train, X_test, y_train, y_test, feature_cols = train_models(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Eligibility")
    st.markdown("**SK5016 — Data Mining**")
    st.markdown("Kelompok 7 · ITB")
    st.divider()
    page = st.radio("Navigasi", [
        "📊 EDA",
        "🧬 Feature Engineering",
        "🤖 Model Performance",
        "🔮 Prediksi Langsung"
    ])
    st.divider()
    st.markdown("**Dataset:** Kaggle Finance Loan Approval")
    st.markdown(f"**Samples:** {len(df)}")
    st.markdown(f"**Features (after selection):** {len(feature_cols)}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EDA
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Eksplorasi distribusi dan hubungan antar fitur terhadap Loan Status.")
    
    # KPI row
    total     = len(df)
    approved  = df['Loan_Status'].sum()
    rejected  = total - approved
    apr_rate  = approved / total * 100
    
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, total, "Total Aplikasi"),
        (c2, approved, "Disetujui ✅"),
        (c3, rejected, "Ditolak ❌"),
        (c4, f"{apr_rate:.1f}%", "Approval Rate"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2><p>{label}</p>
        </div>""", unsafe_allow_html=True)
        
    st.markdown('<div class="section-header">Distribusi Target & Credit History</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['Loan_Status'].value_counts()
        ax.bar(['Ditolak (0)', 'Disetujui (1)'], counts.values, color=['#1565C0', '#2E7D32'], width=0.55, edgecolor='white')
        ax.set_title('Distribusi Loan Status', fontweight='bold')
        ax.set_ylabel('Jumlah')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
    with col_b:
        ct = pd.crosstab(df['Credit_History'], df['Loan_Status'])
        fig, ax = plt.subplots(figsize=(5, 4))
        ct.plot(kind='bar', ax=ax, color=['#1565C0', '#2E7D32'], edgecolor='white', width=0.6)
        ax.set_title('Credit History vs Loan Status', fontweight='bold')
        ax.set_xlabel('Credit History')
        ax.set_ylabel('Jumlah')
        ax.set_xticklabels(['Tidak Ada (0)', 'Ada (1)'], rotation=0)
        ax.legend(['Ditolak', 'Disetujui'])
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
    st.markdown('<div class="section-header">Distribusi Fitur Numerik</div>', unsafe_allow_html=True)
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, num_cols):
        ax.hist(df[col], bins=30, color='#2E7D32', edgecolor='white', alpha=0.8)
        ax.set_title(col, fontweight='bold', fontsize=10)
        ax.set_xlabel('Value'); ax.set_ylabel('Count')
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.markdown('<div class="section-header">Correlation Heatmap (Semua Fitur)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(16, 12))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='mako',
                linewidth=0.4, ax=ax, annot_kws={'size': 7}, mask=mask)
    ax.set_title('Correlation Heatmap', fontweight='bold', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Feature Engineering":
    st.title("🧬 Feature Engineering & Selection")
    
    st.markdown('<div class="section-header">Fitur Baru yang Dibuat</div>', unsafe_allow_html=True)
    feats = {
        "Total_Income": "ApplicantIncome + CoapplicantIncome — gabungan pendapatan keluarga",
        "Log_Total_Income": "log1p(Total_Income) — normalisasi distribusi skewed",
        "Income_Per_Dependent": "Total_Income / (Dependents + 1) — daya beli per tanggungan",
        "Loan_to_Income": "LoanAmount / (Total_Income/1000 + 1) — rasio beban utang",
        "Area_Income_Interaction": "Property_Area × Log_Total_Income — interaksi lokasi & penghasilan",
        "Family_Size": "Dependents + Married + 1 — ukuran keluarga total",
    }
    
    for name, desc in feats.items():
        st.markdown(f"""
        <div style="background:#f8f9fa;border-left:4px solid #2E7D32; padding:12px 16px;border-radius:4px;margin-bottom:8px;">
            <strong style="color:#2E7D32;">{name}</strong><br>
            <span style="color:#555;font-size:0.9rem;">{desc}</span>
        </div>""", unsafe_allow_html=True)
        
    st.markdown('<div class="section-header">Feature Selection — Chi-Square & T-Test</div>', unsafe_allow_html=True)
    
    categorical_cols = ['Married', 'Education', 'Credit_History', 'Property_Area', 'Family_Size']
    numerical_cols   = ['Log_ApplicantIncome', 'Log_CoapplicantIncome', 'Log_LoanAmount',
                        'Log_Total_Income', 'Income_Per_Dependent', 'Loan_to_Income',
                        'Area_Income_Interaction']
                        
    chi_rows = []
    for col in categorical_cols:
        if col in df.columns:
            ct = pd.crosstab(df[col], df['Loan_Status'])
            chi2, p, _, _ = chi2_contingency(ct)
            chi_rows.append({
                'Feature': col, 
                'χ²': f'{chi2:.2f}', 
                'p-value': f'{p:.4f}',
                'Status': '✅ Significant' if p < 0.05 else '❌ Not Significant'
            })
            
    t_rows = []
    for col in numerical_cols:
        if col in df.columns:
            approved = df[df['Loan_Status']==1][col]
            rejected = df[df['Loan_Status']==0][col]
            t, p = ttest_ind(approved, rejected)
            t_rows.append({
                'Feature': col, 
                't-stat': f'{t:.2f}',
                'p-value': f'{p:.4f}',
                'Status': '✅ Significant' if p < 0.05 else '❌ Not Significant'
            })
            
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Chi-Square Test (Kategorikal)**")
        st.dataframe(pd.DataFrame(chi_rows), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**T-Test (Numerikal)**")
        st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)
        
    st.markdown('<div class="section-header">Fitur Dihapus (tidak signifikan atau redundan)</div>', unsafe_allow_html=True)
    removed = ['Loan_ID', 'Gender', 'Dependents', 'Self_Employed',
               'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
               'Total_Income', 'Loan_Amount_Term']
    st.markdown(" · ".join([f"`{c}`" for c in removed]))
    
    st.markdown('<div class="section-header">Fitur Final untuk Model</div>', unsafe_allow_html=True)
    st.markdown(" · ".join([f"**`{c}`**" for c in feature_cols]))

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Evaluasi Performa Model")
    
    # ── Summary table ──
    st.markdown('<div class="section-header">Ringkasan Performa (Weighted Avg)</div>', unsafe_allow_html=True)
    summary = []
    best_acc, best_name = 0, ''
    for name, res in results.items():
        r = res['report']['weighted avg']
        acc = res['accuracy']
        summary.append({
            'Model': name,
            'Accuracy': f"{acc:.3f}",
            'Precision': f"{r['precision']:.3f}",
            'Recall': f"{r['recall']:.3f}",
            'F1-Score': f"{r['f1-score']:.3f}",
        })
        if acc > best_acc:
            best_acc, best_name = acc, name
            
    df_sum = pd.DataFrame(summary)
    def highlight_best(row):
        return ['background-color: #E8F5E9; font-weight: bold'
                if row['Model'] == best_name else '' for _ in row]
    st.dataframe(df_sum.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)
    st.success(f"🏆 **Model Terbaik: {best_name}** — Accuracy: {best_acc:.1%}")
    
    # ── ROC Curves ──
    st.markdown('<div class="section-header">ROC Curves (menggunakan predict_proba)</div>', unsafe_allow_html=True)
    colors = ['#1565C0', '#E65100', '#2E7D32', '#7B1FA2']
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f'{name} (AUC = {roc_auc:.2f})')
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance', alpha=0.6)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Perbandingan 4 Model', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    # ── Confusion matrices ──
    st.markdown('<div class="section-header">Confusion Matrix per Model</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (name, res) in zip(cols, results.items()):
        with col:
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False,
                        xticklabels=['Ditolak', 'Disetujui'], yticklabels=['Ditolak', 'Disetujui'])
            ax.set_title(name, fontsize=9, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('Actual', fontsize=8)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
    # ── Feature Importance (RF) ──
    st.markdown('<div class="section-header">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    rf = trained_models['Random Forest']
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = ['#2E7D32' if v == importances.max() else '#90CAF9' for v in importances.values]
    ax.barh(importances.index, importances.values, color=colors_bar, edgecolor='white')
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDIKSI LANGSUNG
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Langsung":
    st.title("🔮 Prediksi Kelayakan Pinjaman")
    st.markdown("Masukkan data pemohon, sistem akan memprediksi apakah pinjaman layak disetujui.")
    
    col_form, col_result = st.columns([1, 1])
    with col_form:
        st.markdown('<div class="section-header">Data Pemohon</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            married     = st.selectbox("Status Pernikahan", ["Belum Menikah", "Menikah"])
            education   = st.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
            credit_hist = st.selectbox("Riwayat Kredit", ["Ada (1)", "Tidak Ada (0)"])
        with c2:
            property_area = st.selectbox("Area Properti", ["Urban", "Semiurban", "Rural"])
            model_choice  = st.selectbox("Pilih Model", list(trained_models.keys()))
            
        st.markdown("**Data Keuangan**")
        c3, c4 = st.columns(2)
        with c3:
            applicant_income   = st.number_input("Pendapatan Pemohon (Rp)", 1000, 100000, 5000, 500)
            coapplicant_income = st.number_input("Pendapatan Co-Pemohon (Rp)", 0, 50000, 0, 500)
        with c4:
            loan_amount = st.number_input("Jumlah Pinjaman (Rp)", 10, 700, 150, 10)
            
        predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)
        
    with col_result:
        st.markdown('<div class="section-header">Hasil Prediksi</div>', unsafe_allow_html=True)
        if predict_btn:
            # Build input exactly like notebook pipeline
            married_val     = 1 if married == "Menikah" else 0
            education_val   = 1 if education == "Graduate" else 0
            credit_val      = 1 if "Ada" in credit_hist else 0
            area_val        = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
            
            total_income = applicant_income + coapplicant_income
            log_total    = np.log1p(total_income)
            dependents   = 0  # default
            family_size  = dependents + married_val + 1
            
            input_data = pd.DataFrame([{
                'Married':                 married_val,
                'Education':               education_val,
                'Credit_History':          credit_val,
                'Property_Area':           area_val,
                'Log_ApplicantIncome':     np.log1p(applicant_income),
                'Log_CoapplicantIncome':   np.log1p(coapplicant_income),
                'Log_LoanAmount':          np.log1p(loan_amount),
                'Log_Total_Income':        log_total,
                'Income_Per_Dependent':    total_income / (dependents + 1),
                'Loan_to_Income':          loan_amount / (total_income / 1000 + 1),
                'Area_Income_Interaction': area_val * log_total,
                'Family_Size':             family_size,
            }])
            
            # Align columns to training feature order
            for col in feature_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[feature_cols]
            
            model   = trained_models[model_choice]
            pred    = model.predict(input_data)[0]
            proba   = model.predict_proba(input_data)[0]
            conf    = proba[pred] * 100
            
            if pred == 1:
                st.markdown(f"""
                <div class="approved-box">
                    <h2 style="color:#2E7D32;">✅ DISETUJUI</h2>
                    <p style="color:#555;font-size:1rem;">Pinjaman layak untuk disetujui</p>
                    <h3 style="color:#2E7D32;">{conf:.1f}% confidence</h3>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected-box">
                    <h2 style="color:#C62828;">❌ DITOLAK</h2>
                    <p style="color:#555;font-size:1rem;">Pinjaman tidak memenuhi kriteria kelayakan</p>
                    <h3 style="color:#C62828;">{conf:.1f}% confidence</h3>
                </div>""", unsafe_allow_html=True)
                
            # Probability bar
            st.markdown("**Probabilitas per Kelas:**")
            prob_df = pd.DataFrame({
                'Status': ['Ditolak (0)', 'Disetujui (1)'],
                'Probabilitas': [f"{proba[0]:.1%}", f"{proba[1]:.1%}"]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(['Ditolak', 'Disetujui'], [proba[0], proba[1]],
                    color=['#E57373', '#66BB6A'], height=0.5, edgecolor='white')
            for i, v in enumerate([proba[0], proba[1]]):
                ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold')
            ax.set_xlim(0, 1.15)
            ax.set_xlabel('Probabilitas')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
            # Feature breakdown
            st.markdown("**Ringkasan Input:**")
            breakdown = pd.DataFrame({
                'Fitur': ['Credit History', 'Total Income', 'Loan Amount', 'Loan-to-Income', 'Property Area'],
                'Nilai': [
                    '✅ Ada' if credit_val else '❌ Tidak Ada',
                    f"Rp {total_income:,}",
                    f"Rp {loan_amount:,}",
                    f"{loan_amount / (total_income / 1000 + 1):.2f}",
                    property_area
                ]
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Isi form di sebelah kiri, lalu klik **Prediksi Sekarang**.")
            st.markdown("""
            **Tips penggunaan:**
            - Credit History adalah faktor paling berpengaruh
            - Pilih model berbeda untuk bandingkan hasil
            - Semiurban area umumnya memiliki approval rate lebih tinggi
            """)