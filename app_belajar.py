import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Konfigurasi Halaman
st.set_page_config(page_title="Aplikasi Analisis Big Data", layout="wide")

st.title("📊 Aplikasi Analisis Big Data Dinamis")
st.markdown("Unggah **Data Training** untuk melatih model, dan **Data Testing** untuk melakukan prediksi.")

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("1. Unggah Dataset")
train_file = st.sidebar.file_uploader("Unggah Data Training (CSV/Excel)", type=["csv", "xlsx"])
test_file = st.sidebar.file_uploader("Unggah Data Testing (CSV/Excel)", type=["csv", "xlsx"])

# Fungsi pembaca data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

# --- LOGIKA DATA TRAINING ---
if train_file is not None:
    df_train = load_data(train_file)
    
    st.subheader("📋 Cuplikan Data Training")
    st.dataframe(df_train.head())
    
    st.sidebar.header("2. Pengaturan Variabel")
    # Dinamis: Biarkan user memilih target dan fitur
    target_column = st.sidebar.selectbox("Pilih Kolom Target (y)", df_train.columns)
    
    # Secara default, pilih semua kolom kecuali target
    default_features = [col for col in df_train.columns if col != target_column]
    feature_columns = st.sidebar.multiselect("Pilih Kolom Fitur (X_train)", default_features, default=default_features)
    
    if len(feature_columns) > 0:
        X_train = df_train[feature_columns].copy()
        y_train = df_train[target_column].copy()
        
        # Preprocessing sederhana: Label Encoding untuk data teks/kategorikal
        le = LabelEncoder()
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            
        if y_train.dtype == 'object':
            y_train = le.fit_transform(y_train.astype(str))

        st.write(f"**Dimensi X_train:** {X_train.shape} | **Dimensi y_train:** {y_train.shape}")
        
        # Latih Model
        if st.sidebar.button("⚙️ Latih Model (Random Forest)"):
            with st.spinner("Sedang melatih model..."):
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                
                # Simpan model ke session_state agar bisa dipakai saat prediksi
                st.session_state['trained_model'] = model
                st.session_state['le'] = le
                st.session_state['feature_columns'] = feature_columns
                st.sidebar.success("Model berhasil dilatih!")

# --- LOGIKA DATA TESTING & PREDIKSI ---
if test_file is not None:
    if train_file is None:
        st.warning("⚠️ Silakan unggah Data Training terlebih dahulu untuk melatih model.")
    else:
        df_test = load_data(test_file)
        st.subheader("📋 Cuplikan Data Testing")
        st.dataframe(df_test.head())
        
        if 'trained_model' in st.session_state:
            st.sidebar.header("3. Prediksi")
            if st.sidebar.button("🚀 Lakukan Prediksi"):
                try:
                    # Ambil model dan fitur dari session
                    model = st.session_state['trained_model']
                    le = st.session_state['le']
                    features = st.session_state['feature_columns']
                    
                    X_test = df_test[features].copy()
                    
                    # Preprocessing data testing (sama dengan training)
                    for col in X_test.select_dtypes(include=['object']).columns:
                        X_test[col] = le.fit_transform(X_test[col].astype(str))
                        
                    # Eksekusi Prediksi
                    predictions = model.predict(X_test)
                    df_test['Hasil_Prediksi'] = predictions
                    
                    st.subheader("✅ Hasil Prediksi")
                    st.dataframe(df_test)
                    
                    # Fitur Download Hasil
                    csv_result = df_test.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Unduh Hasil Prediksi (CSV)",
                        data=csv_result,
                        file_name='hasil_prediksi_big_data.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}. Pastikan Data Testing memiliki kolom fitur yang persis sama dengan Data Training.")