import streamlit as st
import pandas as pd
import joblib

# Load semua komponen
loaded_label_encoder = joblib.load('label_encoder.joblib')
loaded_tfidf = joblib.load('tfidf_vectorizer.joblib')
loaded_pca = joblib.load('pca_model_167.joblib')
loaded_model = joblib.load('logistic_regression_model_167.joblib')

# Prediksi
def predict(text):
    X = loaded_tfidf.transform([text])
    X_pca = loaded_pca.transform(X.toarray())
    pred = loaded_model.predict(X_pca)
    return loaded_label_encoder.inverse_transform(pred)[0]

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Genre Film", layout="centered")
st.title("🎬 Klasifikasi Genre Film Otomatis")
st.write("Gunakan teks sinopsis film atau upload file CSV untuk memprediksi genre menggunakan model Logistic Regression.")

# Tombol Start Klasifikasi
if 'start' not in st.session_state:
    st.session_state.start = False
    
if st.button("🚀 Start Klasifikasi"):
    st.session_state.start = True

# Jika Tombol Sudah Ditekan
if st.session_state.get("start", False):
    st.subheader("📥 Masukkan Sinopsis Film atau Upload CSV")
    
    tab1, tab2 = st.tabs(["📝 Teks Manual", "📁 Upload CSV"])
    
    # Input Teks Manual
    with tab1:
        input_text = st.text_area("Masukkan sinopsis film:", height=500)
        if st.button("🔍 Klasifikasikan Teks"):
            if input_text.strip() == "":
                st.warning("Teks tidak boleh kosong.")
            else:
                result = predict(input_text)
                st.success(f"🎯 Prediksi Genre: **{result}**")
    
    # Upload CSV
    with tab2:
        uploaded_file = st.file_uploader("Upload file CSV (harus ada kolom 'sinopsis')", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if "sinopsis" not in df.columns:
                    st.error("Kolom 'sinopsis' tidak ditemukan dalam file.")
                else:
                    # Lakukan prediksi
                    texts = df["sinopsis"].fillna("").tolist()
                    tfidf_features = loaded_tfidf.transform(texts)
                    tfidf_pca = loaded_pca.transform(tfidf_features.toarray())
                    predictions = loaded_model.predict(tfidf_pca)
                    genres = loaded_label_encoder.inverse_transform(predictions)
                    
                    df["prediksi_genre"] = genres
                    st.subheader("📋 Hasil Klasifikasi:")
                    st.dataframe(df[["sinopsis", "prediksi_genre"]])
                    
                    # Visualisasi distribusi genre
                    st.subheader("📊 Distribusi Genre:")
                    genre_counts = df["prediksi_genre"].value_counts()
                    st.bar_chart(genre_counts)
                    
                    # Tombol download
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Unduh Hasil Klasifikasi", data=csv, file_name="hasil_klasifikasi.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")