import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_sidebar():
    st.session_state.page = "sidebar"

# Load models
fruit_model_svm = pd.read_pickle('SVM_fruit.pkl')
fruit_model_rfc = pd.read_pickle('RFC_fruit.pkl')
wine_model_kmeans = pd.read_pickle('kmean_wine.pkl')

st.title("Prediksi Machine Learning")
st.write("**Prediksi Machine Learning Menggunakan Agortima SVM, Random Forest dan K-Means.**")
# Pilih kategori
st.write("### Pilih Kategori")
option = st.selectbox("Klasifikasi:", ("Fruit", "Wine"))
# Pilih algoritma
if option == "Wine":
    st.write("### Pilih Algoritma")
    algorithm = "K-Means"
    try:
        # Load model .pkl
        with open("kmean_wine.pkl", "rb") as file:
            kmeans_model = pickle.load(file)
        # Simulasikan dataset
        wine_data = pd.DataFrame({
            "alcohol": np.random.uniform(10, 15, 200),
            "total_phenols": np.random.uniform(0.1, 5, 200),
        })
        wine_features = wine_data.values
        # Input untuk jumlah maksimal K
        max_k = st.slider("Pilih Maksimal K", min_value=2, max_value=20, value=10)
        # Tombol untuk menampilkan Elbow Method
        if st.button("Tampilkan Grafik Elbow Method"):
            # Hitung SSE untuk setiap nilai K
            sse = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(wine_features)
                sse.append(kmeans.inertia_)
            # Plot grafik Elbow Method
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_k + 1), sse, marker='o')
            plt.xlabel("Jumlah Kluster (K)")
            plt.ylabel("Sum of Squared Errors (SSE)")
            plt.title("Elbow Method untuk Menentukan Nilai Optimal K")
            plt.grid(True)
            # Tampilkan grafik di Streamlit
            st.pyplot(plt)
    except FileNotFoundError:
        st.error("Model kmean_wine.pkl tidak ditemukan! Pastikan file ada di direktori yang sama.")
else:
    st.write("### Pilih Algoritma")
    algorithm = st.selectbox("Algoritma:", ("SVM", "Random Forest"))
st.markdown("---")
# Dictionaries for fish, fruit, and pumpkin types
fish_types = {
    0: "Anabas testudineus",
    1: "Coilia dussumieri",
    2: "Otolithoides biauritus",
    3: "Otolithoides pama",
    4: "Pethia conchonius",
    5: "Polynemus paradiseus",
    6: "Puntius lateristriga",
    7: "Setipinna taty",
    8: "Sillaginopsis panijus"
}
fruit_types = {0: "Grapefruit", 1: "Orange"}
pumpkin_types = {0: "Çerçevelik", 1: "Ürgüp Sivrisi"}
# Input form based on category
with st.form(key='prediction_form'):
    if option == "Fruit":
        st.write("### Masukkan Data Buah")
        diameter = st.number_input('Diameter Buah (dalam cm)', min_value=0.0, format="%.2f")
        weight = st.number_input('Berat Buah (dalam gram)', min_value=0.0, format="%.2f")
        red = st.slider('Skor Warna Buah Merah', 0, 255, 0)
        green = st.slider('Skor Warna Buah Hijau', 0, 255, 0)
        blue = st.slider('Skor Warna Buah Biru', 0, 255, 0)
        submit = st.form_submit_button(label='Prediksi Jenis Buah')
        if submit:
            input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)
            if algorithm == "SVM":
                prediction = fruit_model_svm.predict(input_data)
            else:  # Random Forest
                prediction = fruit_model_rfc.predict(input_data)
                # Visualize tree
                st.write("### Visualisasi Pohon Keputusan")
                tree = fruit_model_rfc.estimators_[0]
                dot_data = export_graphviz(
                    tree,
                    out_file=None,
                    feature_names=["Diameter", "Weight", "Red", "Green", "Blue"],
                    class_names=list(fruit_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph.source)
            fruit_result = fruit_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Buah: {fruit_result}")
