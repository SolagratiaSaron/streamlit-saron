    img = Image.open('data/gbrr.jpg')
    st.image(img)

    def main():
        st.header("Analisis Sentimen E-Commerce dengan Algoritma C4.5 K-Nearest Neighbor")
        st.write("Selamat datang di aplikasi Analisis Sentimen Ulasan Pengguna E-Commerce di Indonesia! Aplikasi ini dapat memungkinkan anda untuk menganalisis sentimen ulasan dari para pengguna aplikasi, dan mendapatkan wawasan berharga tentang bagaimana pengguna menyampaikan perasaan dan perndapat mereka.")
        st.write("")
        st.write("Anda dapat mulai dengan memilih menu 'Top 5 E-Commerce' jika ingin melihat kumpulan data ulasan dari Lazada, Shoppe, Blibli, Tokopedia dan Bukalapak yang diambil dari Google Play Store. atau pilih menu 'Import File CSV' jika anda sudah memiliki file yang ingin dilakukan analisis sentimen. Setelah itu anda dapat melanjutkan dengan melakukan 'Processing & Model Evaluasi' untuk melakukan pembersihan data serta memberikan label sentimen pada setiap ulasan, dan dapat mencoba berbagai model machine learning untuk mengklasifikasikan sentimen dari ulasan dan menganalisis kinerja model tersebut.")
        st.write("")
        st.write("Selanjutnya, Anda dapat menggunakan 'Visualisasi Data' untuk menampilkan visualisasi atau world Cloud, yang dapat membantu anda memahami distribusi sentimen dari ulasan pengguna")
        
        st.write("")
        with open('new_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        # Load the KNN model
        with open('knn_model.pkl', 'rb') as f:
            knn_model = pickle.load(f)

        kalimat = st.text_input("Silahkan masukan satu kalimat untuk melihat Sentimennya:")

        if st.button("Prediksi"):
            # Lakukan proses prediksi di sini
            test_input = vectorizer.transform([kalimat])
            predictions = knn_model.predict(test_input)
            labels = ["negatif", "positif"]
            st.write("""## Sentimen ini bernilai """, labels[predictions[0]])


    if __name__ == "__main__":
        main()
