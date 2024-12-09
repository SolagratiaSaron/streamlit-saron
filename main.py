import base64
import streamlit as st 
import numpy as np

import matplotlib.pyplot as plt
import os
import pandas as pd

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Algoritma knn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import extra_streamlit_components as stx 
from streamlit_option_menu import option_menu
from PIL import Image
from textblob import TextBlob
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle

from google_play_scraper import app
from sklearn.tree import DecisionTreeClassifier


# Download stopwords from NLTK
nltk.download('stopwords')

#======================================================================================================

st.sidebar.title('Analisis Sentimen E-Commerce di Indonesia Untuk UMKM')


with st.sidebar:
    selected = option_menu("", ["Home", 'Top 5 E-commerce', "Import File CSV"], 
        icons=['house', 'list-task', 'cloud-upload'], menu_icon="", default_index=0)
    selected


if selected =='Home':
    
    # Struktur antarmuka
    st.title("🔍 Analisis Sentimen Ulasan Aplikasi E-Commerce untuk Membantu UMKM Memilih Platform yang Tepat")
    st.subheader("📊 Mengungkap Wawasan dari Ulasan Pengguna")

    # Membagi antarmuka menjadi dua kolom
    col1, col2 = st.columns([1, 2])

    with col1:
        # Menampilkan gambar ilustrasi
        image_path = "gbrhome.webp"  # Ubah path ke lokasi gambar Anda
        st.image(
            image_path,
            
            
        )

    with col2:
        # Menampilkan deskripsi aplikasi
        st.markdown(
            """
            <div class="feature-list">
            🚀 Aplikasi Analisis Sentimen E-Commerce dirancang untuk membantu Anda memahami persepsi pelanggan terhadap layanan Anda. Dengan dukungan K-Nearest Neighbor (KNN) dan visualisasi interaktif, aplikasi ini memberikan wawasan penting untuk mendukung keputusan bisnis Anda. Aplikasi ini sangat cocok untuk membantu UMKM (Usaha Mikro, Kecil, dan Menengah) dalam memahami preferensi suatu platform E-commerce tanpa memerlukan keahlian teknis mendalam.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Garis pemisah untuk estetika
    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

    # Menampilkan fitur utama
    st.markdown(
        """
        ### 💡 **Fitur Utama**:
        #### 1. 🛠️ **Top 5 E-commerce**
        - **Deskripsi:** Melakukan analisis sentimen dari file CSV ulasan platform yang telah tersedia.  
        - **Langkah-Langkah:**  
        1. Pilih salah satu file CSV dari ulasan platform seperti Shopee, Tokopedia, Bukalapak, Blibli, atau Lazada.  
        2. Klik tombol **Mulai Analisis** untuk memulai proses data.  
        3. Analisis dilakukan menggunakan **K-Nearest Neighbor (KNN)**.

        #### 2. 📊 **Import file CSV**
        - **Deskripsi:** Memproses ulasan pengguna untuk menentukan sentimen **positif**, **negatif**, atau **netral**.  
        - **Langkah-Langkah:**  
        1. Unggah file CSV berisi data ulasan yang ingin dilakukan analisis.  
        2. Aplikasi akan memproses data dengan teknik preprocessing seperti tokenisasi dan stemming.  
        3. Analisis dilakukan menggunakan **K-Nearest Neighbor (KNN)**.  

        #### 3. 📈 **Visualisasi Data**
        - **Deskripsi:** Menampilkan hasil analisis dalam bentuk visualisasi yang mudah dipahami.  
        - **Langkah-Langkah:**  
        1. Pilih file CSV dengan hasil analisis sentimen.  
        2. Tampilkan distribusi sentimen menggunakan **pie chart**.  
        3. Analisis kata-kata penting dalam ulasan dengan **word cloud** untuk memahami pola pengguna.
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
    """
    ### 💡 **Analisis sentimen Satu kalimat**:
    """
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







    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
elif selected =='Top 5 E-commerce':
    selected3 = option_menu(None, ["Processing & Model Evaluasi", "Visualisasi Data"], 
        default_index=0, orientation="horizontal")
    
    if selected3 =='Processing & Model Evaluasi':
        st.markdown("""
        ### **Langkah-Langkah Analisis Data**
        1. **Pilih salah satu File CSV akan dianalisis dari Top 5 E-commerce.
        2. **Preprocessing Teks:** Teks akan diproses untuk menghilangkan karakter khusus, stopwords, tokenizing, dan stemming.
        3. **Analisis Sentimen:** Menggunakan lexicon untuk menentukan polaritas sentimen (positif, negatif, netral).
        4. **Pembangunan Model KNN:** Melatih model K-Nearest Neighbor untuk klasifikasi sentimen.
        5. **Evaluasi Model:** Menampilkan metrik seperti akurasi, precision, recall, dan F1-score.
        """)
        st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
        
        st.subheader("Top 5 E-Commerce di Indonesia")
        working_dir = os.path.dirname(os.path.abspath(__file__))

        folder_path = f"{working_dir}/data"

        files_list = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        selected_file = st.selectbox("Select a file Top 5 E-Commerce di Indonesia", files_list, index=None)

        if selected_file:
            file_path = os.path.join(folder_path, selected_file)
            df = pd.read_csv(file_path)
            st.write("Nama File:", selected_file)
            st.write(df)

            #preproccesing text
            #cleaning 
            if st.button("Mulai Analisis"):
              
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                st.write("Start Pre-processing")
                st.caption("|case folding...")
                st.caption("|stopword...")
                st.caption("|tokenizing...")
                st.caption("|stemming...")
                #polarity andd labeling
                

                if file_path:
                    pd.set_option('display.max_columns', None)
                    df = df[['content', 'score']]

                    #CASE FOLDING
                    df['text_casefolding'] = df['content'].str.lower()
                    df['text_casefolding'] = df['text_casefolding'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
                    # remove numbers
                    df['text_casefolding'] = df['text_casefolding'].apply(lambda elem: re.sub(r"\d+", "", elem))
                    
                    #STOPWORD REMOVAL
                    stop = stopwords.words('english')
                    df['text_StopWord'] = df['text_casefolding'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
                        
                    #TOKENIZING
                    df['text_tokens'] = df['text_StopWord'].apply(lambda x: word_tokenize(x))

                    #-----------------STEMMING -----------------
                    # create stemmer
                    factory = StemmerFactory()
                    stemmer = factory.create_stemmer()
                    def stemmed_wrapper(term):
                        return stemmer.stem(term)
                    term_dict = {}
                    hitung=0
                    for document in df['text_tokens']:
                        for term in document:
                            if term not in term_dict:
                                term_dict[term] = ' '
                    print(len(term_dict))
                    print("------------------------")
                    for term in term_dict:
                        term_dict[term] = stemmed_wrapper(term)
                        hitung+=1
                        print(hitung,":",term,":" ,term_dict[term])
                    print(term_dict)
                    print("------------------------")
                    # apply stemmed term to dataframe
                    def get_stemmed_term(document):
                        return [term_dict[term] for term in document]
                    #script ini bisa dipisah dari eksekusinya setelah pembacaaan term selesai
                    df['text_steamengl'] = df['text_tokens'].apply(lambda x:' '.join(get_stemmed_term(x)))

                    #polarity andd labeling
                    lexicon_positive = dict()
                    import csv
                    with open('lexicon_positive_ver1.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            lexicon_positive[row[0]] = int(row[1])

                    lexicon_negative = dict()
                    import csv
                    with open('lexicon_negative_ver1.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            lexicon_negative[row[0]] = int(row[1])

                    def sentiment_analysis_lexicon_indonesia(text):
                        score = 0
                        for word_pos in text:
                            if (word_pos in lexicon_positive):
                                score = score + lexicon_positive[word_pos]
                        for word_neg in text:
                            if (word_neg in lexicon_negative):
                                score = score + lexicon_negative[word_neg]
                        polarity=''
                        if (score > 0):
                            polarity = 'positif'
                        elif (score < 0):
                            polarity = 'negatif'
                        else:
                            polarity = 'netral'
                        return score, polarity
                    
                    results = df['text_tokens'].apply(sentiment_analysis_lexicon_indonesia)
                    results = list(zip(*results))
                    df['polarity_score'] = results[0]
                    df['polarity'] = results[1]

                    st.write("Finish Pre-processing")
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                    st.write("Count Polarity and Labeling...")
                    st.caption("using indonesia sentiment lexicon")
                    st.write(df['polarity'].value_counts())
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                    st.write(df)
                    # Fitur unduhan untuk CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Unduh sebagai CSV",
                        data=csv,
                        file_name="ulasan_google_play.csv",
                        mime="text/csv"
                    )
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

                    st.subheader("KNN Accuracy")

                    X_train, X_test, y_train, y_test = train_test_split(df['text_steamengl'], df['polarity'],
                                                        test_size = 0.2,
                                                        random_state = 0)
                    
                    tfidf_vectorizer = TfidfVectorizer()
                    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
                    tfidf_test = tfidf_vectorizer.transform(X_test)

                    vectorizer = CountVectorizer()
                    vectorizer.fit(X_train)

                    X_train = vectorizer.transform(X_train)
                    X_test = vectorizer.transform(X_test)

                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(X_train, y_train)

                    y_pred = knn.predict(tfidf_test)

                    accuracy = accuracy_score(y_test, y_pred)

                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(X_train, y_train)
                    predicted = knn.predict(X_test)

                    st.write("KNeighborsClassifier Accuracy:", accuracy_score(y_test,predicted))
                    st.write("KNeighborsClassifier Precision:", precision_score(y_test,predicted, average="micro", pos_label="Negatif"))
                    st.write("KNeighborsClassifier Recall:", recall_score(y_test,predicted, average="micro", pos_label="Negatif"))
                    st.write("KNeighborsClassifier f1_score:", f1_score(y_test,predicted, average="micro", pos_label="Negatif"))
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

                    st.text(f'Confusion Matrix:\n {confusion_matrix(y_test, predicted)}')
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                    st.code('Model Report:\n ' + classification_report(y_test, predicted, zero_division=0))

    #--------
    
    elif selected3 =='Visualisasi Data':
        # Opsi untuk memilih file CSV
        st.title("Visualisasi Data Sentimen")

        st.markdown("Silahkan masukan file CSV yang di-download dari Pre-Processing & Labeling.")
        uploaded_file = st.file_uploader("Choose CSV file")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data yang diunggah:")
            st.write(df)
            
            # Memastikan kolom yang diperlukan ada
            if 'polarity' in df.columns and 'text_steamengl' in df.columns:
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

                # Pilihan untuk melihat dokumen sentimen tertentu
                st.subheader("Ingin Lihat Data Apa?")
                lihat_sentiment = st.radio(
                    "",
                    ["Positif", "Negatif", "Netral"],
                    index=None,
                )

                if lihat_sentiment == 'Positif':
                    st.write("Dokumen dengan sentimen positif:")
                    st.write(df[df['polarity'] == 'positif'])

                elif lihat_sentiment == 'Negatif':
                    st.write("Dokumen dengan sentimen negatif:")
                    st.write(df[df['polarity'] == 'negatif'])

                elif lihat_sentiment == 'Netral':
                    st.write("Dokumen dengan sentimen netral:")
                    st.write(df[df['polarity'] == 'netral'])

                # Pie chart distribusi sentimen
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                st.subheader("Distribusi Sentimen")
                
                positif_count = df[df['polarity'] == 'positif'].shape[0]
                negatif_count = df[df['polarity'] == 'negatif'].shape[0]
                netral_count = df[df['polarity'] == 'netral'].shape[0]

                labels = ['positif', 'negatif', 'netral']
                sizes = [positif_count, negatif_count, netral_count]
                colors = ['#66bb6a', '#ef5350', '#fffd80']

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')
                p = plt.gcf()
                p.gca().add_artist(plt.Circle((0, 0), 0.7, color='white'))
                st.write("PIE CHART")
                st.pyplot(fig)

            if 'polarity' in df.columns and 'text_steamengl' in df.columns:
                # Bersihkan data pada kolom 'text_steamengl'
                df['text_steamengl'] = df['text_steamengl'].fillna('')  # Ganti NaN dengan string kosong
                df['text_steamengl'] = df['text_steamengl'].astype(str)  # Pastikan semua elemen berupa string


                # Word Cloud untuk setiap kategori sentimen
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                st.subheader("Word Cloud untuk Setiap Sentimen")

                    # Word Cloud Sentimen Positif
                text_positive = ' '.join(df[df['polarity'] == 'positif']['text_steamengl'])
                if text_positive:
                    wordcloud_positive = WordCloud(width=800, height=400, background_color='black').generate(text_positive)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud_positive, interpolation='bilinear')
                    ax.axis('off')
                    plt.title('WordCloud Sentimen Positif', color='green')
                    st.pyplot(fig)
                else:
                    st.write("Data sentimen positif kosong, tidak dapat membuat Word Cloud.")

                # Word Cloud Sentimen Negatif
                text_negative = ' '.join(df[df['polarity'] == 'negatif']['text_steamengl'])
                if text_negative:
                    wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(text_negative)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud_negative, interpolation='bilinear')
                    ax.axis('off')
                    plt.title('WordCloud Sentimen Negatif', color='red')
                    st.pyplot(fig)
                else:
                    st.write("Data sentimen negatif kosong, tidak dapat membuat Word Cloud.")

                # Word Cloud Sentimen Netral
                text_netral = ' '.join(df[df['polarity'] == 'netral']['text_steamengl'])
                if text_netral:
                    wordcloud_netral = WordCloud(width=800, height=400, background_color='black').generate(text_netral)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud_netral, interpolation='bilinear')
                    ax.axis('off')
                    plt.title('WordCloud Sentimen Netral', color='gray')
                    st.pyplot(fig)
                else:
                    st.write("Data sentimen netral kosong, tidak dapat membuat Word Cloud.")

            from collections import Counter

            # Bar Chart untuk kata-kata paling sering muncul
            st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
            st.subheader("Grafik Kata-Kata Paling Sering Muncul")

            # Fungsi untuk menghitung kata-kata paling sering muncul dan menampilkan bar chart
            def plot_frequent_words(text_data, title, color):
                # Tokenisasi dan hitung frekuensi
                words = ' '.join(text_data).split()  # Gabungkan semua teks dan pecah menjadi kata
                word_counts = Counter(words)
                most_common = word_counts.most_common(10)  # Ambil 10 kata paling sering muncul

                # Data untuk visualisasi
                words, counts = zip(*most_common)
                
                # Plot bar chart
                fig, ax = plt.subplots()
                ax.bar(words, counts, color=color)
                plt.xticks(rotation=45)
                ax.set_title(title)
                ax.set_xlabel("Kata")
                ax.set_ylabel("Frekuensi")
                st.pyplot(fig)

            # Grafik untuk sentimen positif
            text_positive_list = df[df['polarity'] == 'positif']['text_steamengl'].tolist()
            if text_positive_list:
                plot_frequent_words(text_positive_list, "Kata Paling Sering Muncul - Sentimen Positif", "green")

            # Grafik untuk sentimen negatif
            text_negative_list = df[df['polarity'] == 'negatif']['text_steamengl'].tolist()
            if text_negative_list:
                plot_frequent_words(text_negative_list, "Kata Paling Sering Muncul - Sentimen Negatif", "red")

            # Grafik untuk sentimen netral
            text_netral_list = df[df['polarity'] == 'netral']['text_steamengl'].tolist()
            if text_netral_list:
                plot_frequent_words(text_netral_list, "Kata Paling Sering Muncul - Sentimen Netral", "gray")


            else:
                st.warning("Kolom 'polarity' atau 'text_steamengl' tidak ditemukan dalam data. Pastikan Anda telah melakukan preprocessing dengan benar.")

    else:
            st.info("Silakan unggah file CSV untuk memulai visualisasi.")


    #-------
    
        





    
    
elif selected =='Import File CSV':
    selected4 = option_menu(None, ["Processing & Model Evaluasi", "Visualisasi data"], 
        default_index=0, orientation="horizontal")
    
    if selected4 =='Processing & Model Evaluasi':
        st.markdown("Silahkan masukan file CSV yang memiliki kolom dengan nama 'content' untuk Process text")
        img = Image.open('data/contoh.png')
        st.image(img)
        uploaded_file = st.file_uploader("Choose CSV files")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df)

            #preproccesing text
            #cleaning 
            if st.button("Mulai Analisis"):
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                st.write("Start Pre-processing")
                st.caption("|case folding...")
                st.caption("|stopword...")
                st.caption("|tokenizing...")
                st.caption("|stemming...")
                
                #polarity andd labeling
                

                if uploaded_file:
                    pd.set_option('display.max_columns', None)
                    df = df[['content', 'score']]

                    #CASE FOLDING
                    df['text_casefolding'] = df['content'].str.lower()
                    df['text_casefolding'] = df['text_casefolding'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
                    # remove numbers
                    df['text_casefolding'] = df['text_casefolding'].apply(lambda elem: re.sub(r"\d+", "", elem))
                    
                    #STOPWORD REMOVAL
                    stop = stopwords.words('english')
                    df['text_StopWord'] = df['text_casefolding'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
                        
                    #TOKENIZING
                    df['text_tokens'] = df['text_StopWord'].apply(lambda x: word_tokenize(x))

                    #-----------------STEMMING -----------------
                    # create stemmer
                    factory = StemmerFactory()
                    stemmer = factory.create_stemmer()
                    def stemmed_wrapper(term):
                        return stemmer.stem(term)
                    term_dict = {}
                    hitung=0
                    for document in df['text_tokens']:
                        for term in document:
                            if term not in term_dict:
                                term_dict[term] = ' '
                    print(len(term_dict))
                    print("------------------------")
                    for term in term_dict:
                        term_dict[term] = stemmed_wrapper(term)
                        hitung+=1
                        print(hitung,":",term,":" ,term_dict[term])
                    print(term_dict)
                    print("------------------------")
                    # apply stemmed term to dataframe
                    def get_stemmed_term(document):
                        return [term_dict[term] for term in document]
                    #script ini bisa dipisah dari eksekusinya setelah pembacaaan term selesai
                    df['text_steamengl'] = df['text_tokens'].apply(lambda x:' '.join(get_stemmed_term(x)))

                    #polarity andd labeling
                    lexicon_positive = dict()
                    import csv
                    with open('lexicon_positive_ver1.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            lexicon_positive[row[0]] = int(row[1])

                    lexicon_negative = dict()
                    import csv
                    with open('lexicon_negative_ver1.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            lexicon_negative[row[0]] = int(row[1])

                    def sentiment_analysis_lexicon_indonesia(text):
                        score = 0
                        for word_pos in text:
                            if (word_pos in lexicon_positive):
                                score = score + lexicon_positive[word_pos]
                        for word_neg in text:
                            if (word_neg in lexicon_negative):
                                score = score + lexicon_negative[word_neg]
                        polarity=''
                        if (score > 0):
                            polarity = 'positif'
                        elif (score < 0):
                            polarity = 'negatif'
                        else:
                            polarity = 'netral'
                        return score, polarity
                    
                    results = df['text_tokens'].apply(sentiment_analysis_lexicon_indonesia)
                    results = list(zip(*results))
                    df['polarity_score'] = results[0]
                    df['polarity'] = results[1]

                    st.write("Finish Pre-processing")
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                    st.write("Count Polarity and Labeling...")
                    st.caption("using indonesia sentiment lexicon")
                    st.write(df['polarity'].value_counts())
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                    st.write(df)
                    # Fitur unduhan untuk CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Unduh sebagai CSV",
                        data=csv,
                        file_name="ulasan_google_play.csv",
                        mime="text/csv"
                    )
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

                    st.subheader("KNN Accuracy")

                    X_train, X_test, y_train, y_test = train_test_split(df['text_steamengl'], df['polarity'],
                                                        test_size = 0.2,
                                                        random_state = 0)
                    
                    tfidf_vectorizer = TfidfVectorizer()
                    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
                    tfidf_test = tfidf_vectorizer.transform(X_test)

                    vectorizer = CountVectorizer()
                    vectorizer.fit(X_train)

                    X_train = vectorizer.transform(X_train)
                    X_test = vectorizer.transform(X_test)

                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(X_train, y_train)

                    y_pred = knn.predict(tfidf_test)

                    accuracy = accuracy_score(y_test, y_pred)

                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(X_train, y_train)
                    predicted = knn.predict(X_test)

                    st.write("KNeighborsClassifier Accuracy:", accuracy_score(y_test,predicted))
                    st.write("KNeighborsClassifier Precision:", precision_score(y_test,predicted, average="micro", pos_label="Negatif"))
                    st.write("KNeighborsClassifier Recall:", recall_score(y_test,predicted, average="micro", pos_label="Negatif"))
                    st.write("KNeighborsClassifier f1_score:", f1_score(y_test,predicted, average="micro", pos_label="Negatif"))
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

                    st.code(f'Confusion Matrix:\n {confusion_matrix(y_test, predicted)}')
                    st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                    st.code('Model Report:\n ' + classification_report(y_test, predicted, zero_division=0))

#--------

    elif selected4 =='Visualisasi data':
        

        # Opsi untuk memilih file CSV
        st.title("Visualisasi Data Sentimen")

        st.markdown("Silahkan masukan file CSV yang di-download dari Pre-Processing & Labeling.")
        uploaded_file = st.file_uploader("Choose CSV file")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data yang diunggah:")
            st.write(df)
            
            # Memastikan kolom yang diperlukan ada
            if 'polarity' in df.columns and 'text_steamengl' in df.columns:
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)

                # Pilihan untuk melihat dokumen sentimen tertentu
                st.subheader("Ingin Lihat Data Apa?")
                lihat_sentiment = st.radio(
                    "",
                    ["Positif", "Negatif", "Netral"],
                    index=None,
                )

                if lihat_sentiment == 'Positif':
                    st.write("Dokumen dengan sentimen positif:")
                    st.write(df[df['polarity'] == 'positif'])

                elif lihat_sentiment == 'Negatif':
                    st.write("Dokumen dengan sentimen negatif:")
                    st.write(df[df['polarity'] == 'negatif'])

                elif lihat_sentiment == 'Netral':
                    st.write("Dokumen dengan sentimen netral:")
                    st.write(df[df['polarity'] == 'netral'])

                # Pie chart distribusi sentimen
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                st.subheader("Distribusi Sentimen")
                
                positif_count = df[df['polarity'] == 'positif'].shape[0]
                negatif_count = df[df['polarity'] == 'negatif'].shape[0]
                netral_count = df[df['polarity'] == 'netral'].shape[0]

                labels = ['positif', 'negatif', 'netral']
                sizes = [positif_count, negatif_count, netral_count]
                colors = ['#66bb6a', '#ef5350', '#fffd80']

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')
                p = plt.gcf()
                p.gca().add_artist(plt.Circle((0, 0), 0.7, color='white'))
                st.write("PIE CHART")
                st.pyplot(fig)

            if 'polarity' in df.columns and 'text_steamengl' in df.columns:
                # Bersihkan data pada kolom 'text_steamengl'
                df['text_steamengl'] = df['text_steamengl'].fillna('')  # Ganti NaN dengan string kosong
                df['text_steamengl'] = df['text_steamengl'].astype(str)  # Pastikan semua elemen berupa string


                # Word Cloud untuk setiap kategori sentimen
                st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
                st.subheader("Word Cloud untuk Setiap Sentimen")

                    # Word Cloud Sentimen Positif
                text_positive = ' '.join(df[df['polarity'] == 'positif']['text_steamengl'])
                if text_positive:
                    wordcloud_positive = WordCloud(width=800, height=400, background_color='black').generate(text_positive)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud_positive, interpolation='bilinear')
                    ax.axis('off')
                    plt.title('WordCloud Sentimen Positif', color='green')
                    st.pyplot(fig)
                else:
                    st.write("Data sentimen positif kosong, tidak dapat membuat Word Cloud.")

                # Word Cloud Sentimen Negatif
                text_negative = ' '.join(df[df['polarity'] == 'negatif']['text_steamengl'])
                if text_negative:
                    wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(text_negative)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud_negative, interpolation='bilinear')
                    ax.axis('off')
                    plt.title('WordCloud Sentimen Negatif', color='red')
                    st.pyplot(fig)
                else:
                    st.write("Data sentimen negatif kosong, tidak dapat membuat Word Cloud.")

                # Word Cloud Sentimen Netral
                text_netral = ' '.join(df[df['polarity'] == 'netral']['text_steamengl'])
                if text_netral:
                    wordcloud_netral = WordCloud(width=800, height=400, background_color='black').generate(text_netral)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud_netral, interpolation='bilinear')
                    ax.axis('off')
                    plt.title('WordCloud Sentimen Netral', color='gray')
                    st.pyplot(fig)
                else:
                    st.write("Data sentimen netral kosong, tidak dapat membuat Word Cloud.")

          

            from collections import Counter

            # Bar Chart untuk kata-kata paling sering muncul
            st.markdown("<hr style='border: 1px solid #ecf0f1;'>", unsafe_allow_html=True)
            st.subheader("Grafik Kata-Kata Paling Sering Muncul")

            # Fungsi untuk menghitung kata-kata paling sering muncul dan menampilkan bar chart
            def plot_frequent_words(text_data, title, color):
                # Tokenisasi dan hitung frekuensi
                words = ' '.join(text_data).split()  # Gabungkan semua teks dan pecah menjadi kata
                word_counts = Counter(words)
                most_common = word_counts.most_common(10)  # Ambil 10 kata paling sering muncul

                # Data untuk visualisasi
                words, counts = zip(*most_common)
                
                # Plot bar chart
                fig, ax = plt.subplots()
                ax.bar(words, counts, color=color)
                plt.xticks(rotation=45)
                ax.set_title(title)
                ax.set_xlabel("Kata")
                ax.set_ylabel("Frekuensi")
                st.pyplot(fig)

            # Grafik untuk sentimen positif
            text_positive_list = df[df['polarity'] == 'positif']['text_steamengl'].tolist()
            if text_positive_list:
                plot_frequent_words(text_positive_list, "Kata Paling Sering Muncul - Sentimen Positif", "green")

            # Grafik untuk sentimen negatif
            text_negative_list = df[df['polarity'] == 'negatif']['text_steamengl'].tolist()
            if text_negative_list:
                plot_frequent_words(text_negative_list, "Kata Paling Sering Muncul - Sentimen Negatif", "red")

            # Grafik untuk sentimen netral
            text_netral_list = df[df['polarity'] == 'netral']['text_steamengl'].tolist()
            if text_netral_list:
                plot_frequent_words(text_netral_list, "Kata Paling Sering Muncul - Sentimen Netral", "gray")


            else:
                st.warning("Kolom 'polarity' atau 'text_steamengl' tidak ditemukan dalam data. Pastikan Anda telah melakukan preprocessing dengan benar.")

        else:
            st.info("Silakan unggah file CSV untuk memulai visualisasi.")


        #-------

    









#===================================================##==================================================###









































#---




    
    
