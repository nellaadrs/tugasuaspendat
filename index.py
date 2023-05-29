import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

st.write("""
# Analisis Sentimen Ulasan Air Terjun Dlundung
### Metode K-NN
"""
)

img = Image.open('dlundung.jpg')
st.image(img, use_column_width=False)

tab_titles = [
    "Homepage",
    "Preposcessing",
    "Wordcloud",]

tabs = st.tabs(tab_titles)

with tabs[0]:
    st.write("""
    Air Terjun Dlundung di Kecamatan Trawas, Kabupaten Mojokerto menjadi salah satu destinasi wisata yang sayang untuk dilewatkan. Panorama alamnya memukau dan udaranya yang sejuk membuat liburan terasa singkat di sini.
    """
    )
    st.write("""
    Dlundung Waterfall berada di area Wana Wisata Dlundung. Tepatnya di Dusun/Desa Ketapanrame, Kecamatan Trawas. Wisata alam ini berjarak sekitar 38 kilometer dengan waktu tempuh sekitar 1 jam 10 menit dari Kota Mojokerto. Jika dari Kantor Kecamatan Trawas, Air Terjun Dlundung hanya sekitar 2,4 kilometer.
    """
    )
    st.write("""
    Air Terjun Dlundung mempunyai panorama alam yang eksotis. Karena lokasinya di antara hutan lereng Gunung Welirang yang lumayan lebat. Banyak pohon besar yang tumbuh di sekitarnya. Tak ayal, udara di wisata alam ini terasa sejuk. Sehingga, cocok untuk melepas penat bersama teman, keluarga atau kekasih tercinta.
    """
    )
    st.write("""
    Ketinggian Air Terjun Dlundung sekitar 14 meter. Air yang berjatuhan dari tebing tak terlalu deras. Sehingga, para pengunjung aman untuk bermain di bawahnya. Selain bermain air, para wisatawan juga bisa berswafoto di bawahnya. Dua spot foto yang tak kalah menarik di atas panggung sisi kanan air terjun dan di depan nama Dlundung Waterfall. Spot selfie juga bisa dijumpai di area parkir Air Terjun Dlundung. Yaitu dengan latar belakang nama Dlundung Waterfall dan lebatnya hutan di atasnya. Akses dari area parkir ke air terjun berupa tangga beton dan jalan paving di tengah rindangnya pepohonan.
    """
    )


with tabs[1]:
    st.write("""
    Data yang diambil dalam projek ini diambil dari Ulasan Air Terjun Dlundung di Google Maps
    """
    )

    data = pd.read_csv('https://raw.githubusercontent.com/DiahDSyntia/ProjekAkhir6/main/datawisata.csv')
    st.write("Data Cancer (https://raw.githubusercontent.com/DiahDSyntia/Data-Mining/main/dataR2.csv) ",data)
    #ukuran data
    data.shape

    #cek data kosong
    data.isnull().sum()

    data.info()

    #drop data kosong
    data.dropna(inplace=True)

    data.isnull().sum()

    #PREPROCESSING
    #hapus karakter
    def delete_char(text):
        text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")
        text = text.encode('ascii', 'replace').decode('ascii')
        return text.replace("http://"," ").replace("https://", " ")
        return text.replace("https://","").replace("http://","")
    data["Ulasan"]=data["Ulasan"].apply(delete_char)
    # st.write("Dibawah ini adalah data yang karakternya sudah dihapus",data)

    #ubah huruf kecil
    def change_var(text):
        text = text.lower()
        return text
    data["Ulasan"]=data["Ulasan"].apply(change_var)
    # data

    from string import punctuation

    #hapus tanda hubung
    def remove_punctuation(text):
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
        return text
    data["Ulasan"]=data["Ulasan"].apply(remove_punctuation)
    data

    #menormalisasikan kata tidak baku (hash)
    normalize = pd.read_csv("https://raw.githubusercontent.com/DiahDSyntia/ProjekAkhir6/main/datawisata.csv")
    normalize_word_dict={}

    for row in normalize.iterrows():
        if row[0] not in normalize_word_dict:
            normalize_word_dict[row[0]] = row[1]

    def normalized_term(comment):
        return [normalize_word_dict[term] if term in normalize_word_dict else term for term in comment]

    data['comment_normalize'] = data['hasil'].apply(normalized_term)
    data['comment_normalize'].head(10)

with tabs[2]:
    st.write("""
    Gambar dibawah ini merupakan wordcloud, hal ini menunjukkan jika tulisan di gambar semakin besar maka kata tersebut sering muncul di ulasan 
    Air terjun Dlundung di Google Maps. Begitu pula sebaliknya.
    """
    )

    from wordcloud import WordCloud #This line here
    allWords = ' '.join([twts for twts in data['Ulasan']])
    wordCloud = WordCloud(width=1600, height=800, random_state=30, max_font_size=200, min_font_size=20).generate(allWords)

    plt.figure( figsize=(20,5), facecolor='k')
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
