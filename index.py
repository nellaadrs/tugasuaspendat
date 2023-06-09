#Modul Library
import streamlit as st
import numpy as np
import pandas as pd


#Modul library Metode 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# #modul library data testing dan training
from sklearn.model_selection import train_test_split

# #modul library score tingkat akurasi
from sklearn.metrics import accuracy_score

def load_dataset():
	url = 'https://raw.githubusercontent.com/nellaadrs/tugasuaspendat/main/anemia.csv'
	dataset = pd.read_csv(url,  header='infer', index_col=False)
	return dataset

st.title('Sistem Pendeteksi Anemia')
st.subheader("""
R. Bella Aprilia Damayanti	200411100082

Nella Adrisia Hartono		200411100107

Machine Learning B

""")
deskripsi, dataset, modelling, implementasi = st.tabs(["Info", "Dataset", "Modelling", "Implementasi"])

with deskripsi:
	st.image("https://cdn.slidesharecdn.com/ss_thumbnails/anemia-140408050251-phpapp02-thumbnail-4.jpg?cb=1397124052", width=400)
	st.write("""
	Anemia adalah kondisi ketika tubuh kekurangan sel darah merah yang sehat atau ketika sel darah merah tidak berfungsi dengan baik. 
	Akibatnya, organ tubuh tidak mendapat cukup oksigen sehingga membuat penderita anemia pucat dan mudah lelah.
	Anemia bisa terjadi sementara atau dalam jangka panjang dengan tingkat keparahan ringan sampai berat. 
	Anemia merupakan gangguan darah atau kelainan hematologi yang terjadi ketika kadar hemoglobin 
	(bagian utama dari sel darah merah yang mengikat oksigen) berada di bawah normal.
	"""
	)
	st.write(
	"""
	Jenis Anemia yang umum terjadi berdasarkan penyebabnya:
	1. Anemia akibat kekurangan zat besi
	2. Anemia pada masa kehamilan
	3. Anemia akibat perdarahan
	4. Anemia aplastik
	5. Anemia hemolitik
	6. Anemia akibat penyakit kronis
	7. Anemia sel sabit (sickle cell anemia)
	8. Thalasemia
	"""
	)
	st.write(
	"""
	Penderita anemia bisa mengalami gejala berupa:

	1. Lemas dan cepat lelah
	2. Sakit kepala dan pusing
	3. Sering mengantuk, misalnya mengantuk setelah makan
	4. Kulit terlihat pucat atau kekuningan
	5. Detak jantung tidak teratur
	6. Napas pendek
	7. Nyeri dada
	8. Dingin di tangan dan kaki
	"""
	)


with dataset:
	st.write("""Link Dataset : https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset""")
	("""
	Dataset anemia yang berisi atribut Gender, Hemoglobin, MCHC, MCV, MCH dan Hasil.

	Dataset ini digunakan untuk memprediksi apakah seorang pasien kemungkinan menderita anemia. Algoritma machine learning binary classifier yang akan digunakan.

	Jenis Kelamin: 0 - laki-laki, 1 - perempuan

	Hemoglobin: Hemoglobin adalah protein dalam sel darah merah Anda yang membawa oksigen ke organ dan jaringan tubuh Anda dan mengangkut karbon dioksida dari organ dan jaringan Anda kembali ke paru-paru Anda.
	
	* Kadar hemoglobin normal pada wanita dewasa berkisar antara 12–15 g/dL, sedangkan kadar hemoglobin pada pria dewasa berkisar antara 13–17 g/dL.
	
	* Beberapa penyebab Hb rendah, misalnya kehilangan darah, gangguan fungsi ginjal dan sumsum tulang, paparan radiasi, atau kekurangan nutrisi seperti zat besi, folat, dan vitamin B12.

	MCH: MCH adalah kependekan dari "mean corpuscular hemoglobin." Ini adalah jumlah rata-rata di setiap sel darah merah Anda dari protein yang disebut hemoglobin, yang membawa oksigen ke seluruh tubuh Anda.

	* MCH normal yang biasanya didapatkan pada orang dewasa adalah 27,5-33,2 pg (pikogram).

	* Nilai mean corpuscular hemoglobin yang lebih rendah dari 27,5 dikategorikan sebagai MCH rendah.

	* Nilai mean corpuscular hemoglobin yang lebih tinggi dari 33,2 pg dikategorikan sebagai MCH tinggi.

	MCHC adalah singkatan dari rata-rata konsentrasi hemoglobin corpuscular. Ini adalah ukuran konsentrasi rata-rata hemoglobin di dalam satu sel darah merah.

	* Pada orang dewasa yakni usia 18 tahun ke atas, hasil normal pemeriksaan MCHC berkisar antara 334-355 g/L.

	MCV adalah singkatan dari mean corpuscular volume. Tes darah MCV mengukur ukuran rata-rata sel darah merah Anda.

	* nilai normal MCV berkisar antara 80-100 fL.

	Hasil: 0- tidak anemia, 1-anemia

		""")
# def load_dataset():
# 	url = 'https://raw.githubusercontent.com/nellaadrs/datamining/gh-pages/anemia.csv'
# 	dataset = pd.read_csv(url,  header='infer', index_col=False)
# 	return dataset

	dataset=load_dataset()
	dataset

# with proses:
# 	with st.form("my_form"):
# 		st.write("Form Pendeteksi")
# 		gender = st.selectbox(
# 			'Jenis Kelamin',
# 			('Pilihan','0', '1'))
# 		MCH = st.number_input('MCH')
# 		MCHC = st.number_input('MCHC')
# 		MCV = st.number_input('MCV')
# 		submitted = st.form_submit_button("Submit")
# 		diagnosa = ""
# 		if submitted:
# 			prediksi_anemia = model.predict([["gender", "MCH", "MCHC", "MCV"]])
# 			# prediksi jika anemia = 1 jika tidak sama dengan 0
# 			if prediksi_anemia[0] == 1:
# 				diagnosa = "Terdeksi Penyakit Anemia"
# 			else:
# 				diagnosa = "Tidak terdeksi diagnosa"

# 			st.success(diagnosa)

with modelling:

	def tambah_input(nama_metode): 
		inputan=dict()
		if nama_metode=="K-Nearst Neighbors" :
			K= st.slider("K", 1,15)
			inputan["K"]=K
		elif nama_metode=="Decission Tree":
			kriteria =st.selectbox("pilih kriteria",("entropy", "gini"))
			inputan["kriteria"]= kriteria
			max_depth =st.slider ("max depth",2,15)
			inputan["max_depth"]=max_depth
		return inputan

	def pilih_kelas(nama_metode, inputan):
		model=None
		if nama_metode   == "K-Nearst Neighbors":
			model =KNeighborsClassifier(n_neighbors = inputan["K"])
		elif nama_metode == "Decission Tree":
			model =DecisionTreeClassifier(criterion= inputan["kriteria"], max_depth= inputan["max_depth"])
		elif nama_metode == "Naive Baiyes GaussianNB":
			model = GaussianNB()

		return model

	metode = st.selectbox("Hasil metode akurasi berdasarkan dataset menggunakan:",('Naive Baiyes GaussianNB', 'K-Nearst Neighbors', 'Decission Tree'))
	inputan = tambah_input(metode)
	model 	= pilih_kelas(metode, inputan)

	
	#fitur
	X=dataset.iloc[:,0:5]
	#hasil
	Y=dataset.iloc[:,5]

#Proses Klasifikasi
	#split unnormalized data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size =0.2, random_state=0)
	model.fit(X_train, y_train)
	y_pred=model.predict(X_test)
	accuracy =accuracy_score(y_test,y_pred)
	st.write("Accuracy  = ", accuracy)



with implementasi:
	with st.form("my_form"):
		st.write("Form Pendeteksi")
		gender = st.selectbox(
			'Jenis Kelamin',
			('Male', 'Female'))
		Hemoglobin = st.number_input('Hemoglobin')
		MCH = st.number_input('MCH')
		MCHC = st.number_input('MCHC')
		MCV = st.number_input('MCV')
		submitted = st.form_submit_button("Submit")
		diagnosa = ""

		if gender=='Male':
			gender=0
		else:
			gender=1

		a = np.array([[gender,Hemoglobin, MCH, MCHC, MCV]])
		data_inputan = pd.DataFrame(a, columns= ["gender","Hemoglobin","MCH", "MCHC", "MCV"])

		if submitted:
			prediksi_anemia = model.predict(data_inputan)
			# prediksi jika anemia = 1 jika tidak sama dengan 0
			if prediksi_anemia[0] == 1:
				diagnosa = "Terdeteksi penyakit Anemia"
			else:
				diagnosa = "Tidak terdeksi Penyakit Anemia" 

			st.success(diagnosa)

		



	# option = st.selectbox(
 #    'How would you like to be contacted?',
 #    ('Email', 'Home phone', 'Mobile phone'))
 #    st.write('You selected:', option)

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )
