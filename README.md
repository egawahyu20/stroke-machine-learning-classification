# Stroke Classification - Ega Wahyu Cahyono

## Domain Proyek

Menurut Organisasi Kesehatan Dunia (WHO) stroke adalah penyebab kematian ke-2 secara global, bertanggung jawab atas sekitar 11% dari total kematian. Oleh sebab itu stroke sebaiknya dideteksi sedini mungkin sehingga mendapatkan penananganan sedini mungkin sehingga mengurangi kemungkinan kematian karena stroke[1]. Teknologi sekarang mempunyai kemampuan untuk membantu manusia untuk menyelesaikan berbagai masalah salah satunya adalah _machine learning_ yang bisa diaplikasikan pada kasus ini untuk memprediksi kemungkinan seseorang terkena stroke.

## Business Understanding

### Problem Statements

- Dari fitur yang ada manakah fitur yang paling menunjukkan bahwa orang tersebut berpotensi terkena stroke?
- Algoritma apa yang memberikan hasil akurasi dan _recall_ tinggi untuk klasifikasi stroke?

### Goals


- Mengetahui fitur yang berkorelasi dengan stroke
- Mendapatkan algoritma terbaik untuk klasifikasi stroke


### Solution statements
- Menggunakan dua atau lebih algoritma untuk mencapai nilai akurasi dan _recall_ tertinggi.
- Melakukan peningkatan pada _baseline_ model dengan _hyperparameter tuning_ dan _cross validation_.

## Data Understanding
Penulis menggunakan _Dataset_ stroke yang ada di kaggle milik [FEDESORIANO](https://www.kaggle.com/fedesoriano). Dataset [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) ini berisi 5110 baris data dengan 12 fitur didalamnya.


### Variabel-variabel pada Kaggle - Stroke Prediction Dataset adalah sebagai berikut:

- id: Identifier data
- gender: Menunjukkan jenis kelamin pasien
- age: Umur dari pasien
- hypertension: Menunjukkan apakah pasien mempunyai penyakit darah tinggi (boolean)
- heart_disease: Menunjukkan apakah pasien mempunyai penyakit jantung (boolean)
- ever_married: Menjukkan apakah pernah menikah
- work_type: Pekerjaan pasien
- Residence_type: Tipe tempat tinggal 
- avg_glucose_level: rata-rata glukosa dalam darah
- bmi: body mass index
- smoking_status: Tipe perokok pasien
- stroke: Stroke (boolean)
*Note: "Unknown" di smoking_status berarti informasi tidak tersedia

## Exploratory Data Analysis:
- Melakukan descriptive analysis.
    Pada Tahap ini akan melihat jumlah data, fitur yang ada di dalamnya dan juga tipe data dari setiap fitur seperti pada tabel 1.
    
    | **No** | **Fitur**         | **Jumlah data** | **Type** |
    |--------|-------------------|-----------------|----------|
    | 1      | id                | 5110            | int64    |
    | 2      | gender            | 5110            | object   |
    | 3      | age               | 5110            | float    |
    | 4      | hypertension      | 5110            | int64    |
    | 5      | heart_disease     | 5110            | int64    |
    | 6      | ever_married      | 5110            | object   |
    | 7      | work_type         | 5110            | object   |
    | 8      | residence_type    | 5110            | object   |
    | 9      | avg_glucose_level | 5110            | float64  |
    | 10     | bmi               | 4909            | float64  |
    | 11     | smoking_status    | 5110            | object   |
    | 12     | stroke            | 5110            | int64    |
    
    tabel 1. info dataframe

- Cek dan menangani data yang hilang.
    Pada tahap ini dilakukan pengecekan data kosong yang ternyata ditemukan 201 data kosong sehingga akan dilakukan penanganan dengan mengisi data kosong tersebut, sebelum itu dilakukan pengecekan outliers menggunakan boxplot yang ditunjukkan pada gambar 1 pada kolom kosong tersebut sehingga data kosong akan diisi dengan median dari data.
    
    ![gambar1](https://drive.google.com/uc?id=1AFDQ1UEHPf-WoPFNidDtlG7WgXAE_GtX)

    gambar 1. Boxplot fitur BMI

- Univariate analysis untuk fitur _numerical_ dan _categorical_.
    Pada tahap ini dilakukan pengcekan setiap fitur dan melihat kecenderungan dari setiap data.

    - Jenis Kelamin

      Ternyata pada data yang diambil dominan bergender perempuan ditunjukkan pada gambar 2.
      
      ![gambar2](https://drive.google.com/uc?id=1kR7WNVACEZIUP9zDbB0TJMomjr1J-G9b)

      gambar 2. Bar Chart jenis kelamin

    - Status Pernikahan
      
      Dari data yang diambil pasien dominan pernah menikah seperti terlihat pada gambar 3.
      
      ![gambar3](https://drive.google.com/uc?id=1vemU3Tf7ZAIEkaMcBr6IWy1lquBejLZs)
      
      gambar 3. Bar Chart Status Pernikahan 

    - Jenis Pekerjaan
      
      Ternyata pada gambar 4 banyak pasien yang merahasiakan pekerjaannya.
      
      ![gambar4](https://drive.google.com/uc?id=1pJAhPIjwmRYrB6fiCgzmlpQ1iyxc0Rwi)

      gambar 4. Bar Chart Jenis Pekerjaan

    - Jenis Tempat Tinggal 
      
      Pada gambar 5 ditunjukkan bahwa pasien hambir seimbang antara yang tinggal di perkotaan dan pedesaan.
      
      ![gambar5](https://drive.google.com/uc?id=1j9J9o3NO9xi1cc1ZrQGULkuSDEKt-9rX)

      gambar 5. Bar Chart Jenis Tempat Tinggal

    - Status Perokok
      
      Pada gambar 6 ditunjukkan bahwa pasien lebih banyak yang tidak pernah merokok, tapi pasien yang status perokoknya tidak diketahui juga banyak.
      
      ![gambar6](https://drive.google.com/uc?id=1YXqrXzl_UIffAlg0bXNbyXEKhI0aiKez)

      gambar 6. Bar Chart Status Perokok

    - Darah Tinggi
      
      Hanya sedikit pasien yang mengalami darah tinggi bahkan dibawah 1000 yang ditunjukkan pada gambar 7.

      ![gambar7](https://drive.google.com/uc?id=1xqWLfhBHikBHY2kG1MSP5UE9468ZkUR9)

      gambar 7. Bar Chart Status Darah Tinggi

    - Penyakit Jantung
      
      Pasien dominan tidak memiliki penyakit jantung seperti yang diperlihatkan pada gambar 8.

      ![gambar8](https://drive.google.com/uc?id=14NpbOoazOGUIRXg_IVudx-JNKhkwcbx3)

      gambar 8. Bar Chart Penyakit Jantung

    - Stroke
      
      Ternyata data sangat tidak seimbang yang ditunjukkan oleh gambar nomor 9, pasien yang mengalami stroke sangat sedikit sehingga pada tahap _data preparation_ nanti diperlukan proses untuk menyeimbangkan data agar model tidak mengalami _bias_ dan hanya meprediksi pasien tidak terkena stroke.

      ![gambar9](https://drive.google.com/uc?id=1YzQDgT_LiaWAbu-xSBBpyWBMDMFVDft6)

      gambar 9. Bar Chart Stroke

    * Data Numerik

      Pada data numerik ditunjukkan bahwa persebaran umur cukup luas, dan juga untuk data glukosa dan bmi cenderung persebarannya ke kiri seperti yang terlihat pada gambar 10.

      ![gambar10](https://drive.google.com/uc?id=1eDZJg5OzswCxTxsqKoKzjziawTNPKwVA)

      gambar 10. Histogram data numerik 

- Multivariate analysis untuk mengetahui korelasi antar fitur.
    Pada tahap ini dilakukan pengecekan korelasi dari tiap fitur, terutama terhapa fitur target stroke, sehingga bisa ditentukan fitur-fitur apa yang paling mempengaruhi seseorang dinyatakan stroke.

    Ternyata umur adalah faktor yang mempunyai korelasi paling kuat dengan stroke diikuti fitur darah tinggi, penyakit jantung dan glukosa lebel yang ditunjukkan pada gambar 11.

    ![gambar11](https://drive.google.com/uc?id=1qHGvTdy_w7SY4Ev2EUZWOmJTLS5TLryY)

    gambar 11. Korelasi Fitur

    

## Data Preparation
Pada tahap ini dilakukan persiapan data sebelum dilakukan tahap selanjutnya yaitu modeling. hal yang dilakukan pada tahap ini adalah:

- _Feature Encoding_: akan merubah data _categorical_ menjadi _numeric_ dengan _one hot encoding_, karena _machine learning_ tidak bisa melakukan _train_ pada data _categorical_ sehingga perlu dirubah dahulu.

  setelah dilakukan _feature encoding_ kemudian dicek lagi korelasi setiap fitur didapatkan hasil bahwa 4 fitur yang berkorelasi kuat dengan stroke adalah umur, darah tinggi, penyakit jantung, dan glukosa level seperti yang terlihat pada gambar 12 sehingga 4 data ini yang akan digunakan sebagai fitur untuk melakukan _training_
  
  ![gambar12](https://drive.google.com/uc?id=1Jzjmn1Vnzm2S7hS-xXUGgCer_8uV3D1q)

  gambar 12. Korelasi Fitur setelah _Feature Encoding_
  
- _Data Balancing_: pada tahap EDA ternyata diketahui bahwa target/label data yang akan digunakan tidak seimbang maka perlu dilakukan penyeimbangan data agar model tidak bias saat melakukan training yang akan berakibat pada prediksi yang dikeluarkan.
  
- _Split Data_: pada tahap ini akan membagi data menjadi 2 yaitu untuk _trainig_ dan _test_, data _train_ akan digunakan untuk pelatihan model dan data _test_ akan digunakan untuk evaluasi model yang telah dilatih. Dengan pembagian 80% untuk data _training_ dan 20% data _test_ dari total 4861 data yang sudah melewati tahap _Feature Encoding_ dan _Data Balancing_.
  
- _Scaling Data_: setelah semua tahap dilalui dan ternyata mendapatkan 4 fitur yang akan digunakan untuk train diketahui bahwa 2 fitur yaitu darah tinggi dan penyakit jantung adalah _boolean_ dan 2 fitur lainnya yaitu umur dan glukosa level adalah numerical, sehingga perlu dilakukan _scaling_ untuk data _numerical_ agar skala data menjadi seimbang sehingga pelatihan model akan lebih optimal.


## Modeling
### Baseline Model
- Pada tahap ini akan menggunakan 5 model dasar untuk dilakukan _training_ dengan menggunakan nilai parameter default setiap model **tanpa merubah atau mengisi parameter apapun**.
  
- 5 model yang digunakan adalah _Decision Tree_, _K-Nearest Neighbour_, _Naive Bayes_, _Logistic Regression_, dan _Random Forest_ yang akan dilakukan training pertama dan melihat hasilnya
  
  *  _Decision Tree_ adalah jenis algoritma klasifikasi yang strukturnya mirip seperti sebuah pohon yang memiliki akar, ranting, dan daun. Simpul akar (internal node) mewakili fitur pada dataset, simpul ranting (branch node) mewakili aturan keputusan (decision rule), dan tiap-tiap simpul daun (leaf node) mewakili hasil keluaran. Itulah kenapa algoritma ini disebut Decision tree atau pohon keputusan. parameter: criterion("gini", "entropy", "log_loss"), splitter("best", "random"), dll.

  *  _K-Nearest Neighbor_ merupakan algoritma yang melakukan klasifikasi berdasarkan kedekatan jarak suatu data dengan data yang lain. Dekat atau jauh suatu jarak dihitung berdasarkan jarak Euclidean. KNN merupakan salah satu algoritma non parametrik yang digunakan dalam pengklasifikasian. Selain naive bayes, algoritma KNN juga menjadi algoritma pengklasifikasian yang terkenal dengan tingkat keakuratan yang baik. parameter: n_neighbors, weights("uniform", "distance"), dll.

  *  _Naive Bayes_ merupakan metode pengklasifikasian paling populer digunakan dengan tingkat keakuratan yang baik. Banyak penelitian tentang pengklasifikasian yang telah dilakukan dengan menggunakan algoritma ini. Berbeda dengan metode pengklasifikasian dengan logistic regression ordinal maupun nominal, pada algoritma naive bayes pengklasifikasian tidak membutuhkan adanya pemodelan maupun uji statistik.  Naive bayes merupakan metode pengklasifikasian berdasarkan probabilitas sederhana dan dirancang agar dapat dipergunakan dengan asumsi antar variabel penjelas saling bebas (independen). Pada algoritma ini pembelajaran lebih ditekankan pada pengestimasian probabilitas. Keuntungan algoritma naive bayes adalah tingkat nilai error yang didapat lebih rendah ketika dataset berjumlah besar, selain itu akurasi naive bayes dan kecepatannya lebih tinggi pada saat diaplikasikan ke dalam dataset yang jumlahnya lebih besar. parameter: var_smoothing, dll.
  

  *  _Logistic Regression_ algoritma supervised learning yang digunakan untuk memperkirakan nilai diskrete (biasanya nilai biner seperti 0/1, ya/tidak, benar//salah) dari sekumpulan variabel independen dengan memprediksi probabilitas suatu peristiwa dengan menyesuaikan data ke fungsi logit. parameter: penalty("l1", "l2", "elasticnet", "none"), dll.


  *  _Random Forest_ algoritma yang digunakan untuk pengklasifikasian dataset dalam jumlah besar. Klasifikasi random forest dilakukan melalui penggabungan tree dengan melakukan training dataset yang kamu miliki. Selain itu, algoritma random forest menggunakan algoritma decision tree untuk melakukan proses seleksi. Dimana tree atau pohon yang dibangun dibagi secara rekursif dari data pada kelas yang sama. Proses klasifikasi pada random forest berawal dari memecah data sampel yang ada dalam decision tree secara acak. Setelah pohon terbentuk,maka akan dilakukan voting pada setiap kelas dari data sampel. Kemudian, mengkombinasikan vote dari setiap kelas kemudian diambil vote yang paling banyak.Dengan menggunakan random forest pada klasifikasi data maka, akan menghasilkan vote yang paling baik. Pada saat proses klasifikasi selesai dilakukan, inisialisasi dilakukan dengan sebanyak data berdasarkan nilai akurasinya. parameter: n_estimators, criterion"gini", "entropy", "log_loss"), dll.


### Cross Validation dan Hyper Parameter Tuning
 
 Setelah melatih model _baseline_ dan mendapatkan model paling baik kemudian melakukan _train_ pada model tersebut dengan dilakukan _Cross validation_ dan _Hyper Parameter tuning_ untuk mendapatkan parameter optimal dari suatu model.

  Pada kasus kali ini model _random forest_ mendapatkan nilai tertinggi untuk setiap metrik, jadi untuk tahap ini model _random forest_ akan dilakukan _Cross validation_ dan _Hyper Parameter Tuning_ untuk parameter n_estimator dan _criteon_. kemudian didapatkan nilai optimal untuk parameter _criterion_ dengan nilai _entropy_ dan untuk parameter n_estimators bernilai 130.


## Evaluation
Setelah melatih model maka akan menuju tahap selanjutnya yaitu adalah evaluasi, pada tahap ini model akan dievaluasi tingkat Accuracy, Recall, Precision dan F1-Score.

* Akurasi adalah akurasi nilai yang didapatkan dari jumlah data bernilai positif yang diprediksi positif dan data bernilai negatif yang diprediksi negatif dibagi dengan jumlah seluruh data di dalam dataset.
  $$ Accuracy = {TP + TN \over TP + FP + FN + TN} $$

* Recall adalah rasio kasus dengan prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
  $$ Recall = {TP \over TP + FN} $$

* Precision adalah adalah rasio kasus yang diprediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positif.
  $$ Precision = {TP \over TP + FP} $$

* F1-Score Nilai atau dikenal juga dengan nama F-Measure didapatkan dari perbandingan rata-rata presisi dengan recall yang dibobotkan.
  $$ F1 = {2 x  Recall x Precision \over Recall + Precision} $$

Pada kasus model kali ini adalah model harus dapat mendeteksi storke dengan baik sehingga kita memilih algoritma yang memprediksi pasien terkena strok tetapi sebenarnya negatif stroke daripada algoritma mendeteksi pasien tidak kena stroke tapi sebenarnya positif stroke. maka pada tahap ini akan lebih diprioritaskan model yang memiliki nilai Recall yang tinggi.

|      **Model**      | **Accuracy** | **Recall** | **Precision** | **F1-Score** |
|:-------------------:|:------------:|:----------:|---------------|--------------|
|    Decision Tree    |    89.92%    |   89.92%   |     89.92%    |    89.92%    |
| K-Nearest Neighbour |    85.86%    |   85.86%   |     86.54%    |    85.79%    |
|     Naive Bayes     |    74.86%    |   74.86%   |     75.20%    |    74.76%    |
| Logistic Regression |    76.66%    |   76.66%   |     76.73%    |    76.64%    |
|    Random Forest    |    91.88%    |   91.88%   |     91.88%    |    91.88%    |

tabel 2. hasil model _baseline_


dari hasil pengujian tabel 2, model _Random Forest_ mendapatkan nilai recall yang tinggi sehingga dilakukan _Hyper Parameter Tuning_ pada paramters n_estimator dan criterion, sehingga berikut hasilnya: 

|              **Model**             | **Accuracy** | **Recall** | **Precision** | **F1-Score** |
|:----------------------------------:|:------------:|:----------:|---------------|--------------|
| Random Forest |    92.44%    |   92.44%   |     92.45%    |    92.44%    |

tabel 3. hasil _parameter tuning model Random Forest_

### Kesimpulan
Dari hasil evaluasi, model random forest yang sudah mempunyai nilai akurasi dan _recall_ paling bagus diantara ke 5 model bisa dilihat pada tabel 2, ketika dilakukan _Hyper Parameter Tuning_ hasilnya adalah metrik _accuracy_, _recall_ dan _f1-score_ naik 0.56% dan untuk metrik _precision_ naik 0.57%.


**REFERENCES**
[1] Charles D A Wolfe, The impact of stroke, British Medical Bulletin, Volume 56, Issue 2, 2000, Pages 275–286, https://doi.org/10.1258/0007142001903120