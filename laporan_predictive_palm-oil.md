# Laporan Proyek Machine Learning - Putri Mentari

## Domain Proyek

Indonesia adalah produsen minyak kelapa sawit terbesar di dunia. Pada tahun 2024/2025, volume produksi minyak kelapa sawit di Indonesia mencapai 46,5 juta metrik ton (Shahbandeh, 2025). Minyak kelapa sawit adalah minyak nabati yang dapat dimakan yang biasa digunakan untuk produk makanan, deterjen, dan kosmetik.  
Berdasarkan penelitian dari Setiaji & Widodo (2013), harga minyak kelapa sawit menentukan permintaan minyak kelapa sawit di masa depan. Di sisi lain, harga minyak kelapa sawit di Indonesia dipengaruhi oleh faktor tingkat suku bunga, nilai tukar mata uang, dan harga komoditas substitusi seperti minyak zaitun dan minyak kedelai (Taib, 2014). Di Malaysia, harga minyak kepala sawit turut dipengaruhi oleh harga minyak mentah dunia (Senadjki, Nathan, & Yong, 2023).  


## Business Understanding

### Problem Statements

Minyak kelapa sawit merupakan komoditas yang strategis terutama bagi masyarakat Indonesia. Harga minyak kelapa sawit juga dapat menentukan pendapatan ekspor negara. Adanya fluktuasi harga minyak kelapa sawit terjadi salah satunya karena meningkatnya permintaan minyak kelapa sawit, yang selaras dengan hukum permintaan. Berdasarkan uraian, perlu dilakukan riset untuk mengetahui hal berikut:  
- Faktor apa saja yang memiliki korelasi terhadap perubahan harga minyak kelapa sawit?

### Goals

- Membuat model machine learning yang dapat memberikan pemahaman terkait faktor-faktor yang berkorelasi dengan perubahan harga palm oil.

### Solution statements
- Melatih beberapa model prediksi dengan machine learning. Hal ini dilakukan untuk mendapatkan model terbaik. Evaluasi yang dilakukan dengan membandingkan nilai MAE dan R2 dari setiap model yang dilatih.


## Data Understanding
Dataset yang digunakan dalam proyek ini adalah dataset harga komoditas global yaitu Palm Oil Future Historical Data yang diperoleh dari https://www.kaggle.com/datasets/datavidia/indonesia-commodity-price

Dataset ini terdiri dari 670 baris dan 7 kolom.

### Variabel-variabel pada Palm Oil Future Historical Data adalah sebagai berikut:
- Date : Tanggal pengamatan data, yaitu 3 Maret 2022 sampai 30 September 2024, dapat digunakan untuk analisis time series  
- Price : Harga palm oil saat penutupan (akhir hari)  
- Open : Harga palm oil saat pembukaan (awal hari)  
- High : Harga tertinggi dalam satu hari perdagangan  
- Low : Harga terendan dalam satu hari perdagangan  
- Vol. : Volume yang menunjukkan jumlah palm oil yang terjual dalam satu hari  
- Change % : Persentasi perubahan harga dari harga di hari sebelumnya

Semua kolom data saat ini bertipe object sehingga selanjutnya perlu dilakukan perubahan tipe data sebelum modeling. Dataset ini juga memiliki 10 missing values pada kolom Vol. dengan nilai unik sebanyak 229. 

Untuk membuat grafik tren fluktuasi harga, data diurutkan berdasarkan Price_temp yang merupakan data salinan sementara dari kolom Price karena kolom Price masih bertipe object.

Grafik ini menunjukkan bahwa pada Q1-Q2 tahun 2022, harga palm oil menunjukkan tren peningkatan. Harga palm oil pada Q3 2022 mengalami penurunan signifikan. Tren ini berlanjut hingga Q3 tahun 2024 dengan fluktuasi harga yang cenderung stagnan.


## Data Preparation
Pada data ini perlu dilakukan perubahan tipe data karena pada data awal semua variabelnya bertipe object.  
- Fitur "Date" diubah menjadi tipe datetime  
- Fitur "Price", "Open", "High", "Low", dan "Change %" diubah menjadi integer  
- Fitur "Vol." diubah menjadi integer dan disesuaikan isi datanya, dari data awal tertulis 1,23K menjadi 1230

Missing values pada kolom Vol. diisi dengan nilai median dari Vol.

Sebelum membuat histogram distribusi data, terlebih dahulu menghapus data Price_temp.

Distribusi nilai untuk variabel Price, Open, High, dan Low cenderung normal, sementara variabel Vol. memiliki distribusi skewed ke kanan dengan dominasi nilai di bawah 1000.

Standarisasi dengan Standar Scaler perlu dilakukan  karena adanya perbedaan skala dalam data sehingga pada saar pelatihan tidak akan berfokus pada data dengan skala yang besar dan memiliki performa yang baik. Proses standarisasi dengan Standar Scaler dilakukan mengurangkan nilai variabel dengan mean dan membaginya dengan standar deviasi.

Adapun pembagian data dilakukan untuk membuat data training dan data test. Pembagian data dilakukan dengan rasio 8:2 yaitu 80% untuk data training (x_train dan y_train) dan 20% untuk data test (x_test dan y_test). Dari 670 data, didapatkan 536 data train dan 134 data test.


## Modeling
Terdapat 3 model yang digunakan di sini, yaitu Linear Regression, Random Forest Regression, dan SVR.

### Linear Regression
Model linear regression mencoba menemukan garis lurus terbaik yang menggambarkan hubungan antara variabel independen (X) dan variabel dependen (Y).  
Model linear regression melatih x_train sebagai input terhadap y_train lalu akan melakukan prediksi terhadap data test.  
Kelebihan:  
- Mudah dipahami karena melibatkan hubungan linear  
- Tidak memerlukan banyak komputasi dan cocok untuk data kecil

Kekurangan:  
- Hanya dapat digunakan untuk variabel independen dan dependen yang memiliki hubungan linear  
- Sensitif terhadap outlier

### Random Forest Regression
Random Forest Regressor adalah algoritma machine learning berbasis ensemble, yang menggunakan banyak pohon keputusan (decision trees) untuk melakukan prediksi nilai numerik (regresi).  
Random Forest Regression akan membuat banyak decision tree dari data training. Kemudian, setiap pohon akan menggunakan data dan fitur acak lalu memperikan prediksinya. Hasil prediksi dari setiap pohon kemudian dirata-ratakan.  
Model random forest regression melatih x_train sebagai input terhadap y_train lalu akan melakukan prediksi terhadap data test. Model ini menggunakan parameter random_state=42 untuk mengontrol randomness data.  
Kelebihan:  
- Stabil untuk data non-linear dan kompleks  
- Tahan terhadap overfitting  
- Tidak memeerlukan scaling data

Kekurangan:  
- Membutuhkan banyak memori jika memiliki banyak pohon

### SVR
SVR mencoba menemukan garis atau hyperplane yang memprediksi nilai output dengan margin error yang diizinkan. Dalam SVR, hanya titik-titik data di luar margin error yang berkontribusi dalam menentukan hyperplane atau disebut dengan support vectors.  
Model SVR melatih x_train sebagai input terhadap y_train lalu akan melakukan prediksi terhadap data test. Model ini menggunakan default parameter, meliputi kernel rbf, C 1.0, ϵ 0,1.  
Kelebihan:  
- SVR dapat menangkap hubungan non-linear antara fitur dan target.  
- parameter ϵ dan C memberikan fleksibilitas dalam mengontrol error dan kompleksitas model.  
- SVR dapat menjadi robust terhadap outliers karena hanya eror di luar margin yang dihitung.

Kekurangan:  
- SVR dengan kernel non-linear bisa menjadi sangat mahal secara komputasi, terutama untuk dataset besar.  
- Memilih parameter yang tepat untuk ϵ, C, dan kernel bisa menjadi sulit dan memerlukan tuning yang ekstensif.  
- Model SVR dengan kernel non-linear kurang intuitif dan lebih sulit diinterpretasikan dibandingkan dengan model regresi linear sederhana.  

Selain dilakukan pelatihan, pada setiap model ini juga dibuat grafik yang membandingkan nilai aktual dan prediksi dari setiap model. Berdasarkan grafik tersebut, model Linear Regression menunjukkan adanya kesesuaian lebih baik antara nilai aktual dan prediksi bila dibandingkan dengan model Random Forest Regression dan SVR. Oleh karena itu, model Linear Regression dapat dipilih menjadi model terbaik untuk memprediksi harga palm oil.


## Evaluation
Pada model regresi dalam proyek ini, terdapat 3 metrik evaluasi yang dinilai, yaitu MAE, RMSE, dan R2.

1. MAE (Mean Absolute Error)  
MAE (mean absolute error) adalah rata-rata dari kesalahan antara nilai aktual dan nilai prediksi. MAE dihitung dengan menjumlahkan semua selisih dari nilai aktual dan prediksi kemudian membaginya dengan jumlah data.

Berikut adalah rumus menghitung MAE  
**MAE = (1/n) * Σ |yᵢ - ŷᵢ|**  
Keterangan:  
n: Jumlah total data (observasi)  
yᵢ: Nilai aktual pada observasi ke-i  
ŷᵢ: Nilai prediksi model pada observasi ke-i

2. RMSE (Root Mean Squared Error)  
RMSE adalah akar dari rata-rata kuadrat selisih antara nilai asli dan prediksi yang biasanya lebih sensitif terhadap error besar. RMSE dihitung dengan mengkuadratkan nilai MAR kemudian mengubahnya dengan nilai akar sehingga lebih mudah diinterpretasikan.

Berikut adalah rumus menghitung RMSE  
**RMSE = sqrt((1/n) * Σ (yᵢ - ŷᵢ)²)**  
Keterangan:  
n: Jumlah total data  
yᵢ: Nilai aktual pada observasi ke-i  
ŷᵢ: Nilai prediksi model pada observasi ke-i

3. R-squared  
R-squared juga dikenal sebagai coefficient of determination adalah salah satu metrik yang digunakan untuk mengevaluasi seberapa baik model regresi linear menjelaskan variasi dalam data. R-squared memberikan ukuran proporsi variasi dalam variabel dependen (output) yang dapat dijelaskan oleh variabel independen (input) dalam model. R2 didapatkan dengan mengurangi nilai 1 dengan hasil pembagian dari jumlah error model dibagi variasi total dalam data.

Berikut adalah rumus R2  
**R² = 1 - [Σ (yᵢ - ŷᵢ)²] / [Σ (yᵢ - ȳ)²]**  
Keterangan:  
yᵢ: Nilai aktual  
ŷᵢ: Nilai prediksi  
ȳ: Rata-rata dari seluruh nilai aktual 


Berikut adalah hasil evaluasi dari ketiga model:  
1. Linear Regression  
- MAE  : 0.0222  
- RMSE : 0.0314  
- R²   : 0.9991  
2. Random Forest  
- MAE  : 0.0345  
- RMSE : 0.0575  
- R²   : 0.9971  
3. SVR  
- MAE  : 0.0473  
- RMSE : 0.0753  
- R²   : 0.9951  

Berdasarkan metrik evaluasi yang ada, diketahui model Linear Regression merupakan model terbaik. Model ini memiliki nilai error paling kecil dan nilai R2 0,9991 yang berarti model ini dapat menjelaskan 99,91% variasi data.

Berdasarkan hasil pada model Linear Regression, diketahui terdapat 4 fitur yang berkorelasi terhadap perubahan harga palm oil, yaitu Low, High, Vol., dan Open.
Fitur Low, High, dan Vol. memiliki korelasi positif sedangkan Open memiliki korelasi negatif.

Interpretasi koefisien:  
- Jika harga terendah dalam satu hari perdagangan (Low) mengalami kenaikan 1 satuan, maka harga palm oil diperkirangan akan naik sebesar 0,915 satuan, ceteris paribus.  
- Jika harga tertinggi dalam satu hari perdagangan (High) mengalami kenaikan 1 satuan, maka harga palm oil diperkirangan akan naik sebesar 0,753 satuan, ceteris paribus.  
- Jika volume perdagangan palm oil mengalami kenaikan 1 satuan, maka harga palm oil diperkirangan akan naik sebesar 0,000592 satuan, ceteris paribus.  
- Jika harga saat pembukaan pada hari perdagangan (High) mengalami kenaikan 1 satuan, maka harga palm oil diperkirangan akan turun sebesar 0,671 satuan, ceteris paribus.

Catatan:  
Diasumsikan satuan harga adalah dollar AS dan satuan volume adalah metrik ton.  

**---Ini adalah bagian akhir laporan---**
