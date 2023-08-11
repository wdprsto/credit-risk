# Laporan Proyek Machine Learning - Wahyu Dwi Prasetio

## Domain Proyek

Pemberian pinjaman oleh lembaga keuangan, seperti bank atau lembaga peminjam lainnya, merupakan proses yang kompleks dan berisiko tinggi. Salah satu tantangan utama yang dihadapi oleh lembaga-lembaga ini adalah bagaimana mereka dapat mengelola risiko kredit dengan efektif. Risiko kredit merujuk pada kemungkinan bahwa peminjam tidak akan mampu atau tidak mau membayar kembali pinjaman yang diberikan, yang dapat berdampak negatif pada keuangan lembaga tersebut.

Oleh karena itu, pembuatan credit risk model berdasarkan data pinjaman yang diterima dan ditolak oleh nasabah menjadi penting untuk membantu lembaga keuangan mengurangi risiko kredit, membuat keputusan kredit yang lebih akurat, dan memastikan kesehatan keuangan jangka panjang. Dengan memanfaatkan metode machine learning untuk mengklasifikasikan nasabah, lembaga dapat meminimalisir kegagalan pinjaman yang dapat terjadi.


## Business Understanding

### Problem Statements
Berdasarkan permasalahan yang ada, lembaga keuangan akan mengembangkan sebuah sistem klasifikasi nasabah sebagai berikut:
- Bagaimana cara menentukan status kelayakan pelanggan untuk menerima pinjaman berdasarkan karakteristik atau fitur tertentu?

### Goals
- Membuat model machine learning yang dapat mengklasifikasikan credit risk nasabah secara end-to-end berdasarkan atribut yang ada.

### Solution statements
- Menggunakan tiga algoritma untuk mencapai solusi yang diinginkan dengan evaluasi menggunakan Precision, Recall, F1-Score, ROC Curve, dan Kolmogrov-Smirnov (KS) Statistic.


## Data Understanding
Data yang digunakan adalah Lending Club Data dari tahun 2007-2014 yang tersedia di [Kaggle](https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014).

### Variabel-variabel pada Loan Data 2007-2014 dataset adalah sebagai berikut:
1.	_rec	:	Jumlah total yang dilakukan oleh investor untuk pinjaman itu pada saat itu.
2.	acc_now_delinq	:	Jumlah akun di mana peminjam sekarang nakal.
3.	addr_state	:	Negara yang disediakan oleh peminjam dalam aplikasi pinjaman
4.	all_util	:	Saldo ke batas kredit untuk semua perdagangan
5.	annual_inc	:	v
6.	annual_inc_joint	:	Penghasilan tahunan yang dilaporkan sendiri gabungan yang disediakan oleh co-peminjam selama pendaftaran
7.	application_type	:
8.	collection_recovery_fee	:	Biaya pengumpulan biaya pengumpulan pos
9.	collections_12_mths_ex_med	:	Jumlah koleksi dalam 12 bulan tidak termasuk koleksi medis
10.	delinq_2yrs	:	Jumlah 30+ hari insiden kenakalan yang lewat di dalam file kredit peminjam selama 2 tahun terakhir
11.	desc	:	Deskripsi pinjaman yang disediakan oleh peminjam
12.	dti_joint	:	Rasio yang dihitung menggunakan total pembayaran bulanan peminjam bersama atas total kewajiban utang, tidak termasuk hipotek dan pinjaman LC yang diminta, dibagi oleh pendapatan bulanan yang dilaporkan sendiri oleh co-peminjam yang dilaporkan sendiri
13.	earliest_cr_line	:	Bulan jalur kredit yang paling awal dilaporkan peminjam dibuka
14.	emp_length	:	Panjang pekerjaan dalam beberapa tahun. Nilai yang mungkin adalah antara 0 dan 10 di mana 0 berarti kurang dari satu tahun dan 10 berarti sepuluh tahun atau lebih.
15.	emp_title	:	Judul pekerjaan yang disediakan oleh peminjam saat mengajukan pinjaman.*
16.	Femp	:	Rasio yang dihitung menggunakan total pembayaran utang bulanan peminjam atas total kewajiban utang, tidak termasuk hipotek dan pinjaman LC yang diminta, dibagi dengan pendapatan bulanan peminjam yang dilaporkan sendiri.
17.	fico_range_high	:	Kisaran batas atas fico peminjam pada awal pinjaman.
18.	fico_range_low	:	Kisaran batas bawah fico peminjam pada awal pinjaman.
19.	funded_amnt	:	Jumlah total yang berkomitmen untuk pinjaman itu pada saat itu.
20.	grade	:	LC menugaskan nilai pinjaman
21.	home_ownership	:	Status kepemilikan rumah yang disediakan oleh peminjam selama pendaftaran. Nilai -nilai kami adalah: sewa, sendiri, hipotek, lainnya.
22.	id	:	ID yang ditugaskan LC yang unik untuk daftar pinjaman.
23.	il_util	:	Rasio total saldo saat ini dengan batas kredit/kredit tinggi pada semua instal acct
24.	initial_list_status	:	Status daftar awal pinjaman. Nilai yang mungkin adalah - utuh, fraksional
25.	inq_fi	:	Jumlah pertanyaan keuangan pribadi
26.	inq_last_12m	:	Jumlah pertanyaan kredit dalam 12 bulan terakhir
27.	inq_last_6mths	:	Jumlah pertanyaan dalam 6 bulan terakhir (tidak termasuk penyelidikan mobil dan hipotek)
28.	installment	:	Pembayaran bulanan yang terhutang oleh peminjam jika pinjaman berasal.
29.	int_rate	:	Menunjukkan jika pendapatan diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi
30.	is_inc_v	:	Menunjukkan jika pendapatan diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi
31.	issue_d	:	Bulan yang didanai pinjaman
32.	id	:	Bulan terbaru LC menarik kredit untuk pinjaman ini
33.	last_fico_range_high	:	Rentang batas atas yang ditarik oleh fico terakhir peminjam.
34.	last_fico_range_low	:	Rentang batas bawah yang dimiliki oleh fico terakhir peminjam.
35.	last_pymnt_amnt	:	Jumlah total pembayaran terakhir yang diterima
36.	last_pymnt_d	:	Bulan lalu pembayaran diterima
37.	loan_amnt	:	Bulan lalu pembayaran diterima
38.	loan_status	:	Status pinjaman saat ini
39.	max_bal_bc	:	Saldo arus maksimum terutang pada semua akun bergulir
40.	member_id	:	ID yang ditugaskan LC yang unik untuk anggota peminjam.
41.	mths_since_last_delinq	:	Jumlah bulan sejak kenakalan terakhir peminjam.
42.	mths_since_last_major_derog	:	Bulan sejak peringkat 90 hari atau lebih buruk terakhir
43.	mths_since_last_record	:	Jumlah bulan sejak catatan publik terakhir.
44.	mths_since_rcnt_il	:	Bulan sejak akun angsuran terbaru dibuka
45.	next_pymnt_d	:	Tanggal Pembayaran Terjadwal Berikutnya
46.	open_acc	:	Jumlah jalur kredit terbuka dalam file kredit peminjam.
47.	open_acc_6m	:	Jumlah perdagangan terbuka dalam 6 bulan terakhir
48.	open_il_12m	:	Jumlah perdagangan terbuka dalam 6 bulan terakhir
49.	open_il_24m	:	Jumlah akun angsuran yang dibuka dalam 24 bulan terakhir
50.	open_il_6m	:	Jumlah akun angsuran yang dibuka dalam 12 bulan terakhir
51.	open_rv_12m	:	Jumlah Perdagangan Revolving Dibuka dalam 12 Bulan Terakhir
52.	open_rv_24m	:	Jumlah perdagangan revolving dibuka dalam 24 bulan terakhir
53.	out_prncp	:	Kepala sekolah yang tersisa untuk jumlah total yang didanai
54.	out_prncp_inv	:	Kepala sekolah yang tersisa untuk sebagian dari jumlah total yang didanai oleh investor
55.	policy_code	:	"Policy_code yang tersedia untuk umum = 1
Produk Baru Tidak Tersedia Umum Kebijakan_Code = 2"
56.	pub_rec	:	Jumlah catatan publik yang menghina
57.	purpose	:	Kategori yang disediakan oleh peminjam untuk permintaan pinjaman.
58.		:	Menunjukkan jika rencana pembayaran telah diberlakukan untuk pinjaman
59.	recoveries	:	Menunjukkan jika rencana pembayaran telah diberlakukan untuk pinjaman
60.	revol_bal	:	Total Saldo Revolving Credit
61.	revol_util	:	Tingkat pemanfaatan jalur bergulir, atau jumlah kredit yang digunakan peminjam relatif terhadap semua kredit revolving yang tersedia.
62.	sub_grade	:	LC Ditugaskan Subgrade Pinjaman
63.	term	:	Jumlah pembayaran atas pinjaman. Nilai dalam beberapa bulan dan dapat berupa 36 atau 60.
64.	title	:	Judul pinjaman yang disediakan oleh peminjam
65.	tot_coll_amt	:	Total jumlah pengumpulan yang pernah ada
66.	tot_cur_bal	:	Total saldo saat ini dari semua akun
67.	total_acc	:	Jumlah total jalur kredit saat ini dalam file kredit peminjam
68.	total_bal_il	:	Total saldo saat ini dari semua akun angsuran
69.	total_cu_tl	:	Jumlah Perdagangan Keuangan
70.	total_pymnt	:	Pembayaran diterima hingga saat ini untuk jumlah total yang didanai
71.	total_pymnt_inv	:	Pembayaran diterima hingga saat ini untuk sebagian dari jumlah total yang didanai oleh investor
72.	total_rec_int	:	Bunga diterima hingga saat ini
73.	total_rec_late_fee	:	Biaya keterlambatan yang diterima hingga saat ini
74.	total_rec_prncp	:	Kepala sekolah diterima hingga saat ini
75.	total_rev_hi_lim  	:	Total Batas Kredit/Kredit Tinggi Revolving
76.	url	:	URL untuk halaman LC dengan data daftar.
77.	verified_status_joint	:	Menunjukkan jika pendapatan bersama co-peminjam diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi
78.	zip_code	:	3 nomor pertama dari kode pos yang disediakan oleh peminjam dalam aplikasi pinjaman.


## Data Preparation

### Attribute Removal
Pada proses ini, dilakukan penghapusan atribut yang tidak perlu. Adapun atribut-atribut yang dihapus diantaranya memiliki kriteria sebagai berikut:
- Semua nilainya unik/kardinalitas tinggi
- Missing valuenya terlalu banyak
- Free text
- Hanya memiliki satu nilai unik

Adapun atribut yang dihapus pada tahapan ini adalah `id`, `member_id`,`emp_title`, `zip_code` , `annual_inc_joint` , `dti_joint` , `verification_status_joint` , `open_acc_6m` , `open_il_6m` , `open_il_12m` , `open_il_24m` , `mths_since_rcnt_il` , `total_bal_il` , `il_util` , `open_rv_12m` , `open_rv_24m` , `max_bal_bc` , `all_util` , `inq_fi` , `total_cu_tl` , `inq_last_12m`, `mths_since_last_record`,`mths_since_last_major_derog`, `mths_since_last_delinq`, `url`, `desc`, `title`, `policy_code`, `application_type`, `addr_state`, `sub_grade`.

### Data Cleansing
Pada tahapan ini, dilakukan pembersihan terhadap data duplikat. Proses ini dilakukan agar model tidak mempelajari data yang sama yang mengakibatkan overfit. 
- Pada data ini, tidak ada data duplikat.

Selain itu, dilakukan penyesuaian tipe data terhadap atribut yang tipe nilainya tidak sesuai. Adapun atribut yang tipe datanya disesuaikan yaitu `last_credit_pull_d`,`last_pymnt_d`,`next_pymnt_d`,`earliest_cr_line`,`issue_d`, di mana atribut-atribut ini seharusnya bertipe `datetime`

### Labeling
Pada tahapan ini, label klasifikasi didefinisikan berdasarkan atribut `loan_status` dengan target 2 kelas, dengan rincian:
- good_loan bernilai 1, meliputi status `Current, Fully Paid , and In Grace Period`
- bad_loan bernilai 0, meliputi status lainnya

### Handling Missing Value
Untuk menangani missing value, proses pendekatan dibagi berdasarkan tipe data dari atribut tersebut. Nilai kosong pada fitur `tot_coll_amt`,`tot_cur_bal`,`total_rev_hi_lim` diisi 0 dengan asumsi bahwa nilai 0 menandakan peminjam tidak meminjam lagi
```
  for col in ['tot_coll_amt','tot_cur_bal','total_rev_hi_lim']:
      df_eda[col] = df_eda[col].fillna(0)
```

Data numerik diimputasi dengan median
```
  for col in df_eda.select_dtypes(exclude = 'object'):
      df_eda[col] = df_eda[col].fillna(df_eda[col].median())
```

Data kategorik diisi dengan nilai yang paling sering muncul
```
  for col in df_eda.select_dtypes(include = 'object'):
      df_eda[col] = df_eda[col].fillna(df_eda[col].mode().iloc[0])
```

### Feature Extraction
Pada bagian ini, dilakukan pembuatan fitur baru berdasarkan fitur yang sudah ada dengan harapan dapat meningkatkan hasil modeling. Kita dapat menghitung durasi pembayaran (dalam bulan) dengan menggunakan data `next_pymnt_d` dan `last_pymnt_d`
```
  df_eda['pymnt_time'] = df_eda.apply(lambda x: (x.next_pymnt_d.year - x.last_pymnt_d.year) * 12 + x.next_pymnt_d.month - x.last_pymnt_d.month, axis=1)
```
Namun, ternyata terdapat data yang bernilai negatif, padahal waktu harusnya bernilai positif. Maka dari itu, nilai ini perlu diganti menjadi 0 dengan asumsi bahwa peminjam tidak memiliki pinjaman

Selain itu, terdapat kolom term dengan nilai objek yang dapat direpresentasikan dalam bentuk numerik (dalam waktu bulan)
```
  df_eda['term'] = df_eda['term'].apply(lambda term: int(term[:3]))
```

### Feature Scaling & Encoding
Sebelum dilakukan scaling dan encode pada data, data dibagi ke dalam set train dan test dengan proporsi 80:20. `OneHotEncoding` dilakukan terhadap atribut kategorikal, sedangkan `Standardization` dilakukan terhadap data numerik. Tujuan proses encoding ialah agar data objek dapat dipelajari oleh model yang hanya mampu menerima input numerik, sedangkan proses standardisasi bertujuan untuk memetakan nilai sehingga memiliki rata-rata 0 dan standar deviasi 1, dengan harapan dapat meningkatkan performa model. 

Di sisi lain, data test tidak menerima perlakukan scaling dan encode sebab ia adalah data uji yang akan menggambarkan performa model secara general.


## Modeling
Pada proses modeling dilakukan percobaan terhadap 3 jenis model, yaitu Linear Regression, Decision Tree, dan Random Forest.
### Linear Regression
Untuk menangani imbalance dataset pada set train, dilakukan proses oversampling menggunakan `SMOTE`. Model Linear Regression yang digunakan memanfaatkan parameter default serta nilai `random_state` tertentu agar percobaan nantinya dapat diduplikasi kembali dan diukur perbandingan hasil performanya. 

Untuk mengimplementasikan linear regression menggunakan library sklearn, kita dapat menginstansiasi objek `LogisticRegression`, kemudian melatih data train pada model tersebut.
```
  logreg = LogisticRegression(random_state = 41)
  logreg.fit(X_smote, y_smote)
```

### Decision Tree
Sama seperti sebelumnya, data train di upsampling menggunakan `SMOTE`. Model Decision Tree yang digunakan memanfaatkan parameter default dengan `criterion` entropy dan `max_depth` 5. Parameter criterion merepresentasikan fungsi untuk mengukur kualitas split pada tree, sedangkan max_depth akan membatasi tinggi dari tree yang dihasilkan.

Untuk mengimplementasikan decision tree menggunakan library sklearn, kita dapat menginstansiasi objek `DecisionTreeClassifier`, kemudian melatih data train pada model tersebut.
```
  dtr = DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
  dtr.fit(X_smote, y_smote)
```

### Random Forest
Sama seperti sebelumnya, data train di upsampling menggunakan `SMOTE`. Model ensemble Random Forest yang digunakan memanfaatkan parameter default serta nilai `n_estimators` 100 yang mewakili jumlah tree pada forest. Random forest sendiri merupakan pengoptimalan dari Decision tree sebelumnya, yang mana model ini menggabungkan beberapa Decision Tree untuk meningkatkan kualitas prediksi dan mengatasi beberapa kelemahan Decision Tree.

Untuk mengimplementasikan Random Forest menggunakan library sklearn, kita dapat menginstansiasi objek `RandomForestClassifier`, kemudian melatih data train pada model tersebut.
```
  rf1 = RandomForestClassifier(n_estimators=100)
  rf1.fit(X_smote, y_smote)
```


## Evaluation
Karena model yang dibuat merupakan model untuk kasus klasifikasi, metriks evaluasi yang digunakan adalah Precision, Recall, F-1 Score, AUC pada ROC Curve, dan KS Statistic.
- **Precision** mengukur sejauh mana dari semua prediksi positif yang dilakukan oleh model, berapa di antaranya yang benar-benar positif. Precision dihitung dengan mengukur rasio antara True Positive (TP) dan jumlah prediksi positif yang dilakukan oleh model (TP + False Positive, atau FP). Precision penting ketika fokus pada mengurangi false positive (kasus di mana model salah mengidentifikasi sesuatu sebagai positif).
- **Recall** mengukur sejauh mana dari semua instance positif dalam data, berapa di antaranya yang berhasil diidentifikasi oleh model. Recall dihitung dengan mengukur rasio antara True Positive (TP) dan total jumlah instance positif dalam data (TP + False Negative, atau FN). Recall penting ketika fokus pada mengurangi false negative (kasus di mana model gagal mengidentifikasi sesuatu yang seharusnya positif).
- **F-1 Score**: F-1 Score adalah harmonic mean dari Precision dan Recall.
- **AUC pada ROC Curve**: AUC (Area Under the Curve) mengukur kemampuan model untuk membedakan antara kelas positif dan negatif. ROC Curve adalah grafik yang mengilustrasikan hubungan antara True Positive Rate (Recall) dan False Positive Rate (1 - Specificity) pada berbagai ambang batas. Nilai AUC yang lebih tinggi menunjukkan kemampuan model yang lebih baik dalam mengklasifikasikan secara benar.
- **KS Statistic**: KS (Kolmogorov-Smirnov) Statistic digunakan untuk mengukur sejauh mana distribusi probabilitas prediksi positif dan negatif dari model berbeda. Nilai KS yang lebih tinggi menunjukkan bahwa model mampu membedakan kelas dengan lebih baik.


|                   | Precision | Recall | F1-score | ROC   | KS    |
|-------------------|-----------|--------|----------|-------|-------|
| Linear Regression | 0.91      | 0.91   | 0.91     | 0.965 | 0.827 |
| Decision Tree     | 0.86      | 0.94   | 0.89     | 0.980 | 0.882 |
| Random Forest     | 0.99          | 0.97       | 0.98         | 0.991      | 0.951      |

Berdasarkan hasil pengujian, model `Random Forest` memberikan nilai ROC dan KS yang lebih besar, yaitu 0.991 dan 0.951 berturut-turut. Nilai ini menggambarkan performa yang baik dari model credit risk yang dihasilkan

**---Ini adalah bagian akhir laporan---**

