# -*- coding: utf-8 -*-
"""Credit Risk Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KDY3tUMVrE_19493axFFoom9wwuch_il

# Prediksi Credit Risk pada Lending Company ABC

# Latar Belakang

Sebuah lending company ingin membangun model yang dapat memprediksi credit risk berdasarkan data pinjaman yang diterima dan yang ditolak menggunakan dataset yang tersedia.

Tujuan:
*   Membuat model yang dapat memprediksi credit risk secara end-to-end berdasarkan data pinjaman yang diterima dan ditolak yang tersedia

# Data Understanding
Pada tahapan ini, akan dijelaskan atribut yang terdapat dalam dataset.

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
"""

# Import library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

# pindah ke direktori kerja
import os

os.chdir('/content/drive/MyDrive/VIX')

"""### Muat Data"""

# Muat data
df = pd.read_csv('loan_data_2007_2014.csv')
df.head()

# Melihat perbandingan jumlah row dengan jumlah user
pd.DataFrame(
    [[len(df.index), df['member_id'].nunique()]]
).rename(
    {0:"Jumlah Baris",1:"Jumlah Id"},
    axis=1
)

"""Berdasarkan informasi di atas, tiap baris merepresentasikan record dari satu individu, sehingga tidak ada data berulang untuk masing-masing orang.

Selain itu, karena tiap baris memiliki ID user yang unik, maak data sudah teragregasi.
"""

# Cek kolom apa saja yang tersedia beserta tipe data dan jumlah data
df.info()

"""Sebelumnya, data pada file csv telah berhasil di-load. Selanjutnya, data perlu disesuaikan yang meliputi:
*   Penghapusan Kolom yang tidak perlu
*   Tipe data atribut

Data dari file tahap 1 berisi 466285 record dengan kondisi yang beragam (beberapa atribut ada yang lengkap, ada missing value, ada juga yang benar-benar kosong). Atribut yang kosong perlu dihapus karena tidak memberikan makna pada data.
"""

# cek nilai unik
for col in sorted(df.columns):
  print(col, df[col].nunique())

# menghapus atribut yang tidak perlu
unimportant_cols = {
    # semua unik/kardinalitas tinggi
    'Unnamed: 0', 'id', 'member_id','emp_title',
    # missing value terlalu banyak/kosong
    'zip_code' , 'annual_inc_joint' , 'dti_joint' , 'verification_status_joint' ,
    'open_acc_6m' , 'open_il_6m' , 'open_il_12m' , 'open_il_24m' , 'mths_since_rcnt_il' ,
    'total_bal_il' , 'il_util' , 'open_rv_12m' , 'open_rv_24m' , 'max_bal_bc' , 'all_util' ,
    'inq_fi' , 'total_cu_tl' , 'inq_last_12m', 'mths_since_last_record','mths_since_last_major_derog',
    'mths_since_last_delinq',
    # free text
    'url', 'desc', 'title',
    # hanya memiliki satu nilai unik
    'policy_code', 'application_type',
    # nilai tidak penting lainnya
    'addr_state', 'sub_grade' #subgrade dapat direpresentasikan oleh grade
}

df_afterdrop = df.drop(unimportant_cols, axis=1)

# cek nilai unik kolom-kolom yang tidak di-drop
for col in sorted(df_afterdrop.columns):
  print(col, df_afterdrop[col].nunique())

print(df_afterdrop.info())

df_afterdrop.sample(5)

"""## Data Cleansing

Data yang digunakan perlu dibersihkan dari:
- missing value
- data duplikat
"""

# cek data duplikat pada tahap 1
print("Data duplikat: ", df_afterdrop.duplicated().sum())

# cek nilai yang tidak akurat
df_afterdrop.describe()

df_afterdrop[df_afterdrop['total_rev_hi_lim']==df_afterdrop['total_rev_hi_lim'].max()]

"""Pada tahap ini terlihat bahwa tidak ada data duplikat. Selain itu, data numerik berada pada rentang yang sesuai.

Adapun beberapa variabel perlu disesuaikan tipe datanya, seperti
* last_credit_pull_d
* last_pymnt_d
* next_pymnt_d
* earliest_cr_line
* issue_d
"""

for col in ['last_credit_pull_d','last_pymnt_d','next_pymnt_d','earliest_cr_line','issue_d']:
  df_afterdrop[col] = pd.to_datetime(df_afterdrop[col], format='%b-%y')

"""# Data Exploration & Data Visualization"""

df_eda = df_afterdrop.copy()
df_eda.info()

"""Pengamatan:
1. Dataset terdiri dari 48 kolom
2. Terdapat kolom bertipe numerik, kategorik, dan datetime
3. Beberapa nilai berisi missing values dan perlu diproses untuk melakukan imputasi ataupun di drop

## Defining Label

Pada kasus penilaian credit risk ini, akan dilakukan pemodelan dengan metode supervised khususnya dengan tugas klasifikasi untuk mengklasifikasikan `bad loan` dan `good loan`. Adapun atribut yang mewakili nilai ini pada tabel adalah `loan_status`
"""

# Melihat sebaran nilai unik loan status
df_eda['loan_status'].value_counts()

sns.countplot(data=df_eda, y='loan_status', order=df_eda['loan_status'].value_counts().index)

"""Berdasarkan hasil pengamatan, terdapat 9 kategori loan status. Dari data ini, akan dilakukan pengelompokan sehingga dihasilkan dua target saja, yaitu "bad loan" dan "good loan".
* good_loan bernilai 1, meliputi status `Current, Fully Paid , and In Grace Period`
* bad_loan bernilai 0, meliputi status lainnya
"""

good_loan = ['Current','Fully Paid','In Grace Period']
df_eda['loan_status'] = np.where(df_eda['loan_status'].isin(good_loan),1,0)
df_eda['loan_status'].value_counts()

"""## Handling Missing Value"""

# Nilai kosong pada fitur `tot_coll_amt`,`tot_cur_bal`,`total_rev_hi_lim` diisi 0 dengan asumsi bahwa nilai 0 menandakan peminjam tidak meminjam lagi
for col in ['tot_coll_amt','tot_cur_bal','total_rev_hi_lim']:
    df_eda[col] = df_eda[col].fillna(0)

# Data numerik diimputasi dengan median
for col in df_eda.select_dtypes(exclude = 'object'):
    df_eda[col] = df_eda[col].fillna(df_eda[col].median())

# Data kategorik diisi dengan nilai yang paling sering muncul
for col in df_eda.select_dtypes(include = 'object'):
    df_eda[col] = df_eda[col].fillna(df_eda[col].mode().iloc[0])

df_eda.isnull().sum()

"""## Feature Engineering

Membuat fitur baru berdasarkan fitur yang sudah ada dengan harapan dapat meningkatkan hasil modeling

#### Ekstraksi fitur dari data datetime
Pada tabel terdapat 5 atribut bertipe datetime,
- 'earliest_cr_line'
- 'last_credit_pull_d'
- 'last_pymnt_d'
- 'issue_d'
- 'next_pymnt_d'

Kita dapat menghitung durasi pembayaran (dalam bulan) dengan menggunakan data `next_pymnt_d` dan `last_pymnt_d`
"""

df_eda['pymnt_time'] = df_eda.apply(lambda x: (x.next_pymnt_d.year - x.last_pymnt_d.year) * 12 + x.next_pymnt_d.month - x.last_pymnt_d.month, axis=1)

df_eda[df_eda['pymnt_time']<0][['next_pymnt_d','last_pymnt_d','pymnt_time']]

"""Namun, ternyata terdapat data yang bernilai negatif, padahal waktu harusnya bernilai positif. Maka dari itu, nilai ini perlu diganti menjadi 0 dengan asumsi bahwa peminjam tidak memiliki pinjaman"""

# ubah nilai pymnt_time menjadi 0 jika nilainya negatif
df_eda.loc[df_eda['pymnt_time'] < 0,'pymnt_time'] = 0

# hapus kolom datetime
df_eda.drop(columns=['issue_d','earliest_cr_line','next_pymnt_d','last_pymnt_d','last_credit_pull_d'], inplace = True)

"""Selain itu, terdapat kolom `term` dengan nilai objek yang dapat direpresentasikan dalam bentuk numerik (bulan)"""

df_eda['term'] = df_eda['term'].apply(lambda term: int(term[:3]))
df_eda['term'].value_counts()

"""## Eksplorasi Parameter Statistik"""

num_col = df_eda.select_dtypes(include='number').columns
cat_col = df_eda.select_dtypes(include='object').columns

"""#### Korelasi Antarvariabel"""

# eksplorasi nilai korelasi
korelasi = df_eda[num_col].corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(korelasi, annot=True, fmt=".2f", ax=ax);

"""Pada gambar terlihat terdapat variabel yang berkorelasi cukup tinggi (nilai korelasi > 0.7) yang dapat mengakibatkan bias, terlebih variabel yang berkolerasi tersebut bukanlah variabel target. Maka dari itu, variabel dengan korelasi tinggi tersebut akan dihapus"""

corr_matrix = df_eda.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_multicol = [column for column in upper.columns if any(upper[column] > 0.8)]
df_eda.drop(high_multicol, axis=1, inplace=True)

df_eda.info()

"""#### Analisis Univariate"""

num_col = [x for x in num_col if x not in high_multicol]

plt.figure(figsize=(24,28))
for i in range(0,len(num_col)):
    plt.subplot(10,4,i+1)
    sns.kdeplot(x=df_eda[num_col[i]], fill=True)
    plt.title(num_col[i], fontsize=16)
    plt.xlabel(' ')
    plt.tight_layout()

"""Dari hasil visualisasi, data yang ada cenderung mengalami kurtosis/skew"""

plt.figure(figsize=(20,20))
for i in range(0,len(cat_col)):
    plt.subplot(5,2,i+1)
    sns.countplot(y=df_eda[cat_col[i]], orient = 'h')
    plt.title(cat_col[i])
    plt.xlabel('')
    plt.tight_layout()

"""Berdasarkan visualisasi di atas, variabel `pymnt_plan` memiliki satu nilai yang sangat dominan, sehingga variabel ini akan dihapus dari dataframe."""

df_eda.drop('pymnt_plan', axis=1, inplace=True)

"""## Feature Selection

-

## Feature Scaling & Encoding
"""

# import library modeling
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

X = df_eda.drop(['loan_status'], axis=1)
y = df_eda['loan_status']

#Split dataset train:test = 80:20
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=41)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

num = list(X_train.select_dtypes(exclude = 'object').columns)
cat = list(X_train.select_dtypes(include = 'object').columns)

for col in cat:
  print(col)
  print(df_eda[col].value_counts())
  print("\n")

# Encode variabel kategorikal
ohe = OneHotEncoder(sparse_output=False).fit(X_train[cat])
X_train_ohe = ohe.transform(X_train[cat])
encoded_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names_out(cat))

encoded_df.sample(5)

# Scaling variabel numerik
ss = StandardScaler().fit(X_train[num])
X_train_ss = ss.transform(X_train[num])
scaled_df = pd.DataFrame(X_train_ss, columns=num)

X_train_res = pd.concat([scaled_df, encoded_df], axis=1)

X_train_res.head(5)

"""# Modeling

## Linear Regression

### Training
"""

# Handle target kelas imbalance dengan SMOTE. SMOTe hanya dilakukan pada data train
sm = SMOTE(random_state=24)
sm.fit(X_train_res, y_train)
X_smote, y_smote = sm.fit_resample(X_train_res, y_train)
X_smote.shape, X_train_res.shape, y_smote.shape, y_train.shape

# train model
logreg = LogisticRegression(random_state = 41)
logreg.fit(X_smote, y_smote)

y_pred_proba_train = logreg.predict_proba(X_train_res)[:][:,1]
print('AUC pada data train :', roc_auc_score(y_train, y_pred_proba_train))

"""### Evaluasi"""

# Transformasi dulu data test dengan encoding OHE dan StandardScaler data train tadi
X_test_ohe = ohe.transform(X_test[cat])
X_test_ss = ss.transform(X_test[num])

X_test_res = pd.concat([pd.DataFrame(X_test_ss, columns=num),
                        pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names_out(cat))], axis=1)

y_pred_proba_test = logreg.predict_proba(X_test_res)[:][:,1]
print('AUC pada data test :', roc_auc_score(y_test, y_pred_proba_test))

y_pred_class = [1 if x > 0.5 else 0 for x in y_pred_proba_test]

print(classification_report(y_test, y_pred_class))

"""Visualisasi Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred_class)
target_names = ['Bad Loan','Good Loan']

df_cm = pd.DataFrame(cm, index = target_names, columns = target_names)
plt.figure(figsize = (8,8))
sns.heatmap(df_cm, annot=True,fmt='.0f')

"""Visualisasi ROC Curve"""

fpr, tpr, tr = roc_curve(y_test, y_pred_proba_test)
auc = roc_auc_score(y_test, y_pred_proba_test)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='AUC = %0.3f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='grey')
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('ROC Curve', fontsize=15)
plt.legend()

"""Visualisasi Kolmogrov-Smirnov (KS)"""

!pip install scikit-plot

import scikitplot as skplt
y_pred_proba = logreg.predict_proba(X_test_res)

skplt.metrics.plot_ks_statistic(y_test, y_pred_proba, figsize=(7,5));

"""## Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

# train model
dtr = DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
dtr.fit(X_smote, y_smote)

y_pred_proba_train = dtr.predict_proba(X_train_res)[:][:,1]
print('AUC pada data train :', roc_auc_score(y_train, y_pred_proba_train))

y_pred_proba_test = dtr.predict_proba(X_test_res)[:][:,1]
print('AUC pada data test :', roc_auc_score(y_test, y_pred_proba_test))

y_pred_class = [1 if x > 0.5 else 0 for x in y_pred_proba_test]

print(classification_report(y_test, y_pred_class))

y_pred_proba = dtr.predict_proba(X_test_res)

skplt.metrics.plot_ks_statistic(y_test, y_pred_proba, figsize=(7,5));

"""## Random Forest"""

from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(n_estimators=100)
rf1.fit(X_smote, y_smote)

y_pred_proba_train = rf1.predict_proba(X_train_res)[:][:,1]
print('AUC pada data train :', roc_auc_score(y_train, y_pred_proba_train))

y_pred_proba_test = rf1.predict_proba(X_test_res)[:][:,1]
print('AUC pada data test :', roc_auc_score(y_test, y_pred_proba_test))

y_pred_class = [1 if x > 0.5 else 0 for x in y_pred_proba_test]

print(classification_report(y_test, y_pred_class))

y_pred_proba = rf1.predict_proba(X_test_res)

skplt.metrics.plot_ks_statistic(y_test, y_pred_proba, figsize=(7,5));

"""# Kesimpulan

|                   | Precision | Recall | F1-score | ROC   | KS    |
|-------------------|-----------|--------|----------|-------|-------|
| Linear Regression | 0.91      | 0.91   | 0.91     | 0.965 | 0.827 |
| Decision Tree     | 0.86      | 0.94   | 0.89     | 0.980 | 0.882 |
| Random Forest     | 0.99          | 0.97       | 0.98         | 0.991      | 0.951      |

Berdasarkan hasil pengujian, model `Random Forest` memberikan nilai ROC dan KS yang lebih besar, yaitu 0.991 dan 0.951 berturut-turut. Nilai ini menggambarkan performa yang baik dari model credit risk yang dihasilkan
"""

# Single inference
X_test_sample = X_test.iloc[0,:]
X_test_sample

X_test_res = pd.concat([pd.DataFrame(ss.transform([X_test_sample[num]])),
                        pd.DataFrame(ohe.transform([X_test_sample[cat]]))], axis=1)

y_pred_proba_test = logreg.predict_proba(X_test_res)[:][:,1]
print("Prediksi: ", 1 if y_pred_proba_test>0.5 else 0,
      "Fakta: ", y_test.iloc[0])