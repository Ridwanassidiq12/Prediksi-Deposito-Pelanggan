# Implementasi Machine Learning untuk prediksi nasabah potensial Pada PortBank
## Project Overview
#Author

Ridwan As-Sidiq
M Arief Widagdo

🌐 Overview

Portbank


Portbank merupakan institusi perbankan yang melakukan kampanye pemasaran langsung (telemarketing) untuk menawarkan produk deposito berjangka kepada nasabah. Efektivitas kampanye sebelumnya belum optimal, di mana proses telemarketing memakan waktu dan sumber daya manusia yang signifikan, sementara targeting yang tidak tepat berpotensi menyebabkan hilangnya peluang pendapatan (*revenue lost*) dan biaya operasional yang sia-sia.


Dengan tingkat konversi (deposit) yang hanya sekitar 11.3% dari total data, terjadi ketidakseimbangan kelas yang signifikan. Tantangan utamanya adalah bagaimana meningkatkan efisiensi kampanye dengan menargetkan nasabah yang memiliki probabilitas tinggi untuk berlangganan deposito, sehingga Portbank dapat menghemat biaya pemasaran dan memaksimalkan profit.
Proyek ini bertujuan untuk menganalisis data historis pelanggan, mengidentifikasi faktor-faktor yang mempengaruhi keputusan nasabah, dan membangun model klasifikasi Machine Learning yang dapat memprediksi calon nasabah potensial secara akurat.


🎯 Objective


Menganalisis pola perilaku nasabah dan karakteristik demografis yang membedakan antara nasabah yang berpotensi membuka deposito dan yang tidak.
Membangun model machine learning klasifikasi untuk memprediksi probabilitas nasabah berlangganan deposito berjangka.
Mendukung pengambilan keputusan berbasis data (Decision Support System) untuk mengoptimalkan strategi pemasaran, mengurangi biaya telemarketing yang tidak perlu (False Positive), dan meminimalkan hilangnya nasabah potensial (False Negative).

💡 Key Insights


Ketidakseimbangan Data (Imbalanced Data): Dataset memiliki proporsi target yang sangat timpang (88.7% 'No' vs 11.3% 'Yes'), sehingga memerlukan teknik resampling (Random OverSampling terbukti paling efektif).
Indikator Ekonomi Makro: Fitur `economic_stability` (gabungan dari `emp.var.rate` dan `cons.conf.idx`) menjadi indikator penting, menunjukkan bahwa kondisi ekonomi nasional mempengaruhi keputusan investasi nasabah.
Frekuensi Kontak: Fitur `contact_count` (gabungan `campaign` dan `previous`) menunjukkan bahwa frekuensi interaksi mempengaruhi engagement; terlalu sedikit kontak mungkin kurang efektif, namun terlalu banyak bisa mengganggu.
Beban Finansial: Nasabah dengan beban pinjaman ganda (`loan_burden`: housing + personal loan) cenderung memiliki perilaku berbeda dalam mengambil deposito baru.
Durasi Panggilan: Fitur `duration` memiliki korelasi sangat tinggi dengan target, namun dihapus dari pemodelan untuk mencegah *data leakage* karena durasi tidak diketahui sebelum panggilan dilakukan.

### Dataset

```text
01] age
02] job
03] marital
04] education
05] default
06] housing
07] loan
08] contact
09] month
10] day_of_week
11] duration (dropped for modeling)
12] campaign
13] pdays
14] previous
15] poutcome
16] emp.var.rate
17] cons.price.idx
18] cons.conf.idx
19] euribor3m
20] nr.employed
21] y (Target: Deposit Subscription)
```


🧠 Model Candidate
Beberapa algoritma klasifikasi telah diuji dengan berbagai teknik resampling (SMOTE, Random UnderSampling, Random OverSampling) untuk menangani imbalanced data:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
XGBoost Classifier
Evaluasi model difokuskan pada metrik F2-Score untuk memberikan bobot lebih pada Recall (meminimalkan False Negative/kehilangan nasabah potensial) sambil tetap menjaga Precision.

✅ Model Selected and Evaluation
Model terbaik yang dipilih adalah **Logistic Regression** dengan teknik **Random OverSampling**. Model ini dipilih setelah melalui Hyperparameter Tuning (GridSearch) karena memberikan keseimbangan terbaik antara bias dan variance serta performa F2-Score yang stabil.

**Best Parameters:** `C=10`, `penalty='l1'`, `solver='liblinear'`

**Performance Metrics**
| Metric | Score (Test) |
|--------|--------------|
| **F2-Score** | **0.5358** |
| Recall | 0.6422 |
| Precision | 0.3222 |
| ROC-AUC | 0.79 |

**Evaluasi Overfitting:**
Perbedaan antara F2-Score Train (0.5407) dan Test (0.5358) sangat kecil (< 1%), menandakan model **Robust** dan tidak mengalami overfitting (Good Fit).

⭐️ Kesimpulan
Berdasarkan hasil analisis dan pemodelan pada data Bank Marketing Portugal:

1.  **Faktor Penentu:** Kondisi ekonomi makro dan riwayat interaksi sebelumnya merupakan faktor krusial dalamn prediksi. Stabilitas ekonomi dan pendekatan kontak yang tepat sasaran meningkatkan peluang keberhasilan.
2.  **Performa Model:** Model Logistic Regression mendapatkan F2 score sebesar  **0.53** dari total nasabah potensial, dengan kemampuan membedakan kelas positif dan negatif yang baik (AUC 0.79).
3.  **Strategi Bisnis:** Penggunaan model memungkinkan bank untuk beralih dari strategi "menelepon semua orang" menjadi pendekatan yang terarah. Nasabah yang diprediksi memiliki probabilitas tinggi akan diprioritaskan, menghemat sumber daya manusia dan biaya operasional.
4.  **Rekomendasi:** Gunakan *threshold adjustment* (seperti Youden’s J Statistic: 0.52) untuk menyeimbangkan antara menangkap peluang deposito dan biaya operasional, sesuai dengan selera risiko bisnis saat ini.


- [Link Presentasi (Canva)](https://www.canva.com/design/DAG6QoMcmy0/5GVfHejCRV-PT5aORdauQw/edit)
- [Link Tableau](https://public.tableau.com/app/profile/sidiq.qq/viz/shared/6723P9P53)
- [Link Streamlit](https://final-project-prediksi-deposit.streamlit.app/)

