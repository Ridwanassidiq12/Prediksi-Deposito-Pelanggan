# Penggunaan Machine Learning untuk Memprediksi Deposito Pelanggan pada Bank di Portugal
Dibuat oleh Ridwan Assidiq, Cahyani Nur Patria, M Arief Widagdo
## Project Overview

Projek ini menggunakan data nasabah di Portugal pada periode 2008-2013 untuk memprediksi apakah mereka akan melakukan Deposit. Fitur yang ada di dataset ini mencakup profile pelanggan dan juga keadaan social ekonomi Portugal pada periode tersebut. Analisis yang dilakukan menggunakan berbagai model, mulai dari base model sampai dengan model yang menggunakan bagging dan boosting. Final output adalah prediksi yang dapat ditest di dashboard Streamlit dengan input manual atau dengan input CSV

Prediksi Streamlit dapat diakses di https://final-project-prediksi-deposit.streamlit.app/
Dashboard Tableau dapat diakses di https://public.tableau.com/app/profile/cahyani.nur/viz/FinalProject-BankMarketing/Dashboard1?publish=yes

## Bentuk Akhir

* **Model Prediktif** Model yang siap untuk produksi (`model.joblib`) dilatih untuk prediksi deposito.
* **Analisis Notebook:** Analisis komprehensif menggunakan Jupyter Notebook yang menjelaskan ecara detila proses analisis kami
* **Dashboard** Kami membuat dashboard streamlit dan juga dashboard tableau

## Key Questions Answered

1.  Bagaimana cara memprediksi pelanggan yang akan deposit dengan data yang tersedia
2.  Apa metode yang paling efisien untuk mengevaluasi performa campaign
3.  Model Machine Learning apa yang paling bagus, stabil, dan akurat untuk memprediksi hasil
4.  Berapa kerugian yang mungkin timbul dan cost saving yang bisa didapat jika menggunakan Machine Learning

## Insight

Prediksi menggunakan Machine Learning berhasil mendapatkan nilai recall sebesar 0.73 yang berarti model berhasil memprediksi orang yang akan melakukan deposit sebanyak 73%.


## Technical Details

### Sumber data

Dataset berasal dari Kaggle (https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset?) dan diambil dari dataset bank di Portugal yang telah melakukan kampanye telemarketing selama periode 2008-2013

### Tools & Libraries

* **Bahasa Pemrorgraman:** Python 3
* **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn, statsmodels

### Project Structure

Proyek ini menggunakan struktur proyek data science standard:
1.  **Data Cleaning & EDA:** Mendapatkan assessment untuk qualitas data, mengidentifikasikan outliers an distribusi data, serta mencari hubungan statistik antar fitur numerik dan target dan juga fitur kategorik dan target
2.  **Preprocessing:** Membuat pipeline `ColumnTransformer` pipeline dan mengaplikasikan `RobustScaler` ke fitur numerik dan `OneHotEncoder`/`OrdinalEncoder` ke fitur kategorik untuk mencegah adanya leakage data
3.  **Modeling & Evaluation:** Menggunakan berbagai base model (Regresi Logistic, NN, SVM) dan ensemble model (CatBoosting, Ada Boosting, Gradient Boosting) untuk mendapatkan nilai Recall yang tinggi. Model terbaik akan diuji stabilitasnya dan juga validitasnya untuk menghindari overfitting
4.  **Optimalisasi dan Insight** Melakukan optimalisasi model dan juga melakukan resampling dikarenakan data yang sangat imbalance, model terbaik adalah **Regresi Logistik** karena memiliki nilai recall yang tinggi, tidak mengalami overfitting dan stabil. 

### Cara menjalankan

1.  Download Jupyter Notebook dari Repository ini
2.  Masukan data mentah yang ada di repository ini, pastikan data sudah terupload dengan benar pada session storage.
3.  Jalankan cell secara berurutan untuk mendapatkan hasil yang sama dengan Notebook ini
4.  Model akan meng-output model final dalam bentuk Joblib.
5.  Output model dapat digunakan untuk prediksi data mandiri di Streamlit dan dapat digunakan dalam proyek lain
