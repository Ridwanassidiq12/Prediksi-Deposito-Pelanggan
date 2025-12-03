import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from utils import feature_engineering
import plotly.express as px

# --- LOAD MODEL ---
model = joblib.load('best_model_pediksi_deposit.pkl')

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Nasabah Melakukan Deposit Bank Marketing", layout="wide")
st.title('🏦 Prediksi Nasabah Melakukan Deposit Nasabah Bank Marketing')
st.markdown("""
Aplikasi ini memprediksi apakah nasabah akan **berlangganan produk (y=1)** berdasarkan data input kampanye pemasaran.  
**Kolom `duration` tidak digunakan** karena dianggap menyebabkan data leakage.
""")

# --- SIDEBAR METODE INPUT ---
st.sidebar.header("Pilih Metode Input Data")
input_method = st.sidebar.radio("Metode Input:", ['Manual', 'Upload File (CSV/XLSX)'])

# Placeholder untuk hasil
hasil_prediksi_section = st.empty()

# -----------------------------------------
#               INPUT MANUAL
# -----------------------------------------
if input_method == 'Manual':
    st.sidebar.markdown("### Masukkan Data Nasabah")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Usia', min_value=18, max_value=100, value=35)
        job = st.selectbox('Pekerjaan', ['admin.', 'technician', 'services', 'management', 'retired',
                                         'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
                                         'student', 'self-employed'])
        marital = st.selectbox('Status Pernikahan', ['married', 'single', 'divorced'])
        education = st.selectbox('Pendidikan', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                               'professional.course', 'university.degree', 'illiterate'])
        default = st.selectbox('Memiliki Kredit Macet?', ['yes', 'no'])
        housing = st.selectbox('Memiliki Pinjaman Rumah?', ['yes', 'no'])
        loan = st.selectbox('Memiliki Pinjaman Pribadi?', ['yes', 'no'])

    with col2:
        contact = st.selectbox('Jenis Kontak', ['cellular', 'telephone'])
        month = st.selectbox('Bulan Kontak Terakhir', ['mar', 'apr', 'may', 'jun',
                                                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day_of_week = st.selectbox('Hari Kontak', ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input('Jumlah Kontak Kampanye', min_value=1, value=1)
        pdays = st.number_input('Hari Sejak Kontak Terakhir', min_value=0, value=999)
        previous = st.number_input('Jumlah Kontak Sebelumnya', min_value=0, value=0)
        poutcome = st.selectbox('Hasil Kampanye Sebelumnya', ['nonexistent', 'failure', 'success'])

    with st.expander("📉 Variabel Ekonomi"):
        emp_var_rate = st.number_input('Tingkat Pengangguran', value=1.1)
        cons_price_idx = st.number_input('Indeks Harga Konsumen', value=93.994)
        cons_conf_idx = st.number_input('Indeks Keyakinan Konsumen', value=-36.4)
        euribor3m = st.number_input('Suku Bunga Euribor 3 Bulan', value=4.857)
        nr_employed = st.number_input('Jumlah Pegawai', value=5191.0)

    if st.sidebar.button('Prediksi Manual'):
        input_data = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'nr.employed': nr_employed
        }])

        # Feature engineering
        input_data = feature_engineering(input_data)

        hasil = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if hasil == 1:
            hasil_prediksi_section.success(f"✅ Nasabah diprediksi **BERLANGGANAN** (Probabilitas: {prob:.2f})")
        else:
            hasil_prediksi_section.warning(f"⚠️ Nasabah diprediksi **TIDAK berlangganan** (Probabilitas: {prob:.2f})")


# -----------------------------------------
#             UPLOAD FILE
# -----------------------------------------
else:
    file = st.sidebar.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

    if file is not None:

        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        if 'duration' in df.columns:
            df.drop(columns=['duration'], inplace=True)

        st.subheader("📄 Data yang diupload")
        st.dataframe(df, use_container_width=True)

        if st.sidebar.button("Prediksi dari File"):

            # ---- FEATURE ENGINEERING PENTING !!! ----
            df_fe = feature_engineering(df)

            # Prediksi
            predictions = model.predict(df_fe)
            probabilities = model.predict_proba(df_fe)[:, 1]

            df['Prediksi'] = predictions
            df['Probabilitas'] = probabilities.round(2)

            st.success("✅ Prediksi berhasil dilakukan!")
            st.dataframe(df, use_container_width=True)

            # --- GRAFIK PIE ---
            fig_pie = px.pie(df, names='Prediksi', title='Distribusi Prediksi Nasabah')
            st.plotly_chart(fig_pie, use_container_width=True)

            # --- GRAFIK DISTRIBUSI PROBABILITAS ---
            fig_prob = px.histogram(df, x='Probabilitas',
                                    nbins=20,
                                    title='Distribusi Probabilitas Prediksi')
            st.plotly_chart(fig_prob, use_container_width=True)

            # --- DOWNLOAD FILE ---
            output = BytesIO()
            with pd.ExcelWriter(output) as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')

            st.download_button(
                label="⬇️ Unduh Hasil Prediksi (.xlsx)",
                data=output.getvalue(),
                file_name='hasil_prediksi_bank_marketing.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

