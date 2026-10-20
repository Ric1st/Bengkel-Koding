import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Bengkod Fast Track',
                   page_icon='🏘️', layout='wide')


@st.cache_resource
def muat_model():
    return joblib.load('model_harga_rumah.joblib')


@st.cache_data
def muat_data():
    return pd.read_csv('data_bersih.csv')


def rupiah(x):
    return f'Rp {x / 1e9:,.2f} miliar' if x >= 1e9 else f'Rp {x / 1e6:,.0f} juta'


model, data = muat_model(), muat_data()
daftar_kecamatan = sorted(data['Kecamatan'].unique())

st.title('🏘️ Prediki Harga Rumah di Jakarta dan Sekitarnya')
st.caption(
    'Model Random Forest 🌳 yang dilatih dari ±27 ribu listing properti hasil scraping.')

tab_prediksi, tab_fitur, tab_visual = st.tabs(
    ['Prediksi', 'Penjelasan Fitur', 'Visualisasi'])

with tab_prediksi:
    with st.form('form_prediksi'):
        k1, k2, k3 = st.columns(3)
        lt = k1.number_input('Luas Tanah (m²)', 10, 100000, 120, step=5)
        lb = k2.number_input('Luas Bangunan (m²)', 0, 10000, 100, step=5,
                             help='Isi 0 jika berupa kavling (tanah tanpa bangunan).')
        kec = k3.selectbox('Kecamatan', daftar_kecamatan,
                           index=daftar_kecamatan.index('Cinere'))
        kirim = st.form_submit_button('Prediksi Harga', type='primary')

    if kirim:
        masukan = pd.DataFrame(
            [{'Luas Tanah': lt, 'Luas Bangunan': lb, 'Kecamatan': kec}])
        prediksi = model.predict(masukan)[0]
        st.session_state['hasil'] = {'lt': lt, 'harga': prediksi}
        median_kec = data.loc[data['Kecamatan'] == kec, 'Harga'].median()
        h1, h2 = st.columns(2)
        h1.metric('Perkiraan harga', rupiah(prediksi))
        h2.metric(f'Median harga di {kec}', rupiah(median_kec))
        st.caption('Pada data uji, prediksi model rata-rata meleset sekitar 33% (MAPE). '
                   'Gunakan sebagai patokan awal, bukan harga pasti.')

with tab_fitur:
    st.markdown("""
**Fitur yang dipakai model** dipilih lewat korelasi dan *permutation importance*.

| Fitur | Arti | Seberapa penting |
|---|---|---|
| **Luas Tanah** | Luas tanah dalam m² | Paling dominan (importance 0,88) |
| **Kecamatan** | Lokasi listing (29 kategori) | Penting (0,07). Median harga antar lokasi bisa berbeda sekitar 14 kali |
| **Luas Bangunan** | Luas bangunan dalam m². Isi 0 untuk kavling | Pelengkap (0,02) |

Jumlah kamar tidur, kamar mandi, dan garasi tidak dipakai. Kamar sangat berkorelasi dengan luas bangunan
(0,81 sampai 0,85), sehingga informasinya sudah terwakili dan tidak menambah akurasi.

**Performa pada data uji:** MAE sekitar Rp 1,19 miliar, MAPE 32,6%, dan R² 0,741.
""")

with tab_visual:
    median_kec = data.groupby('Kecamatan')['Harga'].median().div(
        1e9).sort_values().reset_index()
    fig1 = px.bar(median_kec, x='Harga', y='Kecamatan', orientation='h',
                  labels={'Harga': 'Median harga (miliar Rp)'}, title='Median harga per Kecamatan')
    fig1.update_layout(height=700)
    st.plotly_chart(fig1, width='stretch')

    contoh = data.sample(min(3000, len(data)), random_state=42)
    fig2 = px.scatter(contoh, x='Luas Tanah', y='Harga', log_x=True, log_y=True, opacity=0.4,
                      title='Harga vs Luas Tanah (sampel 3.000 listing, skala log)')
    if 'hasil' in st.session_state:
        h = st.session_state['hasil']
        fig2.add_scatter(x=[h['lt']], y=[h['harga']], mode='markers', name='Prediksi Anda',
                         marker=dict(color='red', size=14, symbol='star'))
    st.plotly_chart(fig2, width='stretch')
