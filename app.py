import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Prediksi Harga Rumah Depok', page_icon='🏘️', layout='wide')

st.markdown("""
<style>
.hero {padding: 1.5rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem; color: white;
       background: linear-gradient(120deg, #1f6feb 0%, #8957e5 100%);}
.hero h1 {margin: 0; font-size: 2.1rem; color: white !important;}
.hero p {margin: .4rem 0 0 0; opacity: .92; color: white !important;}
.harga {font-size: 2.8rem; font-weight: 700; line-height: 1.15;}
.label {opacity: .7; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def muat_model():
    return joblib.load('model_harga_rumah.joblib')


@st.cache_data
def muat_data():
    d = pd.read_csv('data_bersih.csv')
    d['Jenis'] = np.where(d['Luas Bangunan'] == 0, 'Kavling', 'Rumah')
    d['Harga per m² tanah'] = d['Harga'] / d['Luas Tanah']
    return d


def angka(x, desimal=0):
    return f'{x:,.{desimal}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def rupiah(x):
    if x >= 1e9:
        return f'Rp {angka(x / 1e9, 2)} miliar'
    if x >= 1e6:
        return f'Rp {angka(x / 1e6)} juta'
    return f'Rp {angka(x)}'


def pakai_contoh(lt, lb, kec):
    st.session_state.update(lt=lt, lb=lb, kec=kec)


model, data = muat_model(), muat_data()
daftar_kecamatan = sorted(data['Kecamatan'].unique())
stat_kec = data.groupby('Kecamatan').agg(
    **{'Median harga': ('Harga', 'median'), 'Median harga per m² tanah': ('Harga per m² tanah', 'median'),
       'Jumlah listing': ('Harga', 'size')}).reset_index()
batas_lt = data['Luas Tanah'].quantile(.99)
batas_lb = data.loc[data['Jenis'] == 'Rumah', 'Luas Bangunan'].quantile(.99)

for kunci, nilai in {'lt': 120, 'lb': 100, 'kec': 'Cinere'}.items():
    st.session_state.setdefault(kunci, nilai)

with st.sidebar:
    st.header('Tentang aplikasi')
    st.write('Memperkirakan harga rumah atau kavling di Depok dan sekitarnya dari **luas tanah, luas bangunan, '
             'dan lokasi**.')
    st.divider()
    st.subheader('Kinerja model')
    st.caption('Random Forest hasil tuning, diuji pada 5.445 listing yang tidak dipakai saat pelatihan.')
    st.metric('Median selisih prediksi', 'Rp 284 juta')
    st.metric('Rata-rata meleset (MAPE)', '32,6%')
    st.metric('R²', '0,741')
    st.divider()
    st.caption('Data: ±27 ribu listing properti hasil scraping. Hasil adalah perkiraan, bukan penilaian resmi.')

st.markdown("""
<div class="hero">
  <h1>🏠 Prediksi Harga Rumah Depok dan Sekitarnya</h1>
  <p>Masukkan ukuran dan lokasi properti, lalu lihat perkiraan harga beserta pembandingnya dari data pasar.</p>
</div>
""", unsafe_allow_html=True)

tab_prediksi, tab_fitur, tab_visual, tab_proses = st.tabs(
    ['🔮 Prediksi', '🧩 Penjelasan Fitur', '📊 Visualisasi', '🛠️ Proses dan Hasil'])

with tab_prediksi:
    st.caption('Coba contoh cepat:')
    c1, c2, c3, c4 = st.columns(4)
    c1.button('Rumah kecil (Cipayung)', on_click=pakai_contoh, args=(50, 60, 'Cipayung'), width='stretch')
    c2.button('Rumah menengah (Cinere)', on_click=pakai_contoh, args=(120, 100, 'Cinere'), width='stretch')
    c3.button('Rumah besar (Beji)', on_click=pakai_contoh, args=(300, 250, 'Beji'), width='stretch')
    c4.button('Kavling (Sawangan)', on_click=pakai_contoh, args=(500, 0, 'Sawangan'), width='stretch')

    with st.form('form_prediksi'):
        k1, k2, k3 = st.columns(3)
        k1.number_input('Luas Tanah (m²)', 10, 100000, step=5, key='lt')
        k2.number_input('Luas Bangunan (m²)', 0, 10000, step=5, key='lb',
                        help='Isi 0 jika berupa kavling (tanah tanpa bangunan).')
        k3.selectbox('Kecamatan', daftar_kecamatan, key='kec')
        kirim = st.form_submit_button('Prediksi Harga', type='primary')

    if kirim:
        lt, lb, kec = st.session_state['lt'], st.session_state['lb'], st.session_state['kec']
        masukan = pd.DataFrame([{'Luas Tanah': lt, 'Luas Bangunan': lb, 'Kecamatan': kec}])
        st.session_state['hasil'] = {'lt': lt, 'lb': lb, 'kec': kec, 'harga': float(model.predict(masukan)[0])}

    h = st.session_state.get('hasil')
    if h:
        info = stat_kec.set_index('Kecamatan').loc[h['kec']]
        with st.container(border=True):
            kiri, kanan = st.columns([3, 2])
            with kiri:
                st.markdown(f"<div class='label'>Perkiraan harga untuk {angka(h['lt'])} m² tanah, "
                            f"{angka(h['lb'])} m² bangunan di {h['kec']}</div>"
                            f"<div class='harga'>{rupiah(h['harga'])}</div>", unsafe_allow_html=True)
                st.write(f"**Rentang wajar (±20%):** {rupiah(h['harga'] / 1.2)} sampai {rupiah(h['harga'] / 0.8)}")
                st.write(f"**Rentang lebar (±30%):** {rupiah(h['harga'] / 1.3)} sampai {rupiah(h['harga'] / 0.7)}")
                st.caption('Pada data uji, sekitar 54% prediksi masuk rentang ±20% dan 70% masuk rentang ±30% '
                           'dari harga sebenarnya.')
            with kanan:
                st.metric(f"Median harga di {h['kec']}", rupiah(info['Median harga']),
                          f"prediksi {h['harga'] / info['Median harga'] - 1:+.0%} dari median", delta_color='off')
                st.metric('Harga per m² tanah (prediksi)', rupiah(h['harga'] / h['lt']),
                          f"median {h['kec']}: {rupiah(info['Median harga per m² tanah'])}", delta_color='off')

        if h['lb'] == 0:
            st.warning('Anda memasukkan luas bangunan 0 (kavling). Estimasi kavling kurang akurat: pada data uji '
                       'rata-rata meleset sekitar 55%, dibanding sekitar 30% untuk rumah.')
        if h['lt'] > batas_lt or h['lb'] > batas_lb:
            st.info('Ukuran yang dimasukkan lebih besar dari 99% listing di data, jadi hasilnya kurang andal.')

        g1, g2 = st.columns(2)
        dk = data[data['Kecamatan'] == h['kec']]
        dk = dk[dk['Harga'] <= dk['Harga'].quantile(.98)]
        fig = px.histogram(dk.assign(**{'Harga (miliar Rp)': dk['Harga'] / 1e9}), x='Harga (miliar Rp)', nbins=40,
                           title=f"Sebaran harga listing di {h['kec']}")
        fig.add_vline(x=h['harga'] / 1e9, line_color='red', line_width=3, annotation_text='Prediksi')
        g1.plotly_chart(fig, width='stretch')

        s = data[(data['Kecamatan'] == h['kec'])].dropna(subset=['Luas Tanah']).copy()
        s['jarak'] = np.log(s['Luas Tanah'] / h['lt']).abs() + np.log((s['Luas Bangunan'] + 1) / (h['lb'] + 1)).abs()
        mirip = s.nsmallest(5, 'jarak')
        g2.markdown('**Listing paling mirip di data**')
        g2.dataframe(pd.DataFrame({'Luas Tanah (m²)': mirip['Luas Tanah'].astype(int),
                                   'Luas Bangunan (m²)': mirip['Luas Bangunan'].astype(int),
                                   'Harga': mirip['Harga'].map(rupiah)}), hide_index=True, width='stretch')
        g2.caption('Pembanding nyata dari data, untuk memeriksa apakah prediksi masuk akal.')

with tab_fitur:
    st.subheader('Tiga fitur yang dipakai model')
    f1, f2, f3 = st.columns(3)
    with f1.container(border=True):
        st.markdown('### 📐 Luas Tanah')
        st.write('Penentu harga paling kuat. Importance **0,88** dari total pengaruh.')
    with f2.container(border=True):
        st.markdown('### 📍 Kecamatan')
        st.write('Lokasi sangat memengaruhi harga. Median antar lokasi bisa berbeda sekitar **14 kali**. '
                 'Ada 29 kategori (importance **0,07**).')
    with f3.container(border=True):
        st.markdown('### 🏗️ Luas Bangunan')
        st.write('Pelengkap (importance **0,02**). Isi 0 untuk kavling.')

    penting = pd.DataFrame({
        'Fitur': ['Luas Tanah', 'Kecamatan', 'Luas Bangunan', 'Garasi', 'Kavling', 'Kamar Mandi', 'Kamar Tidur'],
        'Importance': [0.8815, 0.0713, 0.0190, 0.0001, 0.0, -0.0001, -0.0060]})
    penting['Status'] = np.where(penting['Importance'] >= 0.005, 'Dipakai', 'Tidak dipakai')
    fig = px.bar(penting.iloc[::-1], x='Importance', y='Fitur', color='Status', orientation='h',
                 title='Permutation importance: seberapa turun skor model jika fitur diacak')
    st.plotly_chart(fig, width='stretch')

    with st.expander('Kenapa kamar tidur, kamar mandi, dan garasi tidak dipakai?'):
        st.write('Kamar sangat berkorelasi dengan luas bangunan (0,81 sampai 0,85), sehingga informasinya sudah '
                 'terwakili. Fitur ini **redundan, bukan tidak relevan**. Setelah dibuang, akurasi model ensemble '
                 'hampir tidak berubah, tetapi modelnya jauh lebih sederhana.')

    st.subheader('Seberapa akurat model ini?')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Median selisih', 'Rp 284 juta')
    m2.metric('MAE', 'Rp 1,19 miliar', help='Rata-rata selisih dalam rupiah. Ditarik naik oleh rumah mahal dan kavling.')
    m3.metric('MAPE', '32,6%')
    m4.metric('R²', '0,741')
    st.write('- **54%** prediksi meleset 20% atau kurang, dan **70%** meleset 30% atau kurang.\n'
             '- **Rumah:** R² 0,875 dan rata-rata meleset 30%. **Kavling:** lebih lemah, rata-rata meleset 55%.\n'
             '- Model hanya mengetahui luas dan lokasi. Kondisi bangunan, tahun dibangun, akses jalan, dan '
             'sertifikat memengaruhi harga tetapi tidak ada di data.')

with tab_visual:
    ukuran = st.radio('Tampilkan per Kecamatan:', ['Median harga', 'Median harga per m² tanah'], horizontal=True)
    sat = 1e9 if ukuran == 'Median harga' else 1e6
    satuan = 'miliar Rp' if ukuran == 'Median harga' else 'juta Rp per m²'
    urut = stat_kec.assign(Nilai=stat_kec[ukuran] / sat).sort_values('Nilai')
    fig1 = px.bar(urut, x='Nilai', y='Kecamatan', orientation='h', color='Nilai', hover_data=['Jumlah listing'],
                  labels={'Nilai': f'{ukuran} ({satuan})'}, title=f'{ukuran} per Kecamatan')
    fig1.update_layout(height=720, coloraxis_showscale=False)
    st.plotly_chart(fig1, width='stretch')

    contoh = data.dropna(subset=['Luas Tanah']).sample(3000, random_state=42)
    fig2 = px.scatter(contoh, x='Luas Tanah', y='Harga', color='Jenis', log_x=True, log_y=True, opacity=0.45,
                      title='Harga vs Luas Tanah (sampel 3.000 listing, skala log)')
    if h:
        fig2.add_scatter(x=[h['lt']], y=[h['harga']], mode='markers', name='Prediksi Anda',
                         marker=dict(color='red', size=15, symbol='star'))
    st.plotly_chart(fig2, width='stretch')
    st.caption('Semakin luas tanahnya, semakin mahal harganya. Bintang merah muncul setelah Anda menekan prediksi.')

with tab_proses:
    st.subheader('Dari data mentah ke model')
    corong = pd.DataFrame({'Tahap': ['Data mentah', 'Perbaikan harga', 'Hapus duplikat', 'Buang nilai tidak wajar',
                                     'Siap dimodelkan'],
                           'Baris': [40200, 40145, 27791, 27242, 27224]})
    st.plotly_chart(px.funnel(corong, x='Baris', y='Tahap', title='Jumlah listing di setiap tahap'),
                    width='stretch')
    st.caption('Penyusutan terbesar berasal dari duplikat (12.354 baris), bukan dari data kosong.')

    st.subheader('Perbandingan R² antar tahap')
    tabel = pd.DataFrame({
        'Data mentah (kontrol)': [-73.64, -73.38, -9.63, -4.04, -0.06],
        'Setelah preprocessing (7 fitur)': [0.559, 0.560, 0.708, 0.713, 0.742],
        'Setelah seleksi fitur (3 fitur)': [0.650, 0.650, 0.714, 0.709, 0.736],
        'Setelah tuning': [None, None, 0.714, 0.741, 0.650]},
        index=['Linear Regression', 'Ridge', 'Decision Tree', 'Random Forest', 'Hist Gradient Boosting'])
    st.dataframe(tabel.style.format('{:.3f}', na_rep='-'), width='stretch')
    st.caption('Random Forest hasil tuning dipilih karena peringkat rata-rata terbaik di MAE, MAPE, dan R².')