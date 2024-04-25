from flask import Flask, render_template, request, flash, redirect, session, make_response, url_for
from pprint import pprint
import uuid
import os
import time
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from google_play_scraper import app, Sort, reviews, reviews_all

# for plotting
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import nltk
import seaborn as sns

# for text preprocessing
import re
from indoNLP.preprocessing import remove_html, remove_url, replace_slang, replace_word_elongation, remove_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

app = Flask(__name__)
app.secret_key = '@#$123456&*()'

ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER_CSV'] = os.path.join(os.getcwd(), r'static\csv')
app.config['UPLOAD_FOLDER_IMAGES'] = os.path.join(os.getcwd(), r'static\images')

# load model tensorflow
model = tf.keras.models.load_model('model')

model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.F1Score(average='micro', threshold=0.5)])

# buat direktori untuk menyimpan csv yang diupload
if not os.path.isdir(app.config['UPLOAD_FOLDER_CSV']): 
    os.makedirs(app.config['UPLOAD_FOLDER_CSV'])

# buat direktori untuk menyimpan gambar yang dibuat
if not os.path.isdir(app.config['UPLOAD_FOLDER_IMAGES']): 
    os.makedirs(app.config['UPLOAD_FOLDER_IMAGES'])

# ==================== Helper functions ==========================

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Function untuk preprocessing sebuah text
# Input: string
# output: preprocessed string
def preprocess_text(text):
    text = str(text)

    ## Hapus karakter non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    ## Case folding (ubah ke huruf kecil)
    text = text.lower()

    ## Hapus tag html
    text = re.sub('<.*?>', '', text)

    ## Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    ## Hapus angka
    text = re.sub(r"\d+", "", text)

    ## Mengganti kata gaul menjadi kata formal
    text = replace_slang(text)

    ## Replace tanda baca dengan whitespace
    text = re.sub(r"[-,.;@#?!&$]+\ *", " ", text)

    ## Hapus whitespace berlebih di awal, antara kata, dan akhir
    text = re.sub(r'\s+', ' ', text)

    ## Menghandle word elongation
    text = replace_word_elongation(text)

    ## Stemming
    text = text.split(' ')
    exclude = ['lemot']
    stemmed_words = []
    for word in text:
        if word in exclude:
            stemmed_words.append(word)
        else:
            stemmed_words.append(stemmer.stem(word))
    text = ' '.join(stemmed_words)

    ## Remove stopwords
    # text = remove_stopwords(text)

    return text

# Function untuk menghandle upload file csv pada form evaluasi dan upload csv
# output: filename csv (concat(session id + filename)) yang disimpan
# Note: harus dalam konteks request
def handle_upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('File tidak ada', 'danger')
        return redirect(url_for('klasifikasi_massal'))

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'danger')
        return redirect(url_for('klasifikasi_massal'))
    if not allowed_file(file.filename): # file extension is not .csv
        flash('Ekstensi file harus .csv', 'danger')
        return redirect(url_for('klasifikasi_massal'))

    filename = session['uuid'] + '-' + 'klasifikasi_massal.csv'
    file.save(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))

    return filename

# Ambil 200 komentar baru dari google play store dan simpan dalam file csv
def get_new_comments():
    REVIEW_COUNT = 200

    result, continuation_token = reviews(
        'com.telkom.indihome.external',
        lang='id',
        country='id',
        sort=Sort.NEWEST,
        count=REVIEW_COUNT
    )

    # Convert json to table
    df = pd.DataFrame(np.array(result), columns=['review'])
    df = pd.DataFrame(df.pop('review').tolist())

    df = df[['content', 'score', 'at']]
    df['at'] = pd.to_datetime(df['at']).dt.date # Change datetime to date

    filename = session['uuid'] + '-' + 'klasifikasi_massal.csv'

    df.to_csv(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename), index=False)

# Membuat plot word_frequency dan menyimpan gambarnya di disk
# Input: dataframe dengan kolom 'preprocessed_content' dan filename dengan ekstensi
# output: filename gambar (concat(session id + filename)) yang disimpan
def create_word_frequency(df, filename):
    # For every token that is in stopwords, remove it
    # df.map(str)     # Make sure text is string utf-8 encoding, or remove_stopwords might err

    text = ' '.join(remove_stopwords(str(text)) for text in df).split()

    # # Using nltk find all the frequencies
    text_freq = nltk.FreqDist(text).most_common(10)
    text_freq = pd.Series(dict(text_freq))
    fig, ax = plt.subplots(figsize=(10,10))

    ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    posiif_plot = sns.barplot(x=text_freq.index, y=text_freq.values, ax=ax)
    plt.xticks(rotation=30)
    plt.xlabel('Kata')
    plt.xlabel('Frekuensi')

    filename = session['uuid'] + '-' + filename
    frekuensi_filepath = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], filename)
    plt.savefig(frekuensi_filepath)

    return filename


# =================================================================

# ========================== Middleware ===========================

# Create middleware to set uuid to differentiate between users
@app.before_request
def hook():
    if session.get('uuid') is None:
        session['uuid'] = str(uuid.uuid4())

# =================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        hasil_preprocessing = preprocess_text(request.form['ulasan'])
        output_model = model.predict([hasil_preprocessing])[0][0]
        hasil_klasifikasi = 'POSITIF' if output_model >= 0.5 else 'NEGATIF'

        return render_template('index.jinja2',
                ulasan=request.form['ulasan'],
                hasil_preprocessing=hasil_preprocessing,
                output_model=output_model if hasil_preprocessing != '' else 'INVALID',
                hasil_klasifikasi=hasil_klasifikasi if hasil_preprocessing != '' else 'INVALID',
                klasifikasi=True)

    return render_template('index.jinja2', klasifikasi=True)

@app.route('/klasifikasi_massal', methods=['GET', 'POST'])
def klasifikasi_massal():
    if request.method == 'POST':
        if request.form['options'] == 'upload-csv':
            handle_upload()
        elif request.form['options'] == 'get-new-comments':
            get_new_comments()

        return render_template('hasil_klasifikasi_massal.jinja2', klasifikasi_massal=True)

    return render_template('klasifikasi_massal.jinja2', klasifikasi_massal=True)

# endpoint untuk mendapatkan data dari request ajax
@app.route('/hasil_klasifikasi_massal', methods=['GET', 'POST'])
def hasil_klasifikasi_massal():
    if request.method == 'POST':
        filename = session['uuid'] + '-' + 'klasifikasi_massal.csv'
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))

        # Rename kolom pada file csv
        df.columns.values[0] = 'content'

        df['preprocessed_content'] = df['content'].map(preprocess_text)
        df = df[df['preprocessed_content'] != '']  # Jika content kosong setelah preprocessing, buang row
        df = df.reset_index(drop=True)

        df['prediction'] = model.predict(np.array(df['preprocessed_content'])) # predict class
        df['predicted_label'] = df.apply(lambda row: (1 if row['prediction'] >= 0.5 else 0), axis=1) # bulatkan prediksi

        frekuensi_positif_filename = create_word_frequency(df[df['predicted_label'] == 1]['preprocessed_content'], 'frekuensi_positif.png')
        frekuensi_negatif_filename = create_word_frequency(df[df['predicted_label'] == 0]['preprocessed_content'], 'frekuensi_negatif.png')

        result = df.to_json()
        parsed = json.loads(result)

        parsed['frekuensi_positif'] = url_for('static', filename='images/' + frekuensi_positif_filename)
        parsed['frekuensi_negatif'] = url_for('static', filename='images/' + frekuensi_negatif_filename)

        return json.dumps(parsed, indent=4)

@app.route('/evaluasi', methods=['GET', 'POST'])
def evaluasi():
    if request.method == "POST":
        handle_upload()
        
        return render_template('hasil_evaluasi.jinja2', evaluasi=True)

    return render_template('evaluasi.jinja2', evaluasi=True)

@app.route('/hasil_evaluasi', methods=['GET', 'POST'])
def hasil_evaluasi():
    if request.method == 'POST':
        filename = session['uuid'] + '-' + 'klasifikasi_massal.csv'
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))

        # Rename kolom pada file csv
        df.columns.values[0] = 'content'
        df.columns.values[1] = 'score'
        df.columns.values[2] = 'label'

        df['preprocessed_content'] = df['content'].map(preprocess_text)
        df = df[df['preprocessed_content'] != '']  # Jika content kosong setelah preprocessing, buang row
        df = df.reset_index(drop=True)

        df['prediction'] = model.predict(np.array(df['preprocessed_content']), batch_size=1024) # predict class
        df['predicted_label'] = df.apply(lambda row: (1 if row['prediction'] >= 0.5 else 0), axis=1) # bulatkan prediksi

        # confusion matrix
        cm = confusion_matrix(df['label'], df['predicted_label'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        plt.title('confusion matrix')
        confusion_matrix_filename = session['uuid'] + '-' + 'confusion_matrix.png'
        confusion_matrix_filepath = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], confusion_matrix_filename)
        plt.savefig(confusion_matrix_filepath)

        result = df.to_json()
        parsed = json.loads(result)

        parsed['binary_accuracy'] = f"{accuracy_score(df['label'], df['predicted_label']):.2%}"
        parsed['f1_score'] = f"{f1_score(df['label'], df['predicted_label']):.2%}"
        parsed['confusion_matrix'] = url_for('static', filename='images/' + confusion_matrix_filename);

        return json.dumps(parsed, indent=4)

if __name__ == "__main__":
    app.run(debug=True)