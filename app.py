from flask import Flask, render_template, request, flash, redirect, session, make_response, url_for
from pprint import pprint
import uuid
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import nltk
import seaborn as sns

import re
from indoNLP.preprocessing import remove_html, remove_url, replace_slang, replace_word_elongation, remove_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

app = Flask(__name__)
app.secret_key = '@#$123456&*()'

ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER_CSV'] = os.path.join(os.getcwd(), 'static/csv')
app.config['UPLOAD_FOLDER_IMAGES'] = os.path.join(os.getcwd(), 'static/images')

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
    text = stemmer.stem(text)

    return text

# Function untuk menghandle upload file csv pada form evaluasi dan upload csv
def handle_upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('File tidak ada', 'danger')
        return redirect(url_for('upload_csv'))

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'danger')
        return redirect(url_for('upload_csv'))
    if not allowed_file(file.filename): # file extension is not .csv
        flash('Ekstensi file harus .csv', 'danger')
        return redirect(url_for('upload_csv'))

    filename = session['uuid'] + '-' + 'upload_csv.csv'
    file.save(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))

    return filename

# =================================================================

# ========================== Middleware ===========================

# Create middleware to set uuid to differentiate between users
@app.before_request
def hook():
    if session.get('uuid') is None:
        session['uuid'] = str(uuid.uuid4())

# =================================================================
@app.route('/')
def index():
    
    return render_template('index.jinja2', klasifikasi=True)

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        filename = handle_upload()
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))

        ## Text positif
        # For every token that is in stopwords, remove it
        text_positif = ' '.join(remove_stopwords(text) for text in df[df['predicted_label'] == 1]['preprocessed_content']).split()

        # # Using nltk find all the frequencies
        text_positif_freq = nltk.FreqDist(text_positif).most_common(10)

        text_positif_freq = pd.Series(dict(text_positif_freq))

        fig, ax = plt.subplots(figsize=(10,10))

        ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
        posiif_plot = sns.barplot(x=text_positif_freq.index, y=text_positif_freq.values, ax=ax)
        plt.xticks(rotation=30)
        plt.xlabel('Kata')
        plt.xlabel('Frekuensi')
        frekuensi_positif_filepath = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], session['uuid'] + '-' + 'frekuensi_positif.png')
        plt.savefig(frekuensi_positif_filepath)

        ## text negatif
        # For every token that is in stopwords, remove it
        text_negatif = ' '.join(remove_stopwords(text) for text in df[df['predicted_label'] == 0]['preprocessed_content']).split()

        # # Using nltk find all the frequencies
        text_negatif_freq = nltk.FreqDist(text_negatif).most_common(10)

        text_negatif_freq = pd.Series(dict(text_negatif_freq))

        fig, ax = plt.subplots(figsize=(10,10))

        ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
        posiif_plot = sns.barplot(x=text_negatif_freq.index, y=text_negatif_freq.values, ax=ax)
        plt.xticks(rotation=30)
        plt.xlabel('Kata')
        plt.xlabel('Frekuensi')
        frekuensi_negatif_filepath = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], session['uuid'] + '-' + 'frekuensi_negatif.png')
        plt.savefig(frekuensi_negatif_filepath)

        return render_template('hasil_upload_csv.jinja2', upload_csv=True, 
                frekuensi_positif=session['uuid'] + '-' + 'frekuensi_positif.png',
                frekuensi_negatif=session['uuid'] + '-' + 'frekuensi_negatif.png')

    return render_template('upload_csv.jinja2', upload_csv=True)

# endpoint untuk mendapatkan data dari request ajax
@app.route('/hasil_upload_csv', methods=['GET', 'POST'])
def hasil_upload_csv():
    if request.method == 'POST':
        filename = session['uuid'] + '-' + 'upload_csv.csv'
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))

        result = df.to_json()
        parsed = json.loads(result)

        return json.dumps(parsed, indent=4)

@app.route('/evaluasi', methods=['GET', 'POST'])
def evaluasi():
    if request.method == "POST":
        filename = handle_upload()
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))
        
        # confusion matrix
        cm = confusion_matrix(df['label'], df['predicted_label'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        plt.title('confusion matrix')
        confusion_matrix_filepath = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], session['uuid'] + '-' + 'confusion_matrix.png')
        plt.savefig(confusion_matrix_filepath)

        return render_template('hasil_evaluasi.jinja2', confusion_matrix=session['uuid'] + '-' + 'confusion_matrix.png')

    return render_template('evaluasi.jinja2', evaluasi=True)

@app.route('/hasil_evaluasi', methods=['GET', 'POST'])
def hasil_evaluasi():
    if request.method == 'POST':
        filename = session['uuid'] + '-' + 'upload_csv.csv'
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER_CSV'], filename))

        result = df.to_json()
        parsed = json.loads(result)

        return json.dumps(parsed, indent=4)

if __name__ == "__main__":
    app.run(debug=True)