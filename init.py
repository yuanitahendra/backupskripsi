import re
import nltk
import numpy as np
import math
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_mysqldb import MySQL
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (ArrayDictionary, StopWordRemover, StopWordRemoverFactory)
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd

app = Flask(__name__)
app.secret_key = 'many random bytes'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'perundungan'

mysql = MySQL(app)

@app.route('/')
def Index():
    cur = mysql.connection.cursor()
    cur.execute("SELECT  * FROM dataset")
    data = cur.fetchall()
    cur.close()

    return render_template('index.html', tweet=data)



@app.route('/training')
def training():
    cur = mysql.connection.cursor()
    # cur.execute("SELECT  * FROM dataset")
    cur.execute("SELECT * FROM coba_tfidf")
    trainDataQuery = "SELECT prepro,label FROM coba_tfidf WHERE keterangan = 'training'"
    cur.execute(trainDataQuery)
    trainings = list(cur.fetchall())
    trainData = [prepro[0] for prepro in trainings if prepro[0]] 
    labelTrainData = [prepro[1] for prepro in trainings if prepro[1]] 
    testDataQuery = "SELECT prepro,label FROM coba_tfidf WHERE keterangan = 'testing' "
    cur.execute(testDataQuery)
    testings = list(cur.fetchall())
    testData = [prepro[0] for prepro in testings if prepro[0]] 
    labelTestData = [prepro[1] for prepro in testings if prepro[1]] 
    # data = cur.fetchall()
    print(labelTrainData)

    # Create feature vectors
    # min_df = len(trainData)+len(testData)
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None)
    train_vectors = vectorizer.fit_transform(trainData)
    test_vectors = vectorizer.transform(testData)
    print(pd.DataFrame(test_vectors.toarray(),columns=vectorizer.get_feature_names()))


    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, labelTrainData)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1
    # results
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    report = classification_report(labelTestData, prediction_linear, output_dict=True)
    print('resport: ', report)
    print('positive: ', report['positif'])
    print('negative: ', report['negatif'])
    print('netral: ', report['netral'])

    cur.close()

    return render_template('training.html')

@app.route('/pembobotan')
def pembobotan():
    cur = mysql.connection.cursor()
    query = """
    SELECT tf.id,t.term,d.id,tf.count,t.id,'dummy'
    FROM term_frequency tf 
    JOIN terms t on t.id=tf.id_term
    JOIN dataset d on d.id=tf.id_tweet
    """
    cur.execute(query)
    data = cur.fetchall()
    # print(data)
    getdata = np.array(data, dtype=object)
    # print(getdata)
    N = len(getdata)
    for i in range(0, len(getdata)):
        cur.execute("SELECT count(*) FROM term_frequency WHERE id_term='{}'".format(getdata[i][4]))
        df = cur.fetchall()[0][0]

        # tf = count/total term in 1 tweet
        tf = getdata[i][3]
        getdata[i][3] = tf

        # idf = log10(N/df)
        # N adalah total seluruh dokumen/tweet
        # df adalah total tweet yg mengandung term
        idf = math.log10(N/df)
        getdata[i][4] = idf

        getdata[i][5] = tf*idf

    cur.close()

    return render_template('pembobotan.html', tweet=getdata)

#pemanggilan tokenizingg
@app.route('/tokenizing')
def tokenizing():
    cur = mysql.connection.cursor()
    cur.execute("SELECT  * FROM dataset")
    data = cur.fetchall()
    getdata = np.array(data, dtype=object)
    for i in range(0, len(getdata)):
        getdata[i][1] = tokenize_func(getdata[i][1])
  
    cur.close()

    return render_template('tokenizing.html', tweet=getdata)

#pemanggilan filtering
@app.route('/filtering')
def filtering():
    cur = mysql.connection.cursor()
    cur.execute("SELECT  * FROM dataset")
    data = cur.fetchall()
    getdata = np.array(data)
    for i in range(0, len(getdata)):
        getdata[i][1] = filtering_func(tokenize_func(getdata[i][1]))
    cur.close()

    return render_template('filtering.html', tweet=getdata)

# pemanggilan stemming
@app.route('/preprocessing')
def stemming():
    cur = mysql.connection.cursor()
    cur.execute("SELECT  * FROM dataset")
    data = cur.fetchall()
    getdata = np.array(data)
    for i in range(0, len(getdata)):
        id_tweet = getdata[i][0]
        getdata[i][1] = stemming_func(id_tweet,filtering_func(tokenize_func(getdata[i][1])))  
        cur.execute("UPDATE dataset set prepro = '"+getdata[i][1]+"' where id='"+getdata[i][0]+"'") 
    mysql.connection.commit()
    cur.close()
    return render_template('preprocessing.html', tweet=getdata, cobas=getdata[0] )

# Tokenization
def tokenize_func(ready_to_tokenize):
    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    text = re.sub(url_pattern, ' ', ready_to_tokenize) #hapus url
    removed_number = re.sub(r"\d+", "", text) # hapus numeric
    siap = re.sub(r'[^\w\s]',' ',removed_number) # hapus caracter 
    space = re.sub(r'\s+', ' ', siap) #menghapus space kosong 
    #membuat huruf menjadi kecil semua dan menghapus spasi di awalan dan akhir kalimat
    siap_token = space.lower().strip().split(" ")
    # hasil berupa array
    return siap_token
    
# filtering
def filtering_func(kalimat):
    # hasil dari tokenize di gabung menjadi string
    kalimat = (" ").join(kalimat)
    stop_factory = StopWordRemoverFactory().get_stop_words() #load defaul stopword
    more_stopword = ['http','net','com','foto','https','pic', 'cc', 'di','yg','â','us','www','lucinta','gmn','hem','jpnn','jd','dah','trs','trus','link','news','ga','knp','twitter'] #menambahkan stopword yg ingin hilangkan 
    data = stop_factory + more_stopword
    dictionary = ArrayDictionary(data)
    str = StopWordRemover(dictionary)

    ready = str.remove(kalimat)
    return ready
    
# stemming
def stemming_func(id_tweet, ready_to_stem):
    cur = mysql.connection.cursor()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    hasil = stemmer.stem(ready_to_stem)

    fdist = FreqDist(hasil.split(' '))
    freqDist = fdist.most_common(fdist.N())

    # cek term
    hasil_count = []
    for term in hasil.split(' '):
        cur.execute("SELECT id,count(*) FROM terms WHERE term = '{}'".format(term))
        terms = cur.fetchall()
    
        # print(terms)
        id_term = 0
        if terms[0][1]==0:
            cur.execute("INSERT INTO terms (term) VALUES ('"+term+"')")
            id_term = cur.lastrowid # id yg terakhir
        else:
            id_term = terms[0][0]

        for freq in freqDist:
            if freq[0] == term:
                term = {
                    "id_term": id_term,
                    "term": term,
                    "count": freq[1]
                }
                # jika term belum ada maka masukan array
                if not term in hasil_count:
                    hasil_count.append(term)

    # print(hasil_count, id_tweet)
    cur.execute("DELETE FROM term_frequency WHERE id_tweet={}".format(id_tweet))
    for word in hasil_count:
        cur.execute("INSERT INTO term_frequency (id_term,id_tweet,count) VALUES ({},{},{})".format(word['id_term'],id_tweet,word['count']))

        
    mysql.connection.commit()
    return hasil       

@app.route('/delete/<string:id_data>', methods = ['GET'])
def delete(id_data):
    flash("Record Has Been Deleted Successfully")
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM dataset WHERE id=%s", (id_data,))
    mysql.connection.commit()
    return redirect(url_for('Index'))

def computeTF(wordList, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordList:
        tfDict[word] = count/float(bowCount)
    return tfDict

if __name__ == "__main__":
    app.run(debug=True)


