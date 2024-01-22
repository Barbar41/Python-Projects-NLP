#################################################################
# WIKIPEDIA Metin Ön işleme ve Görselleştirme
#################################################################

##############################################
# İş Problemi
##############################################
# Wikipedia metinleri içeren veri setine metin ön işleme ve görselleştirme yapınız
#############################################

##########################
# Proje Görevleri
##########################
# Gerekli kütüphane ve ön ayarlar

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#####################################################################################################
# Görev 1: Metin Ön İşleme
######################################################################################################
df = pd.read_csv("nlp/datasets/wiki_data.csv", index_col=0)
df.shape
df.head()

# Gözlemleyelim bir tanesini
df['text'][1]

# Pc yormamak için küçültelim aralığı
df= df[:2000]
df.shape

# Adım 1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz. Fonksiyon;
#        • Büyük küçük harf dönüşümü,
#        • Noktalama işaretlerini çıkarma,
#        • Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli.

def clean_text(text):
    #Normalizing Case Folding
    text=text.str.lower()
    #Punctuations
    text=text.str.replace(r'[^\w\s]', ' ')
    text=text.str.replace('\n' , ' ')
    #Numbers
    text = text.str.replace('\d', '')
    return text

# Adım 2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

df["text"]=clean_text(df["text"])

df.head()

# Adım 3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak remove_stopwords adında fonksiyon yazınız.
stop_words= stopwords.words("english")
def remove_stopwords(text):
    stop_words= stopwords.words("english")
    text= text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

# Adım 4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

df["text"] = remove_stopwords(df["text"])
df

# Adım 5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz.

sil=pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]

df["text"]=df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sil))


# Adım 6: Metinleri tokenize edip sonuçları gözlemleyiniz.

df["text"].apply(lambda x: TextBlob(x).words)

# Adım 7: Lemmatization işlemi yapınız

df
df["text"]= df["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df

##################################################################################
# Görev 2: Metin Görselleştirme
##################################################################################

# Adım 1: Metindeki terimlerin frekanslarını hesaplayınız.

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.head()
# Sütun isimlendirmesi
tf.columns = ["words", "tf"]

# Adım 2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.


# 2000 den fazla geçen kelimelerin görselleştiriniz
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Adım 3: Kelimeleri WordCloud ile görselleştiriniz.

# Bulut şeklinde görsel oluşturma.Bütün metini tek bir text gibi ifade edeceğiz.

text = " ".join(i for i in df["text"])
# Wordcloud için görselleştirme özellikleri belirliyoruz.

wordcloud = WordCloud(max_font_size=50,max_words=180,background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


##################################################################################
# Görev 3: Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
##################################################################################
# Adım 1: Metin ön işleme işlemlerini gerçekleştiriniz.

# Adım 2: Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.

# Adım 3: Fonksiyonu açıklayan 'docstring' yazınız.

df = pd.read_csv("nlp/datasets/wiki_data.csv", index_col=0)

def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', ' ')
    text = text.str.replace('\n', ' ')
    # Numbers
    text = text.str.replace('\d', '')
    # Stopwords
    stop_words = stopwords.words("english")
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    #Rarewords/ Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))

    if Barplot:
        # Metin frekanslarının hesaplanması
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.head()
        # Sütun isimlendirmesi
        tf.columns = ["words", "tf"]
        # 5000 den fazla geçen kelimelerin görselleştiriniz
        tf[tf["tf"] > 5000].plot.bar(x="words", y="tf")
        plt.show()
    if Wordcloud:
        # Kelimelerin Birleştirilmesi
        text = " ".join(i for i in text)
        # Wordcloud için görselleştirme özellikleri belirliyoruz.
        wordcloud = WordCloud(max_font_size=50, max_words=180, background_color="black").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)


