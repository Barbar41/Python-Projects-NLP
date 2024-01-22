#################################################################
# Amazon Yorumları için Duygu Analizi
#################################################################
##############################################
# İş Problemi
##############################################

# Amazon üzerinden satışlarını gerçekleştiren ev tesktili ve günlük giyim odaklı üretimler yapan
# Kozmos ürünlerine gelen yorumları analiz ederek ve aldığı şikayetlere göre özelliklerini geliştirerek satışlarını artırmayı hedeflemektedir.
# Bu hedef doğrultusunda yorumlara duygu analizi yapılarak etiketlencek ve etiketlenen veri ile sınıflandırma modeli oluşturulacaktır.

################################################
# Veri Seti Hikayesi
################################################

# Veri seti belirli bir ürün grubuna ait yapılan yorumları,yorum başlığını,yıldız sayısını ve yapılan yorumu kaç kişinin faydalı bulduğunu belirten değişkenlerden oluşmaktadır.

# 4 Değişken 5611 Gözlem 489 KB
# Star   :Ürüne verilen yıldız sayısı
# HelpFul:Yorumu faydalı bulan kişi sayısı
# Title  :Yorum içeriğine verilen başlık, kısa yorum
# Review :Ürüne yapılan yorum

##########################
# Proje Görevleri
#########################
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

#######################################################################################################
# Görev 1: Metin Ön İşleme
######################################################################################################
# Adım 1: amazon.xlsx verisini okutunuz.

df = pd.read_excel("nlp/datasets/amazon.xlsx")
df.shape
df.head()

# Adım 2: Review değişkeni üzerinde ;
# -a. Tüm harfleri küçük harfe çeviriniz.
df['Review'] = df['Review'].str.lower()

# -b. Noktalama işaretlerini çıkarınız.
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# -c. Yorumlarda bulunan sayısal ifadeleri çıkarınız.
df['Review'] = df['Review'].str.replace('\d', '')

# -d. Bilgi içermeyen kelimeleri (stopwords) veriden çıkarınız.
stop_words= stopwords.words("english")
df['Review']= df['Review'] .apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))

# -e. 1000'den az geçen kelimeleri veriden çıkarınız.
sil=pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review']=df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# -f. Lemmatization işlemini uygulayınız.
df['Review']= df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################################################################################
# Görev 2: Metin Görselleştirme
##################################################################################
# Adım 1: Barplot görselleştirme işlemi için;
# -a. "Review" değişkeninin içerdiği kelimeleri frekanslarını hesaplayınız, tf olarak kaydediniz
tf = df['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.head()

# -b. tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde
# Sütun isimlendirmesi
tf.columns = ["words", "tf"]

# -c. "tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini tamamlayınız.
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# Adım 2: WordCloud görselleştirme işlemi için;
# -a. "Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydediniz.
text = " ".join(i for i in df.Review)

# -b. WordCloud kullanarak şablon şeklinizi belirleyip kaydediniz.
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud3.png")

# -c. Kaydettiğiniz wordcloud'u ilk adımda oluşturduğunuz string ile generate ediniz.
tr_mask = np.array(Image.open("nlp/tr.png"))
wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)

# -d. Görselleştirme adımlarını tamamlayınız. (figure, imshow, axis, show)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

##################################################################################
# Görev 3: Duygu Analizi
##################################################################################
# Adım 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz.
sia = SentimentIntensityAnalyzer()

# Adım 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarının inceleyiniz;
# -a. "Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız.
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

# -b. İncelenen ilk 10 gözlem için compund skorlarına göre filtrelenerek tekrar gözlemleyiniz.
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# -c. 10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz.
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# -d. "Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e ekleyiniz.
df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df.groupby("Sentiment_Label")["Star"].mean()


# NOT: SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken oluşturulmuş oldu.

##################################################################################
# Görev 4: Makine Öğrenmesine Hazırlık
##################################################################################
# Adım 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olarak ayırınız.
from sklearn.model_selection import train_test_split
# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# Adım 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte;
# -a. TfidfVectorizer kullanarak bir nesne oluşturunuz.
from sklearn.feature_extraction.text import TfidfVectorizer

# -b. Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz.
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)

# -c. Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz.
# Train ve test ler vektöre çevirildi
x_train_tf_idf_word= tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word= tf_idf_word_vectorizer.transform(test_x)

##################################################################################
# Görev 5: Modelleme (Lojistik Regresyon)
##################################################################################
# Train ile fit ediyoruz ve test ile predict ediyoruz

# Adım 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

# Adım 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz;
# -a. Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
y_pred=log_model.predict(x_test_tf_idf_word)

# -b. classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
from sklearn.metrics import classification_report
print(classification_report(y_pred, test_y))

# -c. cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()

# Adım 3: Veride bulunan yorumlardan ratgele seçerek modele sorulması;

# -a. sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçierek yeni bir değere atayınız
random_review = pd.Series(df["Review"].sample(1).values)

# -b. Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
yeni_yorum = TfidfVectorizer().fit(train_x).transform(random_review)

# -c. Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
pred=log_model.predict(yeni_yorum)

# -d. Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
print(f"Review:     {random_review[0]} \n Prediction: {pred}")

# -e. Örneklemi ve tahmin sonucunu ekrana yazdırınız.

text=["terrible product"]
yeni_yorum = TfidfVectorizer().fit(train_x).transform(text)
pred=log_model.predict(yeni_yorum)
print(f"Review:     {text[0]} \n Prediction: {pred}")


text=["terrible curtain"]
yeni_yorum = TfidfVectorizer().fit(train_x).transform(text)
pred=log_model.predict(yeni_yorum)
print(f"Review:     {text[0]} \n Prediction: {pred}")
# pozitif gosterıyor cunku verı cloudunda cok fazla gecıyor bu kelıme..

##################################################################################
# Görev 6: Modelleme (Random Forest)
##################################################################################

# Adım 1: Random Forest modeli ile tahmin sonuçlarının gözlenmesi;
# a. RandomForestClassifier modelini kurup fit ediniz.
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)

# b. Cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()

# c. Lojistik regresyon modeli ile sonuçları karşılaştırınız.

