import streamlit as st
import pandas as pd
from datetime import date
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import regex as re
import spacy
import textblob
from textblob import TextBlob
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk.corpus import stopwords
from collections import Counter
from nrclex import NRCLex


import plotly.express as px
import plotly.figure_factory as ff


from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


stopwords = ['more','better','a','the','to','and','for','of','that','are','i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']

color_map = st.selectbox(
'ColorMap',
('RdYlGn','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'))



st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('wideMode' , True)

punctuation = string.punctuation
#spacy.load('en_core_web_sm')
#sp = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer(language='english')
#stopwords = spacy.lang.en.stop_words.STOP_WORDS

## Helper Functions
def clean_text_all(s):
  #Cleans text
  #Lemmatize
  #Spell Checker
  if isinstance(s, int) or isinstance(s, float):
    return('')
  #Lowercase
  t = s.lower()

  #punctuation
  
  t = t.translate(str.maketrans('', '', punctuation))
  ## Remove whitespaces after this
  t = re.sub(r'\d+', '', t)


  #Remove Stopwords
  t = ' '.join([word for word in t.split(' ') if word.lower() not in stopwords])

  #Lemmatize
  
  #t= ' '.join([token.lemma_ for token in sp(t)])

  #Remove Whitespace
  t=' '.join(t.split())
  return t

  #Spell Checker
  #Somehow not working - need to fix
  #textBlb = TextBlob(t)     
  #textCorrected = textBlb.correct()
  
  
  return(textCorrected)


def sentiment_textblob(text):
	p = TextBlob(text).sentiment.polarity
	s = TextBlob(text).sentiment.subjectivity
	return p


sentiment = SentimentIntensityAnalyzer()

def sentiment_vader(text):
	s = sentiment.polarity_scores(text)
	return s['compound']




## Main Page Configuration
st.write("""
#### Sentiment Analysis for Hotel Reviews
	""")

st.markdown(' ** Upload CSV file 👇 **')

@st.cache(allow_output_mutation=True)
def load_data(file):
	df = pd.read_csv(file, encoding='utf-8', nrows=50)
	return df

uploaded_file = st.file_uploader("", type="csv", key='file_uploader')


if uploaded_file is not None:
	df = load_data(uploaded_file)
	st.write(df.head(3))
	st.write("File uploaded")

	cols= df.columns

	df_clean=df

	with st.form("ChooseColumn",clear_on_submit=False):
		st.write("Select the column on which you want to run the sentiment mining")

		colname = st.selectbox("Select column", (cols))

		run_button = st.form_submit_button("Submit") 

		if run_button:
			st.write(colname)
			
			df_clean[colname+'_Cleaned']=df_clean[colname].apply(lambda x: clean_text_all(x))

			st.write(colname + " Cleaned")
			st.write(df_clean.head(2))

			str_tweet = ','.join(str(item) for item in df_clean[colname+'_Cleaned'])
			text_object = NRCLex(str_tweet)
			data = text_object.raw_emotion_scores
			emotion_df = pd.DataFrame.from_dict(data, orient='index')
			emotion_df = emotion_df.reset_index()
			emotion_df = emotion_df.rename(columns={'index' : 'Emotion Classification' , 0: 'Emotion Count'})
			emotion_df = emotion_df.sort_values(by=['Emotion Count'], ascending=False)
			fig = px.bar(emotion_df, x='Emotion Count', y='Emotion Classification', color = 'Emotion Classification', orientation='h', width = 800, height = 400)
			st.plotly_chart(fig, use_container_width=True)


			affect_df = pd.DataFrame.from_dict(text_object.affect_dict, orient='index')

			affect_freq = pd.DataFrame.from_dict(text_object.affect_frequencies, orient='index')
			affect_freq=affect_freq.reset_index()
			affect_freq = affect_freq.rename(columns={'index' : 'EmotionAffect' , 0: 'Frequency'})
			affect_freq = affect_freq.sort_values(by=['Frequency'], ascending=False)

			fig1 = px.bar(affect_freq, x='EmotionAffect', y='Frequency', color = 'EmotionAffect')
			st.plotly_chart(fig1, use_container_width=True)



	with st.form("SentimentAnalysis",clear_on_submit=False):
		algoname = st.selectbox("Choose the sentiment Analysis Algorithm",('TextBlob','VADER'))

		sentiment_button = st.form_submit_button("Submit") 

		if sentiment_button:
			
			df_clean[colname+'_Cleaned']=df_clean[colname].apply(lambda x: clean_text_all(x))
			if algoname=='TextBlob':
				df_clean['Sentiment_polarity_'+algoname] = df_clean[colname + "_Cleaned"].apply(lambda x: sentiment_textblob(x))
				
				df_clean['Sentiment_tag_'+algoname] = df_clean['Sentiment_polarity_'+algoname].apply(lambda x: 'pos' if x>0.5 else 'neg' if x<-0 else 'neu')
				#sentiment_summary = df_clean.groupby(by=['Sentiment_tag_'+algoname]).count()
				st.write(df_clean)

			else:
				df_clean['Sentiment_polarity_'+algoname] = df_clean[colname + "_Cleaned"].apply(lambda x: sentiment_vader(x))
				
				df_clean['Sentiment_tag_'+algoname] = df_clean['Sentiment_polarity_'+algoname].apply(lambda x: 'pos' if x>0.7 else 'neg' if x<-0.3 else 'neu')
				st.write(df_clean)
				#sentiment_summary = df_clean.groupby(by=['Sentiment_tag_'+algoname]).count()
				#st.write(sentiment_summary)
	@st.cache
	def convert_df(df):
		# IMPORTANT: Cache the conversion to prevent computation on every rerun
		return df.to_csv().encode('utf-8')

	csv = convert_df(df_clean)


	st.download_button(
	label="Download data as CSV",
	data=csv,
	file_name='SentimentAnalysis.csv',
	mime='text/csv',
	)







