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



st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('wideMode' , True)


stopwords = ['more','better','a','the','to','and','for','of','that','are','i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']

color_map = st.selectbox(
'ColorMap',
('RdYlGn','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'))

## Main Page Configuration
st.title('Text mining app')

st.write("""

## Build your text analytics module in Real Time
### Drill down from the left pane


	""")


#Read Data
#df = pd.read_csv('./data/sampleData.csv')


def generate_wordcloud(data, title, mask=None,colormap='RdYlGn'):
    cloud = WordCloud(scale=3,
                      max_words=150,
                      colormap=color_map,
                      mask=None,
                      background_color='white',
                      stopwords=stopwords,
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
    st.pyplot()

punctuation = string.punctuation
#spacy.load('en_core_web_sm')
#sp = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer(language='english')
#stopwords = spacy.lang.en.stop_words.STOP_WORDS

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


def word_frequency(sentence):
	# joins all the sentenses
	#sentence =" ".join(sentence)
	# creates tokens, creates lower class, removes numbers and lemmatizes the words
	new_tokens = word_tokenize(sentence) 
	
	new_tokens = [t.lower() for t in new_tokens]
	new_tokens =[t for t in new_tokens if t not in stopwords]
	new_tokens = [t for t in new_tokens if t.isalpha()]
	lemmatizer = WordNetLemmatizer()
	new_tokens =[lemmatizer.lemmatize(t) for t in new_tokens]
	#counts the words, pairs and trigrams
	counted = Counter(new_tokens)
	counted_2= Counter(ngrams(new_tokens,2))
	counted_3= Counter(ngrams(new_tokens,3))
	#creates 3 data frames and returns thems
	word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
	word_pairs =pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
	trigrams =pd.DataFrame(counted_3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)
	return word_freq,word_pairs,trigrams




#uploaded_file = st.file_uploader("Choose a file")

st.markdown(' ** Upload CSV file 👇 **')

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8', nrows=50)
    return df

uploaded_file = st.file_uploader("", type="csv", key='file_uploader')

filter=None
master_filters=['None']
if uploaded_file is not None:
    df = load_data(uploaded_file)
    # Can be used wherever a "file-like" object is accepted:
    #df = pd.read_csv(uploaded_file)



    st.write(df.head(3))
    st.write("File uploaded")
    st.session_state['df'] = df

    
    cols= df.columns

    

        
        


    with st.form("AllFilters",clear_on_submit=False):
        all_filters = st.multiselect('Enter the filters',(cols))

        
        for y in all_filters:
            master_filters.append(y)


        all_filters_submit = st.form_submit_button("Submit")
        if all_filters_submit:
            st.selectbox("All Filters",master_filters)        


    
    # st.write("""
    # #### Add Aspects to your data / Flags to your data
	# """)
    # aspect_array = []
    # with st.form("my_form",clear_on_submit=True):
    #     aspect_name = st.text_input('Add Aspect')
    #     aspect_name_cleaned= aspect_name.lower().replace(" ","_")
    #     keywords = st.text_input('Enter Flags Separated by comma')

    #     apply_on = st.selectbox('Apply on Column',(cols))



    #     submitted = st.form_submit_button("Submit")
    #     if submitted:
    #         keywords_break = keywords.split(',')
    #         keyword_list = []
    #         for e in keywords_break:
    #             e=e.replace(" ","")
    #             e=e.lower().strip()
    #             keyword_list.append(e)
    #         st.write("Inside the form")
    #         st.write(apply_on)
    #         def check_keyword(text):
    #             flag = 0
    #             for word in keyword_list:
    #                 if text.find(word)!=-1:
    #                     return 1
    #                     break
    #             return 0
        
    #         df[aspect_name_cleaned] = df[apply_on+'_Cleaned'].apply(lambda x: check_keyword(x))
    #         st.write(df.head(10))
    #         aspect_array.append(aspect_name_cleaned)
    #         master_filters.append(aspect_name_cleaned)

    # st.selectbox("All Filters",master_filters)             




    #### Sentiment Analysis - review Data        
    with st.form("RunApp",clear_on_submit=False):
        st.write("Select the column on which you want to run the algo")

        filter = st.selectbox("Select column", (cols))

        column_filter = st.selectbox("Select Filter", master_filters)

        

        if column_filter != 'None':
            unique_values = df[column_filter].unique()
            unique = st.selectbox("Select",unique_values)   

        run_button = st.form_submit_button("Refresh")
        run_button = st.form_submit_button("Submit") 

        if run_button:
            st.write(filter)
            #filter = 'Reason_to_Visit'


            #Filter the dataframe on unique
            newdf = df[df[column_filter]==unique]

            newdf[filter+'_Cleaned']=newdf[filter].apply(lambda x: clean_text_all(x))
            newdf[filter+'_Cleaned'].fillna('',inplace=True)
            st.write('Cleaned Data')
            
            st.write(newdf.head(10))            
            
            ## Helper Functions

            ## Wordcloud
            word_cloud_data =newdf[filter+'_Cleaned']
            long_string = ','.join(list(word_cloud_data.values))
            long_string=long_string.replace('nan', '')

    

            generate_wordcloud(long_string, 'WordCloud', mask=None)


            ## Word Frequeccy
            word_freq,word_pairs,trigrams = word_frequency(long_string)


            word_freq_20 = word_freq[0:25]
            word_pairs_20 = word_pairs[0:25]
            trigrams_20 = trigrams[0:25]


            ## Plot 

            fig, axes = plt.subplots(3,1,figsize=(8,20))
            sns.barplot(ax=axes[0],x='frequency',y='word',data=word_freq_20).set(title='How many times a word comes in the text')
            sns.barplot(ax=axes[1],x='frequency',y='pairs',data=word_pairs_20).set(title='How many times two words occur together')
            sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=trigrams_20).set(title='How many times three words occur together')
            st.pyplot(fig)
    



        
        #Apply on the dataframe

        


