import feedparser
import nltk
import re
import spacy
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from pprint import pprint

# Step 1: Load RSS feed from Indian Express
def fetch_articles(rss_url):
    feed = feedparser.parse("https://indianexpress.com/section/opinion/40-years-ago/feed/")
    articles = [entry['title'] + " " + entry.get('summary', '') for entry in feed.entries]
    return articles

# Example RSS feed
rss_url = "https://indianexpress.com/section/india/feed/"
articles = fetch_articles(rss_url)

# Step 2: Preprocessing
nltk.download('stopwords')
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def preprocess(texts):
    texts_clean = []
    for doc in texts:
        doc = re.sub(r'\s+', ' ', doc)  # remove extra whitespace
        doc = re.sub(r'\W+', ' ', doc)  # remove punctuation
        doc = doc.lower()
        tokens = [token.lemma_ for token in nlp(doc) if token.lemma_ not in stop_words and len(token) > 3]
        texts_clean.append(tokens)
    return texts_clean

processed_data = preprocess(articles)

# Step 3: Dictionary and Corpus
id2word = corpora.Dictionary(processed_data)
corpus = [id2word.doc2bow(text) for text in processed_data]

# Step 4: LDA Model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=id2word,
                                   num_topics=5,
                                   random_state=100,
                                   update_every=1,
                                   chunksize=10,
                                   passes=10,
                                   alpha='auto',
                                   per_word_topics=True)

# Step 5: Display Topics
print("🔍 Topics found:")
pprint(lda_model.print_topics())
