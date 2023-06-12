# SENTIMENT ANALYSIS ON AMAZON REVIEWS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tqdm

# stylesheet for plots
plt.style.use('ggplot')

# Read data
df = pd.read_csv('Reviews.csv')
# down sample it by taking just few rows like 500
df = df.head(500)

# random value
example = (df['Text'][50])

# nltk.download('punkt')
tokens = (nltk.word_tokenize(example))
print(tokens[:10])
nltk.download('averaged_perceptron_tagger')

# getting part of speech
tagged = nltk.pos_tag(tokens)
print(tagged[:10])

# nltk.download('maxent_ne_chunker')
# nltk.download('words')
print(nltk.chunk.ne_chunk(tagged))

# nltk.download('vader_lexicon')

# creating object
sia = SentimentIntensityAnalyzer()

result = {}
# Run the polarity score on the entire dataset
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    result[myid] = sia.polarity_scores(text)

# create dataframe
vaders = pd.DataFrame(result).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
# now we have sentiment score and metadata
vaders = vaders.merge(df, how='left')
print(vaders.columns)

# plotting and analysis
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(ax=ax[0], data=vaders, x='Score', y='pos')
ax[0].set_title('Positive')
sns.barplot(ax=ax[1], data=vaders, x='Score', y='neu')
ax[1].set_title('Neutral')
sns.barplot(ax=ax[2], data=vaders, x='Score', y='neg')
ax[2].set_title('Negative')
plt.show()
