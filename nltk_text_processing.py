# Step 1: Import required modules
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Step 2: Download required NLTK data (run only once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Step 3: Sample input text
text = "The striped bats were hanging on their feet and eating best fruits in the evening."

# Step 4: Tokenization (bypassing punkt)
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)
print("\n🔹 Step 4: Tokenized Words:")
print(tokens)

# Step 5: Stopword Removal
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\n🔹 Step 5: After Removing Stopwords:")
print(filtered_tokens)

# Step 6: Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\n🔹 Step 6: After Stemming:")
print(stemmed_words)

# Step 7: Lemmatization with POS tagging
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer()
pos_tags = pos_tag(filtered_tokens)
lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
print("\n🔹 Step 7: After Lemmatization (with POS tags):")
print(lemmatized_words)
