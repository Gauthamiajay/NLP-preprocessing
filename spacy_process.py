import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Input text
text = "The striped bats were hanging on their feet for best. They were running, jumping and flying all day."

# Process the text
doc = nlp(text)

# Tokenization
print("🔹 Tokens:")
for token in doc:
    print(token.text, end=" | ")

# Stopword removal
print("\n\n🔹 Tokens without stopwords:")
for token in doc:
    if not token.is_stop and not token.is_punct:
        print(token.text, end=" | ")

# Lemmatization
print("\n\n🔹 Lemmatization:")
for token in doc:
    print(f"{token.text} → {token.lemma_}")

# Stemming note
print("\n🔹 Note: spaCy prefers lemmatization; stemming is not included by default.")
