from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))

new_text = "John jumped and was jumping to jump but jumps often jumply"

print("un stemmed = ", new_text)

words = word_tokenize(new_text)

stop_words = set(stopwords.words("english"))

words_no_stop_words = []

for w in words:
    if w not in stop_words:
       words_no_stop_words.append(w)

print("New Text with no stop words = ", words_no_stop_words)

for words in words_no_stop_words :
    print(" Stemmed = ", ps.stem(words))
