
import nltk
import wikipedia
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

# Open the file containing the positive words and negative words
with open('positive-words.txt', 'r') as file:
    positive_words = [line.strip() for line in file]
with open('negative-words.txt', 'r') as file:
    negative_words = [line.strip() for line in file]

# Get summary
def get_summary(name):
    text = wikipedia.page(name).content
    summary = summarize(text,ratio=0.03)
    return summary

# Get keywords
def get_keywords(text):
    key_words = nltk.word_tokenize(keywords(text,ratio=0.01))
    lmt = WordNetLemmatizer()  # Lemmatizer
    porter = PorterStemmer()   # Stemmer
    kw = [lmt.lemmatize(item) if lmt.lemmatize(item).endswith('e') 
          else porter.stem(item) for item in key_words]
    kw = list(set(kw))

    # Find the words with same stemmer, keep the words end with 'e'
    duplicate = []
    for i in range(len(kw)):
        for j in range(i+1,len(kw)):
            if porter.stem(kw[i]) == porter.stem(kw[j]):
                a = kw[i]
                b = kw[j]
                for word in [a,b]:
                    if word.endswith('e'):
                        pass
                    else:
                        duplicate.append(word)
                if a.endswith('e') and b.endswith('e'):
                    duplicate.append(b)

    # Remove the words with duplicate stemmer, which doesn't end with 'e'
    kw  = list(set(kw) - set(duplicate)) 
    return ', '.join(kw)




"""
How to define the word is negative or positive?
 - Negative sentiment: 
   - negative words not preceded by a negation within n words in the same sentence.
   - positive words preceded by a negation within n words in the same sentence.
 - Positive sentiment (in the similar fashion):
   - positive words not preceded by a negation within n words in the same sentence.
   - negative terms following a negation within n words in the same sentence

"""
# Get the negative labels and positive labels
def get_PosNegWords(text):
    
    # Not include all negations
    negations=['not', 'too', 'n\'t', 'no', 'cannot', 'neither','nor']  
    tokens = nltk.word_tokenize(text)  
    positive_tokens=[]
    negative_tokens=[]
    for i, token in enumerate(tokens):
        # When there is a positive word
        if token in positive_words:
            judger1 = True
            if i > 0:
                # A negation within N words? 
                idx1 = i
                while judger1 is True:
                    idx1 = idx1 - 1
                    if idx1 == 0 or tokens[idx1] in ',.!?' or tokens[idx1] == 'and':  
                        break
                    elif tokens[idx1] in negations:
                        negative_tokens.append(tokens[idx1]+ ' ' + token)
                        judger1 = False
                if judger1 is True:
                    positive_tokens.append(token)
            else:
                positive_tokens.append(token)

        # When there is a negative word
        elif token in negative_words:
            judger2 = True
            if i > 0:
                # A negation within N words? 
                idx2 = i
                while judger2 is True:
                    idx2 = idx2 - 1
                    if idx2 == 0 or tokens[idx2] in ',.!?' or tokens[idx2] == 'and': 
                        break
                    elif tokens[idx2] in negations:
                        positive_tokens.append(tokens[idx2]+ ' ' + token)
                        judger2 = False
                if judger2 is True:
                    negative_tokens.append(token)

            else:
                negative_tokens.append(token)
        
    return list(set(positive_tokens)), list(set(negative_tokens))




