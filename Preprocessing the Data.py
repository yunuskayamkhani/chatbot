1
# lemmaztize and lower each word and remove duplicates
2
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
3
words = sorted(list(set(words)))
4
# sort classes
5
classes = sorted(list(set(classes)))
6
# documents = combination between patterns and intents
7
print (len(documents), "documents")
8
# classes = intents
9
print (len(classes), "classes", classes)
10
# words = all words, vocabulary
11
print (len(words), "unique lemmatized words", words)
12
13
pickle.dump(words,open('words.pkl','wb'))
14
pickle.dump(classes,open('classes.pkl','wb'))