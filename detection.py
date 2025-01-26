import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Read data
data = pd.read_csv('News.csv', index_col=0)
data = data.drop(["title", "date"], axis=1)

# Data cleaning
data_cleaned = data.dropna()
data_cleaned = data_cleaned.reset_index(drop=True)

# Data selection
column_subject = 'subject'
datap1 = data_cleaned[data_cleaned["subject"].isin(["politics"])]
datap2 = data_cleaned[data_cleaned["subject"].isin(["politicsNews"])]
data_politicsall1 = data_cleaned[data_cleaned["subject"].isin([ "politics", "politicsNews"])]
data_politicsall = data_politicsall1[0:13676]
data_politicsall = data_politicsall.drop(["subject"], axis = 1)

# Shuffle data
data_politicsall = data_politicsall.sample(frac=1).reset_index(drop=True)
data_healthall1 = data_cleaned[data_cleaned["subject"].isin(["health"])]
data_healthall = data_healthall1.drop(["subject"], axis = 1)

# Shuffle health data
data_healthall = data_healthall.sample(frac=1).reset_index(drop=True)

# Countplot for politics
sns.countplot(data=data_politicsall, x='class', order=data['class'].value_counts().index)
plt.show()

# Preprocessing function
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in tqdm(text_data):
        # Ensure sentence is a string
        sentence = str(sentence)  # Convert non-strings to strings
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
        sentence = ' '.join(token.lower()
                            for token in sentence.split()
                            if token not in stopwords.words('english'))  # Remove stopwords
        preprocessed_text.append(sentence)
    return preprocessed_text


preprocessed_review = preprocess_text(data['text'].values)
data['text'] = preprocessed_review

# Wordcloud for real and fake news
consolidated = ' '.join(word for word in data['text'][data['class'] == 1].astype(str))
wordCloud = WordCloud(width=1100, height=900, random_state=21, max_font_size=110, collocations=False)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('on')
plt.show()

# Wordcloud for fake news
consolidated = ' '.join(word for word in data['text'][data['class'] == 0].astype(str))
wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()

# Get top N words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(data['text'], 20)
df1 = pd.DataFrame(common_words, columns=['Review', 'count'])

df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), xlabel="Top Words", ylabel="Count", title="Bar Chart of Top Words Frequency")
plt.show()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

# Logistic Regression model
modelLR = LogisticRegression()
modelLR.fit(x_train, y_train)

# Print accuracy for Logistic Regression
print(accuracy_score(y_train, modelLR.predict(x_train)))
print(accuracy_score(y_test, modelLR.predict(x_test)))

# Decision Tree model
modelDCT = DecisionTreeClassifier()
modelDCT.fit(x_train, y_train)

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, modelLR.predict(x_test))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()

# Confusion Matrix for Decision Tree
cm = confusion_matrix(y_test, modelDCT.predict(x_test))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()

# Testing on health data
X_test = data_healthall['text']
Y_test = data_healthall['class']
X_test = vectorization.transform(X_test)

# Accuracy score for Logistic Regression on health data
print(accuracy_score(Y_test, modelLR.predict(X_test)))

# Confusion matrix for Logistic Regression on health data
cm = confusion_matrix(Y_test, modelLR.predict(X_test))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()

# Accuracy score for Decision Tree on health data
print(accuracy_score(Y_test, modelDCT.predict(X_test)))

# Testing with a custom input
input = ["Post about a video claims that it is a protest against confinement in the town"]
input_data = vectorization.transform(input)
prediction = modelLR.predict(input_data)

print(prediction)

if prediction[0] == 1:
    print('Real news')
else:
    print('Fake news')
