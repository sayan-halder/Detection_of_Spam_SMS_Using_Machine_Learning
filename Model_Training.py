import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# Get Sms Dataset
sms = pd.read_csv('Spam_SMS_Dataset', sep='\t', names=['label', 'message'])

# Delete the duplicate records if present
sms.drop_duplicates(inplace=True)
sms.reset_index(drop=True, inplace=True)

# Cleaning the messages
corpus = []
ps = PorterStemmer()

for i in range(0, sms.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms.message[i])  # Cleaning special character from the message
    message = message.lower()  # Converting the entire message into lower case
    words = message.split()  # Tokenizing the review by words
    words = [word for word in words if word not in set(stopwords.words('english'))]  # Removing the stop words
    words = [ps.stem(word) for word in words]  # Stemming the words
    message = ' '.join(words)  # Joining the stemmed words
    corpus.append(message)  # Building a corpus of messages

# Creating the Bag of Words model
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

# Extracting dependent variable from the dataset
y = pd.get_dummies(sms['label'])
y = y.iloc[:, 1].values

# Split Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy Score
acc_s = accuracy_score(y_test, y_pred) * 100
print("Accuracy Score: {} %".format(round(acc_s, 2)))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#######################################################################################################################
# precision = precision_score(y_test, y_pred) * 100
# print("\n\nPrecision: {} %".format(round(precision, 2)))
# recall = recall_score(y_test, y_pred) * 100
# print("Recall: {} %".format(round(recall, 2)))
# f1 = f1_score(y_test, y_pred) * 100
# print("F1-Score: {} %".format(round(f1, 2)))
#
# true_positives = confusion_matrix(y_test, y_pred)[1, 1]
# print("\n\nTrue Positives:",true_positives)
# false_positives = confusion_matrix(y_test, y_pred)[0, 1]
# print("False Positives:",false_positives)
# false_negatives = confusion_matrix(y_test, y_pred)[1, 0]
# print("False Negetives:",false_negatives)
# true_negetives = confusion_matrix(y_test, y_pred)[0, 0]
# print("True Negetives:",true_negetives)
#
# precision = (true_positives) / (true_positives + false_positives) * 100
# print("\n\nPrecision: {} %".format(round(precision, 2)))
# recall = (true_positives) / (true_positives + false_negatives) * 100
# print("Recall: {} %".format(round(recall, 2)))
# f1 = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives) * 100
# print("F1-Score: {} %".format(round(f1, 2)))
#######################################################################################################################

# Saving the model
f1 = open('Classifier.pickle', 'wb')
pickle.dump(classifier, f1)
f1.close()

f2 = open('CountVectorizer.pickle', 'wb')
pickle.dump(cv, f2)
f2.close()

# Create a DataFrame with the cleaned messages and their labels
cleaned_sms = pd.DataFrame({'message': corpus, 'label': sms['label']})

# Bar plot for the distribution of spam and non-spam messages
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=cleaned_sms)
plt.title('Distribution of Spam and Non-Spam Messages')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Histogram for the message lengths in the dataset
cleaned_sms['message_length'] = cleaned_sms['message'].apply(lambda x: len(x))
plt.figure(figsize=(8, 5))
sns.histplot(data=cleaned_sms, x='message_length', bins=50)
plt.title('Distribution of Message Lengths')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.show()

# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Receiver Operating Characteristic (ROC) curve
# y_probs = classifier.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_probs)
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
