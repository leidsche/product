import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

df = pd.read_csv('dataset0.csv')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['generated'], test_size=0.2, stratify=df['generated'], random_state=11)

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), max_df=0.9, min_df=2, sublinear_tf=True)
X_trainv = tfidf.fit_transform(X_train)
X_testv = tfidf.transform(X_test)

model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=11)
model.fit(X_trainv, y_train)

prob = model.predict_proba(X_testv)[:, 1]
pred = model.predict(X_testv)
print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
print(f"F1: {f1_score(y_test, pred):.4f}")

joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(model, 'model.pkl')