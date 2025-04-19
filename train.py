import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Example dataset (Replace this with your actual dataset)
X = ["free money now", "urgent call me", "win lottery", "hello friend", "how are you"]
y = [1, 1, 1, 0, 0]  # 1: Spam, 0: Not Spam

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)  # ✅ Fit before saving

# Save trained vectorizer & model
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully!")
