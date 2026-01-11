import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib  # To save the model

# Load the dataset
df = pd.read_csv("nba_draft_predictor/draft.csv")
  # Change to 'data.csv' if that's your file name

print("NBA Draft Success Predictor 🏀")
print(f"Loaded {len(df)} players")
print("\nFirst few rows:")
print(df.head())

# Select features
features = ['PPG', 'RPG', 'APG', 'Height_in_inches', 'Age_at_Draft', 'FG_Percent', 'ThreeP_Percent']
X = df[features]
y = df['Success']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Train model
print("\nTraining the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved as model.pkl")

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Results ===")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.3f}")

# ROC curve and save image
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png', bbox_inches='tight', dpi=150)
plt.show()

# Confusion Matrix (using matplotlib instead of seaborn)
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(['Bust', 'Hit']))
plt.xticks(tick_marks, ['Bust', 'Hit'])
plt.yticks(tick_marks, ['Bust', 'Hit'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()

# Feature Importance
plt.figure(figsize=(8,5))
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance')
plt.show()

# Bonus: Predict on famous players
print("\n=== Predictions on Top Players ===")
examples = df[['Player_Name'] + features + ['Success']].head(10)
examples_scaled = scaler.transform(examples[features])
examples['Predicted_Prob'] = model.predict_proba(examples_scaled)[:, 1]
examples['Prediction'] = model.predict(examples_scaled)

for _, row in examples.iterrows():
    status = "HIT ✅" if row['Prediction'] == 1 else "BUST ❌"
    print(f"{row['Player_Name']} → {row['Predicted_Prob']:.1%} chance → {status}")

# Predict for all players and save results
print("\n=== Predictions for All Players ===")
all_players = df.copy()
all_scaled = scaler.transform(all_players[features])
all_players['Predicted_Prob'] = model.predict_proba(all_scaled)[:, 1]
all_players['Prediction'] = model.predict(all_scaled)
all_players['Prediction_Label'] = all_players['Prediction'].map({1: 'HIT', 0: 'BUST'})

# Save full predictions
all_players.to_csv('predictions.csv', index=False)
print(f"Saved predictions for {len(all_players)} players to predictions.csv")

# Print summary and top examples
print("\nPrediction counts:")
print(all_players['Prediction_Label'].value_counts())

print("\nTop 20 by predicted probability:")
print(all_players[['Player_Name', 'Predicted_Prob', 'Prediction_Label']].sort_values(
    'Predicted_Prob', ascending=False).head(48).to_string(index=False))