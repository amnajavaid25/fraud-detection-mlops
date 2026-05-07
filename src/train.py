import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import category_encoders as ce
import joblib

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv("data/train_transaction.csv")

target = "isFraud"

# optional sampling for speed
df = df.sample(100000, random_state=42)

# ------------------------
# MISSING VALUES
# ------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

num_cols = [col for col in num_cols if col != target]

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

df[cat_cols] = df[cat_cols].fillna("missing")

# ------------------------
# HIGH CARDINALITY
# ------------------------
high_card_cols = ['card1', 'card2', 'addr1']

for col in high_card_cols:
    if col in df.columns:
        freq = df[col].value_counts()
        df[col + "_freq"] = df[col].map(freq)

# ------------------------
# TARGET ENCODING
# ------------------------
encoder = ce.TargetEncoder(cols=cat_cols)
df[cat_cols] = encoder.fit_transform(df[cat_cols], df[target])

# ------------------------
# SPLIT
# ------------------------
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------
# SMOTE
# ------------------------
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ------------------------
# COST-SENSITIVE MODEL
# ------------------------
ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=ratio * 2,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train_smote, y_train_smote)

# ------------------------
# EVALUATION
# ------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(report)
print("AUC:", auc)

# ------------------------
# SAVE MODEL
# ------------------------
joblib.dump(model, "fraud_model.pkl")

print("Model saved")
