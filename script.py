import requests
import json
import pandas as pd
import joblib
from datetime import datetime, timedelta

# === CONFIG ===
API_URL = "https://ops.samarthonline.in/api/v1/calls"
HEADERS = {
    "service-token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6InNhbWFydGggc2VydmljZS10b2tlbiIsImlhdCI6MTczMTA1OTgxN30.GCb-eCRsOaQ06FhAnR_42HXzYsyg-AAJLqxUGzVto44"
}

MODEL_PATH = "/workspaces/codespaces-blank/pipeline/corr_model1.pkl"
ENCODER_PATH = "/workspaces/codespaces-blank/pipeline/corr_label_encoder1.pkl"
EMBEDDER_PATH = "/workspaces/codespaces-blank/pipeline/corr_sentence_embedder1.pkl"

OUTPUT_COLUMNS = [
    "id", "accountTitle", "scheduleDate", "callByUserName",
    "report", "Useful/Not Useful", "Confidence"
]

# === HELPERS ===
def format_date(dt):
    return dt.strftime("%a, %d %b %Y 18:30:00 GMT")

def get_last_week_range():
    today = datetime.utcnow()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)
    return format_date(last_monday), format_date(last_sunday)

def fetch_call_data(start_date, end_date):
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "limit": 3000,
        "status": "Completed"
    }
    response = requests.get(API_URL, headers=HEADERS, params=params)
    if response.status_code == 200:
        records = response.json().get('result', {}).get('data', [])
        print(f"✅ Retrieved {len(records)} records.")
        return pd.DataFrame(records)
    else:
        raise Exception(f"❌ API Error {response.status_code}: {response.text}")

def run_predictions(df):
    # Load model components
    clf = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    embedder = joblib.load(EMBEDDER_PATH)

    # Ensure 'report' column is clean
    df['report'] = df['report'].astype(str).str.strip()
    
    # Embed reports
    embeddings = embedder.encode(df['report'].tolist(), show_progress_bar=True)

    # Predict
    predicted_labels = clf.predict(embeddings)
    probabilities = clf.predict_proba(embeddings)
    useful_index = label_encoder.transform(['Useful'])[0]
    confidence_scores = probabilities[:, useful_index]
    predicted_label_names = label_encoder.inverse_transform(predicted_labels)

    # Apply logic for short texts
    final_labels, final_confidences = [], []
    for report, label, score in zip(df['report'], predicted_label_names, confidence_scores):
        if len(report) < 50:
            final_labels.append('Insufficient')
            final_confidences.append(0.00)
        else:
            final_labels.append(label)
            final_confidences.append(round(score, 2))

    # Add predictions to DataFrame with renamed columns
    df['Useful/Not Useful'] = final_labels
    df['Confidence'] = final_confidences
    return df

# === MAIN PIPELINE ===
if __name__ == "__main__":
    start_date, end_date = get_last_week_range()

    try:
        # Step 1: Fetch call data
        df = fetch_call_data(start_date, end_date)

        # Step 2: Run predictions
        df_with_preds = run_predictions(df)

        # Step 3: Keep only selected columns
        missing_cols = [col for col in OUTPUT_COLUMNS if col not in df_with_preds.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing columns in the data: {missing_cols}")

        output_df = df_with_preds[OUTPUT_COLUMNS]

        # Step 4: Save output
        file_name = f"predicted_calls_{start_date[:11]}_to_{end_date[:11]}.xlsx".replace(",", "").replace(" ", "_")
        output_df.to_excel(file_name, index=False)

        print(f"📁 Predictions saved to {file_name}")
    except Exception as e:
        print(str(e))
