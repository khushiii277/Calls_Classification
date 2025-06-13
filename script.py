import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os

# === CONFIG ===
API_URL = "https://ops.samarthonline.in/api/v1/calls"
HEADERS = {
    "service-token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6InNhbWFydGggc2VydmljZS10b2tlbiIsImlhdCI6MTczMTA1OTgxN30.GCb-eCRsOaQ06FhAnR_42HXzYsyg-AAJLqxUGzVto44" # use your valid token
}

def download_embedder_from_drive(drive_url, dest_path):
    print("⬇️ Downloading sentence embedder from Google Drive...")
    response = requests.get(drive_url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Embedder downloaded to {dest_path}")
    else:
        raise Exception(f"❌ Failed to download embedder: {response.status_code}")

MODEL_PATH = "pipeline/corr_model1.pkl"
ENCODER_PATH = "pipeline/corr_label_encoder1.pkl"
EMBEDDER_PATH = "pipeline/corr_sentence_embedder1.pkl"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1kffteM5sLxCVh3TmlKyH26GTz-cU-944"

# Download if not exists
if not os.path.exists(EMBEDDER_PATH):
    download_embedder_from_drive(DRIVE_URL, EMBEDDER_PATH)


# EMAIL SETTINGS
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # or use 587 for STARTTLS
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ["EMAIL_RECEIVER"].split(",")

OUTPUT_COLUMNS = [
    "id", "accountTitle", "scheduleDate", "callByUserName",
    "report", "Useful/Not Useful", "Confidence"
]

# === HELPERS ===
def format_date(dt):
    return dt.strftime("%a, %d %b %Y 18:30:00 GMT")

def get_last_week_range():
    today = datetime.utcnow()
    lm = today - timedelta(days=today.weekday() + 7)
    ls = lm + timedelta(days=6)
    return format_date(lm), format_date(ls), lm, ls

def fetch_call_data(start, end):
    params = {"startDate": start, "endDate": end, "limit":3000, "status":"Completed"}
    resp = requests.get(API_URL, headers=HEADERS, params=params)
    if resp.status_code != 200:
        raise Exception(f"API Error {resp.status_code}")
    data = resp.json().get("result",{}).get("data",[])
    print(f"✅ Retrieved {len(data)} calls")
    return pd.DataFrame(data)

def run_predictions(df):
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    emb = joblib.load(EMBEDDER_PATH)

    df['report'] = df['report'].astype(str).str.strip()
    embeddings = emb.encode(df['report'].tolist(), show_progress_bar=False)
    preds = clf.predict(embeddings)
    probs = clf.predict_proba(embeddings)
    useful_idx = le.transform(['Useful'])[0]
    scores = probs[:, useful_idx]
    pred_names = le.inverse_transform(preds)

    results = []
    for rpt, name, sc in zip(df['report'], pred_names, scores):
        if len(rpt) < 50:
            results.append(('Not Useful', 0.00))
        else:
            results.append((name, round(sc,2)))
    df['Useful/Not Useful'], df['Confidence'] = zip(*results)
    return df

def send_email(attachment_path, start_dt, end_dt):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg["To"] = ", ".join(EMAIL_RECEIVER)  # visible in inbox
    msg['Subject'] = f"Call Predictions: {start_dt.date()} to {end_dt.date()}"

    body = f"Attached📎: Weekly prediction report for {start_dt.date()} to {end_dt.date()}."
    msg.attach(MIMEText(body, "plain"))

    with open(attachment_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
    msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())  # <-- send to all

    print(f"📧 Email sent to: {', '.join(EMAIL_RECEIVER)}")

# === MAIN ===
if __name__=="__main__":
    sd, ed, dt_start, dt_end = get_last_week_range()
    df = fetch_call_data(sd, ed)
    df = run_predictions(df)

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[OUTPUT_COLUMNS]
    fname = f"pred_calls_{dt_start.strftime('%Y-%m-%d')}_to_{dt_end.strftime('%Y-%m-%d')}.xlsx"
    out.to_excel(fname, index=False)
    print(f"✅ Saved {fname}")

    # Send the file via email
    send_email(fname, dt_start, dt_end)
