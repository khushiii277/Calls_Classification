import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta, timezone
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os

# === CONFIG ===
API_URL = "https://ops.samarthonline.in/api/v1/calls"
HEADERS = {
    "service-token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6InNhbWFydGggc2VydmljZS10b2tlbiIsImlhdCI6MTczMTA1OTgxN30.GCb-eCRsOaQ06FhAnR_42HXzYsyg-AAJLqxUGzVto44"
}

MODEL_PATH = "pipeline/corr_model1.pkl"
ENCODER_PATH = "pipeline/corr_label_encoder1.pkl"
EMBEDDER_PATH = "pipeline/corr_sentence_embedder1.pkl"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1kffteM5sLxCVh3TmlKyH26GTz-cU-944"

# Email Settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ["EMAIL_RECEIVER"].split(",")

OUTPUT_COLUMNS = [
    "id", "accountTitle", "callSubmittedDate", "callByUserName",
    "report", "Useful/Not Useful", "Confidence"
]

def download_embedder_from_drive(drive_url, dest_path):
    print("⬇️ Downloading sentence embedder from Google Drive...")
    response = requests.get(drive_url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Embedder downloaded to {dest_path}")
    else:
        raise Exception(f"❌ Failed to download embedder: {response.status_code}")

# Download embedder if not present
if not os.path.exists(EMBEDDER_PATH):
    download_embedder_from_drive(DRIVE_URL, EMBEDDER_PATH)

def get_yesterday_range():
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    start_dt = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)
    return (
        start_dt.strftime("%a, %d %b %Y 00:00:00 GMT"),
        end_dt.strftime("%a, %d %b %Y 00:00:00 GMT"),
        start_dt,
        end_dt
    )

def fetch_call_data(start, end):
    params = {"startDate": start, "endDate": end, "limit": 3000, "status": "Completed"}
    resp = requests.get(API_URL, headers=HEADERS, params=params)
    if resp.status_code != 200:
        raise Exception(f"API Error {resp.status_code}")
    data = resp.json().get("result", {}).get("data", [])
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
            results.append((name, round(sc, 2)))
    df['Useful/Not Useful'], df['Confidence'] = zip(*results)
    return df

def send_email(attachment_path, start_dt, end_dt):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg["To"] = ", ".join(EMAIL_RECEIVER)
    msg['Subject'] = f"Daily call list: {start_dt.date()}"

    body = f"Daily call list {start_dt.date()}."
    msg.attach(MIMEText(body, "plain"))

    with open(attachment_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
    msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

    print(f"📧 Email sent to: {', '.join(EMAIL_RECEIVER)}")

# === MAIN ===
if __name__ == "__main__":
    sd, ed, dt_start, dt_end = get_yesterday_range()
    df = fetch_call_data(sd, ed)

    df["callSubmittedDate"] = pd.to_datetime(df["callSubmittedDate"], errors='coerce')
    df = df[(df["callSubmittedDate"] >= dt_start) & (df["callSubmittedDate"] < dt_end)]
    print(f"✅ Filtered to {len(df)} calls submitted yesterday")

    if not df.empty:
        df = run_predictions(df)

        missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        out = df[OUTPUT_COLUMNS]
        fname = f"pred_calls_yesterday_{dt_start.strftime('%Y-%m-%d')}.csv"
        out.to_csv(fname, index=False)
        print(f"📁 Saved to: {fname}")

        send_email(fname, dt_start, dt_end)
    else:
        print("⚠️ No valid calls submitted yesterday.")
