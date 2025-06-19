# Daily Call Prediction and Email Automation

This Python script automates the process of:
1. Fetching call data from a remote API.
2. Filtering data to only include yesterday’s calls.
3. Running predictions using a trained classification model to categorize call reports as "Useful" or "Not Useful".
4. Saving the results to a CSV file.
5. Sending the file via email to specified recipients.

---

## 🔧 Features

- Automatically downloads the sentence embedding model from Google Drive if not already present.
- Uses a scikit-learn model to classify call reports based on text content.
- Sends an email with the prediction CSV file as an attachment.

---

## 📦 Dependencies

Ensure the following Python packages are installed:

```bash
pip install pandas requests joblib scikit-learn sentence-transformers
