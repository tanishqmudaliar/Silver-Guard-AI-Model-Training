# 🛡️ Silver Guard — SMS Scam Detection AI

An open-source MobileBERT-based AI model for detecting SMS scams in the wild, with a focus on Indian telecom (DLT headers, UPI fraud, lottery scams). Trained on public datasets, synthetic data, and real personal messages for near-perfect real-world accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![MobileBERT](https://img.shields.io/badge/Model-MobileBERT-FF6F00?logo=google&logoColor=white)
![ONNX](https://img.shields.io/badge/Export-ONNX-005CED?logo=onnx&logoColor=white)
![Colab](https://img.shields.io/badge/Training-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Export Your SMS Messages](#step-1--export-your-sms-messages)
  - [Step 2 — Export Your Contacts (Optional)](#step-2--export-your-contacts-optional)
  - [Step 3 — Run the Auto-Labeler](#step-3--run-the-auto-labeler)
  - [Step 4 — Manual Review](#step-4--manual-review)
  - [Step 5 — Merge Labels](#step-5--merge-labels)
  - [Step 6 — Retrain the Model on Colab](#step-6--retrain-the-model-on-colab)
- [File Reference](#file-reference)
- [Performance](#performance)
- [Contributing a Better Model](#contributing-a-better-model)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Silver Guard is a lightweight SMS scam classifier built for real-world Indian telecom. The base model is trained on:

- Public UCI & India SMS spam datasets (~8,000 messages)
- Synthetic scam/ham data (~5,000 messages)
- Personal messages labeled with the included tooling (~5,000+ messages)

The model runs inference via ONNX (`silver_guard.onnx`) and uses MobileBERT's tokenizer, making it fast enough to run on-device or in a server with minimal overhead.

> **Why personal messages matter:** Public datasets have a ~40% false-positive rate on real Indian SMS. Adding your own labeled messages drops that below 1%. The scripts in this repo make that process as fast as possible.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    AUTO-LABELING PIPELINE               │
├─────────────────────────────────────────────────────────┤
│  DLT Header?  ──────────────────────────► HAM (auto)    │
│  (XX-XXXXXX-X)                                          │
│                                                         │
│  Known Contact? ────────────────────────► HAM (auto)    │
│  (matches contacts.csv / contacts.vcf)                  │
│                                                         │
│  Unknown Number ────────────────────────► REVIEW        │
│  (+91XXXXXXXXXX)                         (manual)       │
└─────────────────────────────────────────────────────────┘

        label_sms.py → review.json → merge_labels.py
                              ↓
                    training_data.json
                              ↓
                  main.ipynb on Google Colab
                              ↓
              silver_guard.onnx  +  vocab.txt
```

Indian businesses are legally required to use TRAI DLT-registered sender IDs (`XX-XXXXXX-T/P/S/G`). Since scammers cannot register DLT IDs, all DLT-format senders are auto-labeled as ham with no manual review needed.

---

## Model Details

| Property      | Value                                                |
| ------------- | ---------------------------------------------------- |
| Architecture  | MobileBERT (fine-tuned for binary classification)    |
| Export format | ONNX (`silver_guard.onnx`)                           |
| Tokenizer     | WordPiece (`vocab.txt`, lowercase, max length 128)   |
| Input format  | `SENDER_HEADER [SEP] message_text`                   |
| Output        | `threat_score` — float in [0, 1] (0 = ham, 1 = scam) |
| Labels        | `0` = ham (safe), `1` = scam                         |

---

## Project Structure

```
Silver Guard AI Model Training/
├── label_sms.py          # Auto-labels SMS exports using DLT + contact matching
├── merge_labels.py       # Merges auto-labeled + manually reviewed data
├── main.ipynb            # MobileBERT training notebook (run on Google Colab)
├── TRAINING_GUIDE.md     # Detailed step-by-step training guide
└── model/
    ├── silver_guard.onnx   # Trained model (ONNX)
    ├── vocab.txt            # MobileBERT tokenizer vocabulary
    └── model_config.json    # Model metadata and I/O spec
```

---

## Getting Started

### Prerequisites

- **Python 3.8+** installed
- **Google account** (for Colab — free tier works)
- **Kaggle account** with API key ([setup guide](https://www.kaggle.com/docs/api))
- Basic familiarity with the command line

Install Python dependencies:

```powershell
pip install pandas
```

---

### Step 1 — Export Your SMS Messages

Export your SMS messages as a JSON file named `data.json` and place it in the project folder.

#### Option A — Windows Phone Link (Recommended for Android)

1. Open **Phone Link** on Windows and go to **Messages**
2. Use [Phone Link Exporter](https://github.com/nickymarino/phone-link-exporter) to export
3. Save the output as `data.json` in the project folder

#### Option B — SMS Backup & Restore (Android)

1. Install [SMS Backup & Restore](https://play.google.com/store/apps/details?id=com.riteshsahu.SMSBackupRestore) from Google Play
2. Back up messages in **JSON** format
3. Transfer to PC and rename the file to `data.json`

#### Option C — Any Other Export Tool

Any tool that produces JSON in this structure works:

```json
{
  "messages": [
    {
      "address": "+919876543210",
      "text": "Your OTP is 123456",
      "kind": "received"
    },
    {
      "address": "JK-HDFCBK-T",
      "text": "Rs.5000 debited from A/c",
      "kind": "received"
    }
  ]
}
```

**Required fields:**

| Field     | Description                                          |
| --------- | ---------------------------------------------------- |
| `address` | Sender ID (phone number or DLT header)               |
| `text`    | Message content                                      |
| `kind`    | `"received"` or `"sent"` (sent messages are ignored) |

---

### Step 2 — Export Your Contacts (Optional)

Contacts allow the auto-labeler to mark messages from known people as ham automatically, significantly reducing manual review.

#### Option A — Google Contacts CSV (Recommended)

1. Go to [contacts.google.com](https://contacts.google.com)
2. Click **Export** in the left sidebar
3. Select **Google CSV**
4. Save as `contacts.csv` in the project folder

#### Option B — Phone Contacts VCF

1. Open your phone's **Contacts** app
2. Go to **Settings → Export**
3. Export as **VCF (vCard)**
4. Transfer to PC as `contacts.vcf`

> If no contacts file is found, all phone numbers go to `review.json` for manual labeling. This is fine — just more work.

---

### Step 3 — Run the Auto-Labeler

```powershell
cd "path\to\Silver Guard AI Model Training"
python label_sms.py
```

#### Understanding DLT Headers

Indian businesses must use TRAI DLT-registered sender IDs. Since scammers cannot register them, all DLT-format senders are auto-labeled as ham:

| Format        | Example       | Meaning                          |
| ------------- | ------------- | -------------------------------- |
| `XX-XXXXXX-T` | `JK-HDFCBK-T` | **T**ransactional (OTPs, alerts) |
| `XX-XXXXXX-P` | `DL-AMAZON-P` | **P**romotional (ads, offers)    |
| `XX-XXXXXX-S` | `MH-AIRTEL-S` | **S**ervice messages             |
| `XX-XXXXXX-G` | `KA-GOVTIN-G` | **G**overnment messages          |

#### Example Output

```
Loaded 5847 messages
Received messages (excluding sent): 5234

============================================================
AUTO-LABELING RESULTS
============================================================
Auto-labeled as HAM:
  - DLT headers (businesses):  4012
  - Known contacts:            287
  - Total auto-labeled:        4299

Needs manual review:           935
  - Frequent senders (3+ msgs): 89
  - Unknown/rare senders:       846
```

#### Output Files

| File          | Contents                              |
| ------------- | ------------------------------------- |
| `output.json` | Auto-labeled ham messages             |
| `review.json` | Unknown senders that need your labels |

---

### Step 4 — Manual Review

Open `review.json` in VS Code or any text editor. Change `"label": null` to:

- `"label": 0` — **Ham** (safe, legitimate)
- `"label": 1` — **Scam** (fraud, phishing, spam)

```json
{
  "instructions": "Set label to 0 for ham (safe), 1 for scam.",
  "total": 935,
  "messages": [
    {
      "text": "Hi, are you coming to the party tomorrow?",
      "address": "+919876543210",
      "label": null,
      "contact_frequency": 15,
      "hint": "frequent_contact"
    },
    {
      "text": "CONGRATS! You won Rs.50,00,000! Click here: bit.ly/xxx",
      "address": "+917890123456",
      "label": null,
      "contact_frequency": 1,
      "hint": "unknown"
    }
  ]
}
```

#### Quick Tips

| Hint                             | What to Do                                                  |
| -------------------------------- | ----------------------------------------------------------- |
| `frequent_contact` (3+ messages) | Almost always **ham (0)** — it's someone you text regularly |
| `unknown` with personal content  | Probably **ham (0)** — friend/family not in contacts        |
| `unknown` with links/offers      | Probably **scam (1)** — especially crypto/lottery/loans     |
| OTP from unknown number          | Usually **ham (0)** — delivery OTPs, etc.                   |

#### Bulk-Label in VS Code

Open Find & Replace (`Ctrl+H`, enable Regex):

```
Find:    "label": null
Replace: "label": 0
```

Then manually flip the obvious scams back to `1`.

---

### Step 5 — Merge Labels

```powershell
python merge_labels.py
```

#### Example Output

```
============================================================
FINAL TRAINING DATA
============================================================
Total messages: 5154
Ham (safe):     5012
Scam:           142
Skipped (unlabeled): 80

Saved to: training_data.json
```

> Messages with `"label": null` are skipped. Go back and label them if you want them included.

#### Final File: `training_data.json`

```json
{
  "total": 5154,
  "ham_count": 5012,
  "scam_count": 142,
  "messages": [
    { "text": "Your OTP is 123456", "address": "JK-HDFCBK-T", "label": 0 },
    {
      "text": "Click here to claim prize!",
      "address": "+917890123456",
      "label": 1
    }
  ]
}
```

---

### Step 6 — Retrain the Model on Colab

1. Open [main.ipynb](main.ipynb) in [Google Colab](https://colab.research.google.com)
2. Upload `training_data.json` and `kaggle.json` when prompted:

```python
from google.colab import files
files.upload()  # Select training_data.json, then kaggle.json
```

3. Run all cells in order:

| Cell | Action                                               |
| ---- | ---------------------------------------------------- |
| 1    | Install dependencies                                 |
| 2    | Download Kaggle datasets                             |
| 3    | Generate synthetic data                              |
| 4    | Merge all data (including your `training_data.json`) |
| 5    | Fine-tune MobileBERT                                 |
| 6    | Export to ONNX                                       |
| 7    | Test inference                                       |
| 8    | Download `silver_guard.onnx` + `vocab.txt`           |

Cell 4 automatically detects and merges your personal data:

```python
PERSONAL_DATA = "training_data.json"

if os.path.exists(PERSONAL_DATA):
    with open(PERSONAL_DATA, 'r') as f:
        personal = json.load(f)
    personal_msgs = personal.get('messages', [])
    print(f"✓ Loaded {len(personal_msgs)} personal messages")
```

4. Replace `model/silver_guard.onnx` and `model/vocab.txt` with the downloaded files.

---

## File Reference

| File                 | Purpose                    | When Created                                          |
| -------------------- | -------------------------- | ----------------------------------------------------- |
| `data.json`          | Your exported SMS messages | You create (Step 1)                                   |
| `contacts.csv`       | Your contacts (optional)   | You create (Step 2)                                   |
| `contacts.vcf`       | Your contacts (optional)   | You create (Step 2)                                   |
| `output.json`        | Auto-labeled ham messages  | `label_sms.py` creates                                |
| `review.json`        | Messages for manual review | `label_sms.py` creates                                |
| `training_data.json` | Final training data        | `merge_labels.py` creates                             |
| `main.ipynb`         | Training notebook          | Already in repo                                       |
| `kaggle.json`        | Kaggle API credentials     | You create ([guide](https://www.kaggle.com/docs/api)) |

---

## Performance

| Metric             | Base model (no personal data) | With personal data |
| ------------------ | ----------------------------- | ------------------ |
| Test accuracy      | ~95%                          | ~99%               |
| Real-world FP rate | ~40%                          | <1%                |
| Edge case accuracy | 14 / 20                       | 19 / 20            |

For best results, aim for:

```
Total training data: 15,000+ messages
├── Public datasets:   ~8,000   (UCI + India SMS Spam)
├── Synthetic:         ~5,000   (auto-generated)
└── Personal:          ~5,000+  (your labeled messages)
    ├── Ham:           ~4,800   (96%)
    └── Scam:          ~200     (4%)
```

Having **real ham** examples is crucial — the model needs to learn what your normal messages look like.

---

## Contributing a Better Model

Silver Guard improves with more diverse, real-world training data. Contributions are very welcome.

### Ways to Contribute

**🔬 Submit a retrained model**

Train a better version of `silver_guard.onnx` using your own labeled data and open a pull request replacing the model files. Include your accuracy metrics and a brief description of your training data (no raw personal messages needed — just the trained artifact).

**📊 Contribute labeled data**

If you have anonymized, labeled SMS data you are comfortable sharing, add it as a JSON file under `data/` and open a pull request. Use the same format as `training_data.json`.

**🛠️ Improve the tooling**

Found a bug in `label_sms.py` or `merge_labels.py`? Have an idea for better scam heuristics? Fix it and send a PR.

**📓 Improve the training notebook**

Suggestions for better hyperparameters, augmentation strategies, or alternative architectures are welcome via issues or PRs against `main.ipynb`.

### How to Open a Pull Request

1. Fork this repository
2. Create a feature branch: `git checkout -b improve/better-scam-detection`
3. Make your changes (model files, scripts, or data)
4. Commit with a descriptive message: `git commit -m "feat: retrain with 3k additional scam examples, FPR drops to 0.3%"`
5. Push and open a pull request against `main`

Please describe in your PR:

- What you changed and why
- Any accuracy / performance numbers you measured
- How the training data was sourced (no need to share raw personal messages)

---

## Troubleshooting

**`ERROR: data.json not found`**

Make sure `data.json` is in the same folder as `label_sms.py` and is valid UTF-8 JSON.

```powershell
dir  # should show data.json
python label_sms.py
```

**`No contacts file found`**

This is a warning, not an error. All phone numbers will go to `review.json` for manual labeling.

**`X messages still have label=null`**

You didn't finish labeling `review.json`. Either go back and label all messages, or run `merge_labels.py` anyway — unlabeled messages are skipped.

**`JSON decode error`**

Validate your `data.json` at [jsonlint.com](https://jsonlint.com) and check that it is UTF-8 encoded with no trailing commas.

**Model still shows high false positives**

You need more real scam examples. Options:

1. Label more messages as scam in `review.json`
2. Find scam SMS datasets online and add them to `training_data.json`
3. Contribute scam datasets via a pull request

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

Made with ❤️ by Tanishq Mudaliar

**🛡️ Silver Guard — Protecting India from SMS Scams.Better data = better protection. Open a PR and help make it smarter.**
