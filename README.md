# 🏥 Healthcare AI Monitoring
### Probabilistic Modeling & AI for Healthcare  
**Python 3.9+ · Flask · Vanilla JS · Chart.js**

---

## 📋 Project Overview

A full-stack web application demonstrating 10 probabilistic and AI concepts applied to the Heart Failure Clinical Records Dataset (299 patients, 13 features). Upload a CSV, explore the data, and step through every concept interactively.

| Section | Topic |
|---------|-------|
| 1 | Dataset Upload & Exploration |
| 2 | Marginal Probability — P(Death=1) |
| 3 | Joint Probability — P(Age>60 ∧ Death=1) |
| 4 | Conditional Probability — P(Death \| Condition) |
| 5 | Maximum Likelihood Estimation (MLE) |
| 6 | KL Divergence |
| 7 | Markov Chain Model |
| 8 | Hidden Markov Model (HMM) |
| 9 | Generative AI & Synthetic Data |
| 10 | Deep Learning Architectures (LSTM / GRU / Transformer) |

---

## 🛠 Technologies Used

| Layer | Stack |
|-------|-------|
| Backend | Python 3.9+, Flask 3.0, flask-cors |
| Data | pandas 2.2, numpy 1.26, scipy 1.13 |
| ML / Stats | scikit-learn 1.5, hmmlearn 0.3 |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4.4 (CDN) |
| Fonts | IBM Plex Sans + IBM Plex Mono (Google Fonts CDN) |

---

## 📁 Project Structure

```
healthcare-ai-monitoring/
├── backend/
│   ├── main.py           # Flask app — all API routes
│   ├── analysis.py       # Marginal, Joint, Conditional, MLE, KL, Explore, Synthetic
│   ├── markov_model.py   # Markov Chain transition matrix & simulation
│   ├── hmm_model.py      # Hidden Markov Model (hmmlearn Viterbi decoder)
│   └── requirements.txt  # Pinned Python dependencies
├── frontend/
│   ├── index.html        # Single-page application shell (10 sections)
│   ├── style.css         # Dark clinical theme (CSS variables)
│   └── app.js            # All UI logic, Chart.js rendering, API calls
├── dataset/
│   └── sample_dataset.csv  # Heart Failure Clinical Records (299 rows)
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Create a virtual environment

```bash
cd healthcare-ai-monitoring/backend

# Create venv
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

If any package fails due to a version conflict, try:
```bash
pip install flask flask-cors pandas numpy scipy scikit-learn hmmlearn
```

### Step 3 — Start the backend

```bash
python main.py
```

You should see:
```
🏥  Healthcare AI Monitoring API
   http://localhost:8000
 * Running on http://0.0.0.0:8000
```

### Step 4 — Open the frontend

Open `frontend/index.html` directly in Chrome / Edge / Firefox.

**OR** serve it with Python's built-in server (avoids any CORS issues from `file://`):
```bash
# Open a second terminal from the project root
cd frontend
python -m http.server 3000
```
Then visit **http://localhost:3000**

---

## 📤 Uploading the Dataset

1. Open the app → you land on **Section ① Dataset Upload & Exploration**.
2. Drag & drop `dataset/sample_dataset.csv` onto the upload zone, or click **Choose File**.
3. The page immediately shows:
   - Summary statistics table
   - Dataset preview (first 10 rows)
   - Scatter plots: Age / Ejection Fraction / Serum Creatinine vs Death Event
   - Feature correlation heatmap
4. Click any navigation button to jump to sections ②–⑩ — each loads automatically.

---

## 🔌 API Reference

All routes are served by Flask on `http://localhost:8000`.

| Method | Route | Query Params | Description |
|--------|-------|-------------|-------------|
| POST | `/upload` | — | Upload CSV (multipart/form-data, field name: `file`) |
| GET | `/marginal` | — | P(Death = 1) |
| GET | `/joint` | `age_threshold` (default 60) | Joint probability |
| GET | `/conditional` | `condition` (default `age_gt_60`) | Conditional probability |
| GET | `/mle` | — | MLE Bernoulli estimate + likelihood curve |
| GET | `/kl-divergence` | `assumed_p` (default 0.30) | KL divergence |
| GET | `/markov/matrix` | — | Transition probability matrix |
| POST | `/markov/simulate` | JSON body: `{current_state, steps}` | Trajectory simulation |
| GET | `/hmm/predict` | — | Viterbi hidden state decoding |
| GET | `/generative/synthetic` | `n_samples` (default 20) | Synthetic patient records |

**Condition values** for `/conditional`:
- `age_gt_60` · `diabetes` · `high_blood_pressure` · `anaemia` · `smoking`

---

## 🐞 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: flask_cors` | `pip install flask-cors` |
| `ModuleNotFoundError: hmmlearn` | `pip install hmmlearn` — HMM section uses a heuristic fallback if not installed |
| CORS error in browser | Make sure backend is running on port 8000 before opening the HTML |
| `No dataset loaded` on any section | Upload the CSV in Section ① first |
| Port 8000 in use | Edit the last line of `main.py`: `app.run(port=8001)` and update `API` in `app.js` |
