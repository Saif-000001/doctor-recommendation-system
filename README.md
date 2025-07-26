
# 🩺 Health Prediction 

This project provides a **FastAPI-based backend** for predicting medical information from natural language input. It includes the following main features:

* 🔍 **Symptom Extraction** from free-text queries
* 🧑‍⚕️ **Doctor Specialization Recommendation** based on symptoms
* 💊 **Treatment Prediction**, including disease name, medications, precautions, diet, and workout suggestions

Live : https://doctor-recommendation-system.onrender.com/docs

---

## 📁 Project Structure

```
.
├── app/
│   ├── __init__.py 
│   ├── main.py                  # FastAPI app
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── core/
├── data/
├── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Saif-000001/Doctor-Recommendation-System.git
cd Doctor-Recommendation-System
```

### 2. Install Requirements

We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

API will be available at:
👉 `http://127.0.0.1:8000`

---
