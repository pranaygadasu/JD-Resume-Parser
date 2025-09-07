# JD Resume Parser  

This project is an **AI-powered Resume & Job Matching System** built with **FastAPI (backend)** and **React (frontend)**.  
It parses resumes, extracts skills, and compares them against job descriptions or a dataset of job postings.  

The system also calculates an **ATS (Applicant Tracking System) Score** so users can see how well their resume matches a specific job description.

---

## ✨ Features

- 📑 **Resume Parsing**  
  - Supports PDF, DOCX, and TXT formats.  
  - Extracts structured fields (summary, skills, education, projects, certifications).  

- 🧹 **Data Cleaning**  
  - Removes stopwords, punctuation.  
  - Normalizes skills using fuzzy-matching.  

- 🤖 **Job Matching**  
  - Compares resume against a dataset of job postings.  
  - Returns **Top-N job matches** with similarity scores.  

- 📊 **ATS Scoring (Resume vs Job Description)**  
  - Hybrid scoring method:
    - Semantic similarity (using Sentence Transformers).  
    - Keyword overlap (skills coverage).  
    - Fuzzy skill match (handles variations like "Python" vs "Python3").  
  - Returns an ATS Score (%) with a breakdown.  

- 🌐 **Frontend (React)**  
  - Upload your resume file and instantly get results.  
  - Option to paste a job description for ATS score calculation.  

---

## 🏗️ Tech Stack

- **Backend**: FastAPI, Python  
- **Frontend**: React.js  
- **NLP & ML**: spaCy, NLTK, SentenceTransformers, sklearn, RapidFuzz  
- **Others**: PyMuPDF (resume parsing), pandas, docx  

---

## ⚙️ Installation

### 1. Clone Repo
```bash
git clone https://github.com/your-username/resume-job-matcher.git
cd resume-job-matcher

### 2. Backend Setup (FastAPI)
cd backend
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
uvicorn main:app --reload


Runs on: http://127.0.0.1:8000

3. Frontend Setup (React)
cd frontend
npm install
npm start


Runs on: http://localhost:3000

🚀 Usage
Option 1: Job Dataset Matching

Upload a resume file.

The system finds Top-5 matching jobs from the dataset.

Displays job descriptions, extracted skills, and match scores.

Option 2: ATS Score (Resume vs JD)

Upload a resume file.

Paste a job description in the text box.

System returns ATS Score (%) with breakdown:

Semantic similarity

Keyword match %

Fuzzy match %

📌 Example Output
{
  "ats_score": 74.5,
  "semantic_similarity": 70.2,
  "keyword_match": 80.0,
  "fuzzy_match": 73.3,
  "resume_skills": ["python", "nlp", "tensorflow", "sql"],
  "jd_skills": ["python", "deep learning", "sql", "pytorch"]
}

🛠️ Next Steps (Future Work)

✅ Visualize ATS Score breakdown (pie chart / bar chart in React).

✅ Expand skill extraction using ontology or pretrained models.

✅ Support LinkedIn job scraping (future enhancement).

✅ Fine-tune transformer models for more accurate semantic similarity.

✅ Multi-resume bulk processing (HR recruiter use-case).
