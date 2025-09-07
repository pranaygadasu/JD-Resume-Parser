import os
import re
import fitz  # PyMuPDF for PDF parsing
import docx
import pandas as pd # for DFs
import string
import nltk 
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
# STEP 1: Extract text by type
# -----------------------------
def parse_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def parse_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

_nlp = None
def get_spacy_model():
    global _nlp
    if _nlp is None:
        print("⚡ Loading spaCy model (en_core_web_sm)... this may take a moment")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp
#converting extracted text to the DF with detailed columns
SECTION_KEYWORDS = {
    "summary": "summary", "profile": "summary", "about me": "summary",
    "skills": "skills", "technical skills": "skills", "key skills": "skills",
    "education": "education", "academic background": "education", "qualifications": "education",
    "experience": "experience", "work experience": "experience", "employment history": "experience", "professional experience": "experience",
    "projects": "projects", "personal projects": "projects",
    "certifications": "certifications", "awards": "certifications", "publications": "certifications"
}

def extract_name(text):
    nlp = get_spacy_model()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def split_sections_by_keywords(text):
    sections = {v: "" for v in set(SECTION_KEYWORDS.values())}
    lines = text.split("\n")
    current_section = None

    for line in lines:
        clean_line = line.strip().lower()
        matched_section = None

        for keyword, canonical in SECTION_KEYWORDS.items():
            if keyword in clean_line:
                matched_section = canonical
                break

        if matched_section:
            current_section = matched_section
        elif current_section:
            sections[current_section] += line.strip() + " "

    return sections

def parse_resume_to_df(file_path):
    raw_text = parse_resume(file_path)
    name = extract_name(raw_text)
    sections = split_sections_by_keywords(raw_text)

    data = {
        "name": name,
        "summary": sections.get("summary", ""),
        "skills": sections.get("skills", ""),
        "education": sections.get("education", ""),
        "experience": sections.get("experience", ""),
        "projects": sections.get("projects", ""),
        "certifications": sections.get("certifications", "")
    }
    return pd.DataFrame([data])

# temporary/example skills
SKILL_KEYWORDS = [
    # General Programming
    "python", "java", "c++", "javascript", "typescript", "html", "css", "react", "angular", "vue", "node.js", "express",
    "spring boot", "django", "flask", "fastapi","c"
    # AI & ML
    "machine learning", "deep learning", "neural networks", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "huggingface", "transformers",
    "gen ai", "generative ai", "llm", "gpt", "bert", "llama", "rag", "retrieval augmented generation",
    # Data Science & Analytics
    "data analysis", "data visualization", "data cleaning", "data preprocessing",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "statistics", "eda", "exploratory data analysis",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "pl/sql",
    # BI
    "power bi", "tableau", "bi tools", "dax", "ssrs", "ssas", "business intelligence",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "git", "github", "gitlab",
    # HR & Ops
    "recruitment", "employee engagement", "payroll", "talent acquisition", "hr policies",
    "operations", "process improvement", "process management",
    # Marketing
    "seo", "sem", "social media", "content marketing", "email marketing", "google analytics",
    # Training
    "training", "mentoring", "coaching", "curriculum design"
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def extract_skills_fuzzy(text, skill_list=SKILL_KEYWORDS, threshold=85):
    if not isinstance(text, str):
        return []
    text_clean = clean_text(text)
    found = []
    for skill in skill_list:
        if any(fuzz.partial_ratio(skill, word) >= threshold for word in text_clean.split()):
            found.append(skill)
    return list(set(found))

def clean_parsed_resume(df):
    for col in ["summary", "skills", "education", "experience", "projects", "certifications"]:
        df[col] = df[col].apply(clean_text)

    df["skills_list"] = df.apply(
        lambda row: extract_skills_fuzzy(
            f"{row['skills']} {row['summary']} {row['experience']}"
        ),
        axis=1
    )

    df["cleaned_text"] = df.apply(
        lambda row: " ".join([row["summary"], row["experience"], row["projects"], row["skills"]]),
        axis=1
    )
    return df

# embeddings
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model all-mpnet-base-v2")
        _embedding_model = SentenceTransformer("all-mpnet-base-v2")
    return _embedding_model

def clean_job_dataset(df):
    df = df.drop_duplicates(subset=["job_description"], keep="first").reset_index(drop=True)
    df["job_description"] = df["job_description"].apply(clean_text)
    # remove stopwords(pythn = python) 
    df["job_description"] = df["job_description"].apply(
        lambda x: " ".join([w for w in x.split() if w not in stop_words])
    )
    df["skills_list"] = df["job_description"].apply(extract_skills_fuzzy)
    df["processed_text"] = df.apply(
        lambda row: row["job_description"] + " " + " ".join(row["skills_list"]), axis=1
    )
    return df

def create_embeddings(text_list):
    model = get_embedding_model()
    return model.encode(text_list, convert_to_tensor=False)

def match_resume_to_jobs(resume_text, job_df, top_n=5):
    df = job_df.copy()
    job_texts = df["processed_text"].tolist() if "processed_text" in df else df["job_description"].tolist()
    resume_embedding = create_embeddings([resume_text])
    job_embeddings = create_embeddings(job_texts)
    similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
    df["match_score"] = similarities
    return df.sort_values(by="match_score", ascending=False).head(top_n)


def hybrid_ats_score(resume_text, jd_text, resume_skills, jd_skills, semantic_weight=0.6, keyword_weight=0.4):
    """
    Hybrid ATS scoring combining semantic similarity (embeddings) + keyword coverage.
    """
    # Semantic similarity
    resume_embedding = create_embeddings([resume_text])
    jd_embedding = create_embeddings([jd_text])
    from sklearn.metrics.pairwise import cosine_similarity
    semantic_score = float(cosine_similarity(resume_embedding, jd_embedding)[0][0])

    # Keyword coverage
    jd_skills_set = set([s.lower() for s in jd_skills])
    resume_skills_set = set([s.lower() for s in resume_skills])
    if jd_skills_set:
        keyword_score = len(jd_skills_set & resume_skills_set) / len(jd_skills_set)
    else:
        keyword_score = 0.0

    # Weighted score
    final_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
    return round(final_score * 100, 2), semantic_score, keyword_score


def load_job_dataset(file_path):
    df = pd.read_csv(file_path)
    if "job_description" not in df.columns:
        raise ValueError("Dataset must contain a 'job_description' column.")
    return clean_job_dataset(df)


def match_resume_with_jd(resume_text, jd_text):
    """Return ATS similarity score between resume and JD."""
    from sklearn.metrics.pairwise import cosine_similarity

    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    resume_embedding = create_embeddings([resume_text])
    jd_embedding = create_embeddings([jd_text])

    score = float(cosine_similarity(resume_embedding, jd_embedding)[0][0])
    return score
