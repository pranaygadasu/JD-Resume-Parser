# main.py
import os
import sys
import tempfile
import math
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

# Import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.utils import parse_resume_to_df, clean_parsed_resume, clean_text, create_embeddings

app = FastAPI()

# CORS (allow React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match_jd_text/")
async def match_jd_text(file: UploadFile = File(...), jd_text: str = Form(...)):
    try:
        # Save uploaded resume
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Parse & clean resume
        resume_df = parse_resume_to_df(tmp_path)
        resume_df = clean_parsed_resume(resume_df)
        resume_text = resume_df["cleaned_text"].iloc[0]

        # Clean JD
        jd_cleaned = clean_text(jd_text)

        # Get ATS score
        from backend.utils import match_resume_with_jd
        score = match_resume_with_jd(resume_text, jd_cleaned)

        return {"ats_score": round(score * 100, 2)}

    except Exception as e:
        print("ERROR in match_jd_text:", str(e))   # log it
        return JSONResponse(content={"error": str(e)}, status_code=500)


from backend.utils import hybrid_ats_score, extract_skills_fuzzy

@app.post("/ats_score/")
async def ats_score(file: UploadFile = File(...), jd_text: str = Form(...)):
    try:
        # Save uploaded resume temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Parse & clean resume
        resume_df = parse_resume_to_df(tmp_path)
        resume_df = clean_parsed_resume(resume_df)
        resume_text = resume_df["cleaned_text"].iloc[0]
        resume_skills = resume_df["skills_list"].iloc[0]

        # Clean JD
        jd_text_clean = jd_text.lower()
        jd_skills = extract_skills_fuzzy(jd_text_clean)

        # Compute hybrid ATS score
        final_score, semantic_score, keyword_score = hybrid_ats_score(
            resume_text, jd_text_clean, resume_skills, jd_skills
        )

        return {
            "ats_score": final_score,
            "semantic_score": round(semantic_score * 100, 2),
            "keyword_score": round(keyword_score * 100, 2),
            "missing_skills": list(set(jd_skills) - set(resume_skills)),
            "matched_skills": list(set(jd_skills) & set(resume_skills))
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
