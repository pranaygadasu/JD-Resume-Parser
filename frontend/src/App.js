import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [jdText, setJdText] = useState("");
  const [atsScore, setAtsScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [atsResult, setAtsResult] = useState(null);


  const handleFileChange = (e) => setFile(e.target.files[0]);
//................................................
  /*const handleJdMatch = async () => {
    if (!file || !jdText) return alert("Upload a resume and paste a JD!");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("jd_text", jdText);

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/match_jd_text/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setAtsScore(res.data.ats_score);
    } catch (err) {
      console.error(err);
      alert("ATS scoring failed!");
    } finally {
      setLoading(false);
    }
  };
*/
const handleAtsScore = async () => {
  if (!file || !jdText) return alert("Upload a resume and enter a JD first!");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("jd_text", jdText);

  setLoading(true);
  try {
    const res = await axios.post("http://127.0.0.1:8000/ats_score/", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    setAtsResult(res.data);
  } catch (err) {
    console.error(err);
    alert("ATS scoring failed!");
  } finally {
    setLoading(false);
  }
};

  return (
    <div style={{ padding: "20px" }}>
      <h2>ATS Score Checker</h2>
      <input type="file" onChange={handleFileChange} />
      <br /><br />
<textarea
  placeholder="Paste your job description here..."
  value={jdText}
  onChange={(e) => setJdText(e.target.value)}
  style={{ width: "100%", height: "120px", marginBottom: "15px" }}
/>
<button onClick={handleAtsScore} disabled={loading}>
  {loading ? "Checking..." : "Check ATS Score"}
</button>

{atsResult && (
  <div style={{ marginTop: "20px" }}>
    <h3>ATS Score: {atsResult.ats_score}%</h3>
    <p><strong>Semantic Similarity:</strong> {atsResult.semantic_score}%</p>
    <p><strong>Keyword Match:</strong> {atsResult.keyword_score}%</p>
    <p><strong>Matched Skills:</strong> {atsResult.matched_skills.join(", ") || "None"}</p>
    <p><strong>Missing Skills:</strong> {atsResult.missing_skills.join(", ") || "None"}</p>
  </div>
)}

    </div>
  );
}

export default App;
/*
      <h1>ATS Resume Scorer</h1>
      <input type="file" onChange={handleFileChange} />
      <br /><br />

      <textarea
        rows="6"
        cols="60"
        placeholder="Paste a job description here..."
        value={jdText}
        onChange={(e) => setJdText(e.target.value)}
      />
      <br />

      <button onClick={handleJdMatch} disabled={loading}>
        {loading ? "Calculating..." : "Check ATS Score"}
      </button>

      {atsScore !== null && (
        <p><strong>ATS Match Score:</strong> {atsScore}%</p>
      )}
      */