# 📄 CV Builder Based on Job Description

An AI-powered tool that tailors your CV to any job description. Upload your existing resume (PDF), paste a job posting, and get a professionally formatted, ATS-optimized CV — plus a cover letter and gap analysis.

**Powered by [Ollama](https://ollama.com/)** — runs 100% locally, no data leaves your machine.

---

## ✨ Features

- **AI-Tailored CV** — Rewrites and restructures your resume to match the target job
- **Cover Letter Generator** — Creates a job-specific cover letter
- **Gap Analysis** — Identifies skill gaps between your profile and the job requirements
- **ATS Score** — Instant keyword-match score so you know how well your CV ranks
- **PDF Export** — Download a clean, professionally styled PDF
- **Unified Mode** — Generate CV + Cover Letter + Gap Analysis in a single click

---

## 🛠️ Prerequisites

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Runtime |
| **Ollama** | Local LLM engine |
| **A model** (e.g. `qwen2.5:7b`) | AI model for text generation |

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/nadermay/CvBuilderBasedOnJob.git
cd CvBuilderBasedOnJob
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Playwright browsers (for PDF generation)

```bash
playwright install chromium
```

### 4. Install & Start Ollama

Download Ollama from [ollama.com](https://ollama.com/download) and install it.

Then pull the required model:

```bash
ollama pull qwen2.5:7b
```

Start the Ollama server:

```bash
ollama serve
```

> [!NOTE]
> Keep the Ollama server running in a separate terminal while using the app.

---

## ▶️ Usage

### Start the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

### Steps

1. **Upload your CV** — Drag or select your existing resume (PDF format)
2. **Paste the job description** — Copy the full job posting text
3. **Click Generate** — Wait for the AI to analyze and tailor your CV
4. **Review results** — View your tailored CV, cover letter, gap analysis, and ATS score
5. **Download** — Get your new CV as a PDF

---

## ⚙️ Configuration

You can change the AI model and server settings in `app.py`:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
OLLAMA_MODEL = "qwen2.5:7b"                         # Change to any Ollama model
```

**Want a faster model?** Try a smaller one:

```bash
ollama pull qwen2.5:3b
```

Then update `OLLAMA_MODEL = "qwen2.5:3b"` in `app.py`.

**Want better quality?** Use a larger model (requires more RAM):

```bash
ollama pull qwen2.5:14b
```

---

## 📁 Project Structure

```
CvBuilderBasedOnJob/
├── app.py                 # Main Flask application
├── resume_helpers.py      # ATS scoring utilities
├── requirements.txt       # Python dependencies
├── prompts/               # AI prompt templates
│   ├── tailor_resume.txt  # CV tailoring prompt
│   ├── cover_letter.txt   # Cover letter prompt
│   ├── gap_analysis.txt   # Gap analysis prompt
│   └── unified_jobkit.txt # Combined generation prompt
├── templates/             # HTML templates
│   ├── index.html         # Upload page (frontend)
│   └── cv_template.html   # CV rendering template
├── static/
│   └── style.css          # Stylesheet
├── uploads/               # Temp uploaded files (auto-created)
└── output/                # Generated PDFs (auto-created)
```

---

## 📝 License

This project is open source. Feel free to use, modify, and distribute.
