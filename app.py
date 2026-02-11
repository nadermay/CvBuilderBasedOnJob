import os
import json
import time
import re
import traceback
import requests
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
from resume_helpers import compute_ats_score

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent / "output"
PROMPTS_FOLDER = Path(__file__).parent / "prompts"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload


# ─────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF, handling multi-column layouts
    by splitting into left/right columns based on page midpoint.
    """
    pdf = pdfplumber.open(pdf_path)
    all_text = []

    for page in pdf.pages:
        w = page.width
        words = page.extract_words(
            x_tolerance=2, y_tolerance=2, keep_blank_chars=False
        )

        if not words:
            # Fallback: try basic extract_text
            text = page.extract_text()
            if text:
                all_text.append(text)
            continue

        # Split into columns
        mid = w * 0.35  # Typical sidebar boundary

        left_words = sorted(
            [x for x in words if x["x0"] < mid],
            key=lambda x: (round(x["top"] / 5) * 5, x["x0"]),
        )
        right_words = sorted(
            [x for x in words if x["x0"] >= mid],
            key=lambda x: (round(x["top"] / 5) * 5, x["x0"]),
        )

        def words_to_text(word_list):
            if not word_list:
                return ""
            lines = []
            current_y = None
            current_line = []
            for w_item in word_list:
                y = round(w_item["top"] / 5) * 5
                if current_y is not None and y != current_y:
                    lines.append(" ".join(current_line))
                    current_line = []
                current_line.append(w_item["text"])
                current_y = y
            if current_line:
                lines.append(" ".join(current_line))
            return "\n".join(lines)

        left_text = words_to_text(left_words)
        right_text = words_to_text(right_words)

        combined = ""
        if left_text.strip():
            combined += "=== SIDEBAR ===\n" + left_text + "\n\n"
        if right_text.strip():
            combined += "=== MAIN CONTENT ===\n" + right_text

        all_text.append(combined)

    pdf.close()
    return "\n\n".join(all_text)


# ─────────────────────────────────────────────
# LLM Integration
# ─────────────────────────────────────────────


def load_prompt(filename):
    """Load a prompt from the prompts folder."""
    prompt_path = PROMPTS_FOLDER / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_system_prompt():
    return load_prompt("tailor_resume.txt")


def call_ollama(system_prompt, user_prompt, max_tokens=4096):
    """
    Call the Ollama API and return the response text.
    max_tokens controls the output length (use higher for unified generation).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,       # Lower = more focused/deterministic
            "num_predict": max_tokens, # Max tokens for output
            "top_p": 0.9,
        },
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.ConnectionError:
        raise Exception(
            "Cannot connect to Ollama. Make sure Ollama is running "
            "(run 'ollama serve' in a terminal)."
        )
    except requests.exceptions.Timeout:
        raise Exception(
            "Ollama request timed out. The model may be too slow for this task. "
            "Try a smaller input or a faster model."
        )
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")


def parse_llm_response(response_text):
    """
    Parse the LLM response as JSON.
    Attempts to repair common JSON issues from LLMs.
    """
    text = response_text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try fixing common issues: trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise Exception(
        "Failed to parse LLM response as JSON. "
        "The model may have produced invalid output. Try again."
    )


def normalize_resume_data(data):
    """
    Normalize the LLM output to ensure all fields match
    the expected structure for the Jinja2 template.
    Handles missing fields, wrong types, and unexpected structures.
    """
    if not isinstance(data, dict):
        data = {}

    # Name & Title
    data.setdefault("name", "Candidate")
    data.setdefault("title", "Professional")
    data.setdefault("summary", "")

    # Contact
    contact = data.get("contact", {})
    if not isinstance(contact, dict):
        contact = {}
    contact.setdefault("email", "")
    contact.setdefault("phone", "")
    contact.setdefault("location", "")
    contact.setdefault("age", "")
    data["contact"] = contact

    # Languages
    langs = data.get("languages", [])
    if not isinstance(langs, list):
        langs = []
    normalized_langs = []
    for lang in langs:
        if isinstance(lang, dict):
            lang.setdefault("name", "Unknown")
            lang.setdefault("level", "Basic")
            lang.setdefault("percent", 50)
            # Ensure percent is an int
            try:
                lang["percent"] = int(lang["percent"])
            except (ValueError, TypeError):
                lang["percent"] = 50
            normalized_langs.append(lang)
        elif isinstance(lang, str):
            normalized_langs.append({"name": lang, "level": "Proficient", "percent": 70})
    data["languages"] = normalized_langs

    # Certifications
    certs = data.get("certifications", [])
    if not isinstance(certs, list):
        certs = [str(certs)] if certs else []
    data["certifications"] = [str(c) for c in certs]

    # Skills — ensure list of {category, items[]}
    skills = data.get("skills", [])
    if not isinstance(skills, list):
        if isinstance(skills, dict):
            # Convert dict to list of skill groups
            skills = [{"category": k, "items": (v if isinstance(v, list) else [str(v)])} for k, v in skills.items()]
        else:
            skills = []
    normalized_skills = []
    for skill in skills:
        if isinstance(skill, dict):
            items = skill.get("items", [])
            if not isinstance(items, list):
                items = [str(items)] if items else []
            items = [str(i) for i in items]
            normalized_skills.append({
                "category": str(skill.get("category", "Skills")),
                "items": items
            })
        elif isinstance(skill, str):
            normalized_skills.append({"category": "Skills", "items": [skill]})
    data["skills"] = normalized_skills

    # Experience
    exp = data.get("experience", [])
    if not isinstance(exp, list):
        exp = []
    normalized_exp = []
    for job in exp:
        if isinstance(job, dict):
            job.setdefault("title", "")
            job.setdefault("company", "")
            job.setdefault("date", "")
            job.setdefault("location", "")
            bullets = job.get("bullets", [])
            if not isinstance(bullets, list):
                bullets = [str(bullets)] if bullets else []
            job["bullets"] = [str(b) for b in bullets]
            normalized_exp.append(job)
    data["experience"] = normalized_exp

    # Education
    edu = data.get("education", [])
    if not isinstance(edu, list):
        edu = []
    normalized_edu = []
    for item in edu:
        if isinstance(item, dict):
            item.setdefault("degree", "")
            item.setdefault("school", "")
            item.setdefault("year", "")
            item.setdefault("details", "")
            item.setdefault("in_progress", False)
            normalized_edu.append(item)
    data["education"] = normalized_edu

    # Interests
    interests = data.get("interests", [])
    if not isinstance(interests, list):
        interests = [str(interests)] if interests else []
    data["interests"] = [str(i) for i in interests]

    # Additional info
    extra = data.get("additional_info", [])
    if not isinstance(extra, list):
        extra = [str(extra)] if extra else []
    data["additional_info"] = [str(i) for i in extra]

    return data


# ─────────────────────────────────────────────
# PDF Generation
# ─────────────────────────────────────────────

def generate_pdf_from_data(resume_data, output_filename):
    """
    Render the resume data into an HTML template,
    then convert to PDF using Playwright.
    """
    # Render HTML from template
    with app.app_context():
        html_content = render_template("cv_template.html", cv=resume_data)

    # Save temp HTML
    temp_html = OUTPUT_FOLDER / f"{output_filename}.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Convert to PDF using Playwright
    import asyncio
    from playwright.async_api import async_playwright

    async def _to_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(f"file:///{str(temp_html).replace(os.sep, '/')}")
            await page.wait_for_timeout(2000)  # Wait for fonts
            pdf_path = str(OUTPUT_FOLDER / f"{output_filename}.pdf")
            await page.pdf(
                path=pdf_path,
                format="A4",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            await browser.close()
            return pdf_path

    # Run async in sync context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pdf_path = pool.submit(asyncio.run, _to_pdf()).result()
        else:
            pdf_path = loop.run_until_complete(_to_pdf())
    except RuntimeError:
        pdf_path = asyncio.run(_to_pdf())

    # Clean up temp HTML
    try:
        os.remove(temp_html)
    except OSError:
        pass

    return pdf_path


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the upload page."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Handle PDF upload + job description, generate tailored resume."""

    # Validate inputs
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_file = request.files["pdf"]
    job_description = request.form.get("job_description", "").strip()

    if pdf_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    try:
        # 1. Save uploaded PDF
        timestamp = int(time.time())
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", pdf_file.filename)
        upload_path = UPLOAD_FOLDER / f"{timestamp}_{safe_name}"
        pdf_file.save(str(upload_path))

        # 2. Extract text
        cv_text = extract_text_from_pdf(str(upload_path))
        if not cv_text.strip():
            return jsonify({"error": "Could not extract text from PDF. The file may be image-based."}), 400

        # 3. Build prompt
        system_prompt = load_system_prompt()
        user_prompt = (
            f"Here is the candidate's current resume:\n\n"
            f"---\n{cv_text}\n---\n\n"
            f"Here is the target job description:\n\n"
            f"---\n{job_description}\n---\n\n"
            f"Analyze the job description, map the candidate's experience to the requirements, "
            f"and produce the tailored resume as a JSON object. "
            f"Remember: only output valid JSON, nothing else."
        )

        # 4. Call LLM
        llm_response = call_ollama(system_prompt, user_prompt)

        # 5. Parse response
        resume_data = parse_llm_response(llm_response)

        # 5b. Normalize data to match template expectations
        resume_data = normalize_resume_data(resume_data)

        # 6. Generate PDF
        output_name = f"tailored_cv_{timestamp}"
        pdf_path = generate_pdf_from_data(resume_data, output_name)

        # 7. Compute ATS Score (pure Python, fast)
        ats_analysis = compute_ats_score(cv_text, job_description)

        # 8. Clean up upload
        try:
            os.remove(str(upload_path))
        except OSError:
            pass

        return jsonify({
            "success": True,
            "download_url": f"/download/{output_name}.pdf",
            "resume_data": resume_data,
            "ats_score": ats_analysis,
            "cv_text": cv_text,  # Send back for next steps
        })

    except Exception as e:
        traceback.print_exc()  # Log full traceback to terminal
        return jsonify({"error": str(e)}), 500


@app.route("/generate_jobkit", methods=["POST"])
def generate_jobkit():
    """Unified endpoint: Generate CV + Cover Letter + Gap Analysis in ONE LLM call."""

    # Validate inputs
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_file = request.files["pdf"]
    job_description = request.form.get("job_description", "").strip()

    if pdf_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    try:
        # 1. Save uploaded PDF
        timestamp = int(time.time())
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", pdf_file.filename)
        upload_path = UPLOAD_FOLDER / f"{timestamp}_{safe_name}"
        pdf_file.save(str(upload_path))

        # 2. Extract text
        cv_text = extract_text_from_pdf(str(upload_path))
        if not cv_text.strip():
            return jsonify({"error": "Could not extract text from PDF. The file may be image-based."}), 400

        # 3. Build UNIFIED prompt (one call for everything)
        system_prompt = load_prompt("unified_jobkit.txt")
        user_prompt = (
            f"Here is the candidate's current resume:\n\n"
            f"---\n{cv_text}\n---\n\n"
            f"Here is the target job description:\n\n"
            f"---\n{job_description}\n---\n\n"
            f"Generate the complete Job Kit: tailored resume, cover letter, and gap analysis. "
            f"Output ONLY the JSON object with keys: resume, cover_letter, gap_analysis."
        )

        # 4. Call LLM with higher token limit for the combined output
        llm_response = call_ollama(system_prompt, user_prompt, max_tokens=8192)

        # 5. Parse the unified response
        full_data = parse_llm_response(llm_response)

        # 6. Extract and normalize each section
        resume_data = full_data.get("resume", full_data)  # Fallback: if flat, treat whole thing as resume
        cover_letter_data = full_data.get("cover_letter", {})
        gap_analysis_data = full_data.get("gap_analysis", {})

        # Normalize resume data for template
        resume_data = normalize_resume_data(resume_data)

        # 7. Generate PDF
        output_name = f"tailored_cv_{timestamp}"
        pdf_path = generate_pdf_from_data(resume_data, output_name)

        # 8. Compute ATS Score (pure Python, instant)
        ats_analysis = compute_ats_score(cv_text, job_description)

        # 9. Clean up upload
        try:
            os.remove(str(upload_path))
        except OSError:
            pass

        return jsonify({
            "success": True,
            "download_url": f"/download/{output_name}.pdf",
            "resume_data": resume_data,
            "ats_score": ats_analysis,
            "cv_text": cv_text,
            "cover_letter": cover_letter_data,
            "gap_analysis": gap_analysis_data,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate_cover_letter", methods=["POST"])
def generate_cover_letter():
    """Generate a cover letter based on resume and JD."""
    data = request.json
    cv_text = data.get("cv_text", "")
    job_description = data.get("job_description", "")
    
    if not cv_text or not job_description:
        return jsonify({"error": "Missing CV text or Job Description"}), 400
        
    try:
        system_prompt = load_prompt("cover_letter.txt")
        user_prompt = (
            f"RESUME:\n{cv_text[:3000]}\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            "Write the cover letter in JSON format."
        )
        
        response = call_ollama(system_prompt, user_prompt)
        parsed = parse_llm_response(response)
        return jsonify(parsed)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate_gap_analysis", methods=["POST"])
def generate_gap_analysis():
    """Generate gap analysis based on resume and JD."""
    data = request.json
    cv_text = data.get("cv_text", "")
    job_description = data.get("job_description", "")
    
    if not cv_text or not job_description:
        return jsonify({"error": "Missing CV text or Job Description"}), 400
        
    try:
        system_prompt = load_prompt("gap_analysis.txt")
        user_prompt = (
            f"RESUME:\n{cv_text[:3000]}\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            "Perform gap analysis in JSON format."
        )
        
        response = call_ollama(system_prompt, user_prompt)
        parsed = parse_llm_response(response)
        return jsonify(parsed)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    """Serve a generated PDF."""
    file_path = OUTPUT_FOLDER / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(
        str(file_path),
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


# ─────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 CV Builder running at http://localhost:5000")
    print("   Make sure Ollama is running (ollama serve)\n")
    app.run(debug=True, use_reloader=False, port=5000)
