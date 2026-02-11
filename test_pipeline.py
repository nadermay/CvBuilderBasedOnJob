"""
Step-by-step pipeline test for CV Builder.
Tests each stage independently to pinpoint failures.
"""
import sys
import json

print("=" * 60)
print("CV BUILDER PIPELINE TEST")
print("=" * 60)

# ─── STEP 1: Test PDF extraction ───
print("\n[STEP 1] Testing PDF extraction...")
try:
    import pdfplumber
    # Find any PDF in the current directory
    from pathlib import Path
    pdfs = list(Path(".").glob("*.pdf"))
    if not pdfs:
        print("  ⚠ No PDF files found in current directory. Skipping.")
        cv_text = "NADER MAY\nEmail: test@test.com\nPhone: +971000\nExperience: 5 years customer service"
        print(f"  Using dummy text instead ({len(cv_text)} chars)")
    else:
        pdf_path = str(pdfs[0])
        print(f"  Found: {pdf_path}")
        pdf = pdfplumber.open(pdf_path)
        all_text = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
        pdf.close()
        cv_text = "\n".join(all_text)
        print(f"  ✅ Extracted {len(cv_text)} characters from {len(all_text)} pages")
        print(f"  Preview: {cv_text[:150]}...")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# ─── STEP 2: Test Ollama connection ───
print("\n[STEP 2] Testing Ollama connection...")
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    r.raise_for_status()
    models = r.json().get("models", [])
    model_names = [m["name"] for m in models]
    print(f"  ✅ Ollama is running. Models: {model_names}")
    if not any("llama3.2" in m for m in model_names):
        print("  ⚠ llama3.2 not found! Available:", model_names)
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print("  → Run 'ollama serve' in another terminal")
    sys.exit(1)

# ─── STEP 3: Test LLM call with small prompt ───
print("\n[STEP 3] Testing LLM call (small test)...")
try:
    test_payload = {
        "model": "llama3.2",
        "prompt": 'Return this exact JSON: {"name": "Test", "title": "Engineer"}',
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 100}
    }
    r = requests.post("http://localhost:11434/api/generate", json=test_payload, timeout=60)
    r.raise_for_status()
    llm_text = r.json().get("response", "")
    print(f"  ✅ LLM responded: {llm_text[:200]}")
    parsed = json.loads(llm_text)
    print(f"  ✅ Valid JSON: {parsed}")
except json.JSONDecodeError:
    print(f"  ⚠ LLM responded but not valid JSON: {llm_text[:200]}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# ─── STEP 4: Test full LLM resume generation ───
print("\n[STEP 4] Testing full resume generation with LLM...")
try:
    system_prompt = open("prompts/tailor_resume.txt", "r", encoding="utf-8").read()
    
    job_desc = "We are looking for a Customer Service Representative. Requirements: English fluency, 2+ years experience, good communication skills."
    
    user_prompt = (
        f"Here is the candidate's current resume:\n\n"
        f"---\n{cv_text[:1500]}\n---\n\n"
        f"Here is the target job description:\n\n"
        f"---\n{job_desc}\n---\n\n"
        f"Produce the tailored resume as a JSON object. Only output valid JSON."
    )
    
    payload = {
        "model": "llama3.2",
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "num_predict": 4096, "top_p": 0.9}
    }
    
    print("  Calling Ollama (this may take 30-120s)...")
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    llm_response = r.json().get("response", "")
    print(f"  ✅ LLM responded ({len(llm_response)} chars)")
    
    resume_data = json.loads(llm_response)
    print(f"  ✅ Valid JSON with keys: {list(resume_data.keys())}")
    
    # Save for inspection
    with open("test_llm_output.json", "w", encoding="utf-8") as f:
        json.dump(resume_data, f, indent=2, ensure_ascii=False)
    print("  Saved to test_llm_output.json")
    
except json.JSONDecodeError as e:
    print(f"  ⚠ JSON parse error: {e}")
    print(f"  Raw response: {llm_response[:500]}")
    # Save raw for debugging
    with open("test_llm_output_raw.txt", "w", encoding="utf-8") as f:
        f.write(llm_response)
    print("  Saved raw output to test_llm_output_raw.txt")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    sys.exit(1)

# ─── STEP 5: Test data normalization ───
print("\n[STEP 5] Testing data normalization...")
try:
    # Import normalize from app
    sys.path.insert(0, ".")
    from app import normalize_resume_data
    
    if 'resume_data' in dir():
        normalized = normalize_resume_data(resume_data)
    else:
        # Use minimal test data
        normalized = normalize_resume_data({
            "name": "Test User", "title": "Engineer",
            "skills": [{"category": "Tech", "items": ["Python", "JS"]}]
        })
    
    print(f"  ✅ Normalized. Keys: {list(normalized.keys())}")
    print(f"  Skills type: {type(normalized.get('skills'))}")
    if normalized.get("skills"):
        for s in normalized["skills"]:
            print(f"    - {s.get('category')}: items type={type(s.get('items'))}, items={s.get('items')[:3] if s.get('items') else '[]'}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# ─── STEP 6: Test template rendering ───
print("\n[STEP 6] Testing Jinja2 template rendering...")
try:
    from flask import Flask
    test_app = Flask(__name__)
    
    test_data = normalized if 'normalized' in dir() else {
        "name": "Test User", "title": "Engineer",
        "contact": {"email": "t@t.com", "phone": "123", "location": "Dubai", "age": "25"},
        "summary": "Experienced professional.",
        "languages": [{"name": "English", "level": "Native", "percent": 100}],
        "certifications": ["First Aid"],
        "skills": [{"category": "Tech", "items": ["Python", "JavaScript"]}],
        "experience": [{"title": "Dev", "company": "Co", "date": "2020-2024", "location": "City", "bullets": ["Did stuff"]}],
        "education": [{"degree": "BSc", "school": "Uni", "year": "2020", "details": "", "in_progress": False}],
        "interests": ["Travel"],
        "additional_info": ["Available immediately"]
    }
    
    with test_app.app_context():
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("cv_template.html")
        html = template.render(cv=test_data)
    
    print(f"  ✅ Template rendered ({len(html)} chars)")
    with open("test_output.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("  Saved to test_output.html")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ─── STEP 7: Test Playwright PDF ───
print("\n[STEP 7] Testing Playwright PDF generation...")
try:
    import asyncio
    from playwright.async_api import async_playwright
    
    async def test_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            abs_path = str(Path("test_output.html").resolve()).replace("\\", "/")
            await page.goto(f"file:///{abs_path}")
            await page.wait_for_timeout(1000)
            await page.pdf(
                path="test_output.pdf",
                format="A4",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            await browser.close()
    
    asyncio.run(test_pdf())
    print(f"  ✅ PDF generated: test_output.pdf ({Path('test_output.pdf').stat().st_size} bytes)")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("PIPELINE TEST COMPLETE")
print("=" * 60)
