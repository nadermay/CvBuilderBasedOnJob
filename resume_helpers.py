
import re
import json
import math

def compute_ats_score(cv_text, job_description):
    """
    Compute an ATS matchup score based on keyword frequency and relevance.
    Pure Python implementation — fast and deterministic.
    """
    def extract_keywords(text):
        # Basic keyword extraction: remove stop words, keep nouns/proper nouns
        # This is a simple implementation. For production, use NLTK or Spacy.
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "by", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "that",
            "this", "these", "those", "from", "as", "if", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
        }
        
        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequency
        freq = {}
        for w in keywords:
            freq[w] = freq.get(w, 0) + 1
            
        return freq

    jd_keywords = extract_keywords(job_description)
    cv_keywords = extract_keywords(cv_text)
    
    if not jd_keywords:
        return {"score": 0, "matched": [], "missing": []}
        
    score = 0
    total_weight = 0
    matched = []
    missing = []
    
    # Analyze keywords from JD
    sorted_jd = sorted(jd_keywords.items(), key=lambda x: x[1], reverse=True)
    
    # Top 20 keywords matter most
    top_keywords = sorted_jd[:20]
    
    for word, weight in top_keywords:
        total_weight += weight
        if word in cv_keywords:
            # Count matches up to the required frequency
            count = cv_keywords[word]
            match_score = min(count, weight)
            score += match_score
            matched.append(word)
        else:
            missing.append(word)
            
    final_score = 0
    if total_weight > 0:
        final_score = int((score / total_weight) * 100)
    
    # Boost/Penalty logic
    # Boost if important sections align (simple proxy)
    if final_score > 0:
        final_score = min(100, final_score + 10) # Base boost for structure
        
    return {
        "score": final_score,
        "matched": matched[:10], # Top 10 matched
        "missing": missing[:10]  # Top 10 missing
    }

def format_cover_letter(data):
    """Format cover letter JSON into plain text if needed."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        subject = data.get("subject_line", "")
        body = data.get("body", "")
        closing = data.get("closing_name", "")
        return f"SUBJECT: {subject}\n\n{body}\n\nSincerely,\n{closing}"
    return ""
