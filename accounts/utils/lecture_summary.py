"""
Smart AI Lecture Summarizer & Audit Retention Engine for ClassAudit AI.
- Generates structured executive lecture summaries, key takeaways, and action items.
- Uses Groq LLaMA 3.3 if GROQ_API_KEY is configured.
- Provides an intelligent educational NLP fallback engine for zero-dependency / offline execution.
- Manages 50-day audit retention and cleanup policies.
"""

import os
import re
import datetime
import logging
from django.utils import timezone

logger = logging.getLogger(__name__)


def generate_lecture_summary(transcript: str, subject: str = "Class", teacher_name: str = "") -> dict:
    """
    Takes a raw class lecture transcript and returns:
    {
        'topics': 'Thermodynamics, Heat Transfer, Entropy',
        'summary': '### 📌 Key Topics Covered...',
        'word_count': 350
    }
    """
    clean_text = transcript.strip() if transcript else ""
    word_count = len(clean_text.split()) if clean_text else 0

    if not clean_text or word_count < 10:
        return {
            'topics': subject,
            'summary': f"### ⚠️ Brief Session Notice\n\nNo extended spoken lecture content was recorded during this session of **{subject}**.\n\n- **Session Status**: Completed\n- **Spoken Word Count**: {word_count} words\n- **Tip**: Ensure your microphone is enabled and speaking is captured during live class monitoring.",
            'word_count': word_count
        }

    # 1. Attempt Groq AI Generation if configured
    groq_api_key = ''
    try:
        from django.conf import settings
        groq_api_key = getattr(settings, 'GROQ_API_KEY', '') or os.environ.get('GROQ_API_KEY', '')
    except Exception:
        groq_api_key = os.environ.get('GROQ_API_KEY', '')

    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            prompt = f"""You are an elite academic curriculum analyst and lecture summarizer for an institutional learning management system.
A teacher just concluded a live classroom session on the subject "{subject}". Below is the speech transcript of what was spoken during the class.

Analyze this transcript thoroughly and generate a comprehensive, structured lecture summary.

Structure your response strictly into these 4 sections:
### 📌 1. Key Topics Covered
(Provide a bulleted list of main topics and sub-topics discussed)

### 📖 2. Detailed Lecture Summary
(Provide a clear 2-3 paragraph breakdown explaining what the teacher taught, definitions given, formulas or mechanisms explained, and examples shared)

### 💡 3. Core Concepts & Definitions
(Highlight critical rules, terms, formulas, or theorems explained in class)

### 📝 4. Homework & Action Items
(List any homework, practice questions, reading assignments, or deadlines mentioned by the teacher. If none explicitly stated, note "No specific assignment was assigned during this lecture.")

TRANSCRIPT:
\"\"\"{clean_text[:6000]}\"\"\"
"""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a professional educational summarizer. Provide clean, well-formatted Markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            summary_text = response.choices[0].message.content.strip()

            # Extract topics for badge tags
            topics_match = re.search(r'###\s*📌\s*1\.\s*Key Topics Covered(.*?)(?=###|\Z)', summary_text, re.DOTALL)
            topics_list = []
            if topics_match:
                lines = [re.sub(r'^[\*\-\d\.\s]+', '', l).strip() for l in topics_match.group(1).splitlines() if l.strip()]
                topics_list = [l for l in lines if len(l) > 2][:5]

            topics_str = ", ".join(topics_list) if topics_list else subject

            return {
                'topics': topics_str,
                'summary': summary_text,
                'word_count': word_count
            }
        except Exception as e:
            logger.warning("Groq AI summarization unavailable or failed (%s). Using fallback engine.", e)

    # 2. Robust NLP Heuristic Fallback Engine
    return _generate_fallback_summary(clean_text, subject, teacher_name, word_count)


def _generate_fallback_summary(text: str, subject: str, teacher_name: str, word_count: int) -> dict:
    """
    Intelligent heuristic educational summarizer that produces clean, structured notes
    without external API dependencies.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]

    concept_sentences = []
    assignment_sentences = []
    general_sentences = []

    concept_keywords = ['is defined as', 'means', 'concept', 'formula', 'law', 'principle', 'rule', 
                        'remember that', 'important', 'we call this', 'definition', 'example', 'property',
                        'always', 'theory', 'technique', 'method', 'process', 'motion', 'force', 'energy']
    assignment_keywords = ['homework', 'assignment', 'solve', 'practice', 'exercise', 'submit', 
                           'read chapter', 'due', 'next class', 'test', 'prepare', 'write down', 'questions']

    for s in sentences:
        s_lower = s.lower()
        if any(k in s_lower for k in assignment_keywords):
            assignment_sentences.append(s)
        elif any(k in s_lower for k in concept_keywords):
            concept_sentences.append(s)
        else:
            general_sentences.append(s)

    # Extract prominent terms for Topics
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)
    stopwords = {'this', 'that', 'with', 'from', 'have', 'were', 'what', 'when', 'where', 'which', 
                 'there', 'their', 'about', 'would', 'could', 'should', 'these', 'those', 'today',
                 'class', 'going', 'start', 'understand', 'explain', 'please', 'listen', 'student',
                 'students', 'teacher', 'right', 'okay', 'first', 'second', 'hello', 'everyone'}
    word_freq = {}
    for w in words:
        wl = w.lower()
        if wl not in stopwords and len(wl) > 3:
            word_freq[wl] = word_freq.get(wl, 0) + 1

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    topics_list = [w[0].capitalize() for w in top_words]
    if not topics_list:
        topics_list = [subject]

    topics_str = ", ".join(topics_list)

    topics_bullets = "\n".join([f"- **{t}**" for t in topics_list])

    if concept_sentences:
        core_concepts = "\n".join([f"- {s}" for s in concept_sentences[:5]])
    else:
        sample = sentences[:3] if sentences else ["Key foundational concepts were discussed during the lecture."]
        core_concepts = "\n".join([f"- {s}" for s in sample])

    if general_sentences:
        selected_overview = " ".join(general_sentences[:4])
    else:
        selected_overview = " ".join(sentences[:4]) if sentences else f"The lecture focused on core principles in {subject}."

    if assignment_sentences:
        assignments_block = "\n".join([f"- 📌 {s}" for s in assignment_sentences[:3]])
    else:
        assignments_block = "- *No explicit homework or deadlines were flagged during this session. Continue revision of today's topics.*"

    summary_md = f"""### 📌 1. Key Topics Covered
{topics_bullets}

### 📖 2. Detailed Lecture Summary
During this session of **{subject}**, the faculty delivered instruction covering **{topics_str}**. 

{selected_overview}

The session maintained active pacing with real-time spoken engagement recorded across **{word_count} words**.

### 💡 3. Core Concepts & Definitions
{core_concepts}

### 📝 4. Homework & Action Items
{assignments_block}
"""

    return {
        'topics': topics_str,
        'summary': summary_md.strip(),
        'word_count': word_count
    }


def purge_old_lecture_records(principal=None, days: int = 50) -> int:
    """
    Safely prunes completed class sessions older than `days` (default 50 days retention).
    Ensures portal database performance remains optimal while strictly adhering to the 
    institutional 50-day audit compliance policy.
    Returns the number of deleted records.
    """
    try:
        from accounts.models import ClassSession
        cutoff = timezone.now() - datetime.timedelta(days=days)
        qs = ClassSession.objects.filter(status='Completed', start_time__lt=cutoff)
        if principal:
            qs = qs.filter(teacher__principal=principal)
        deleted_count, _ = qs.delete()
        logger.info("Purged %d class session records older than %d days.", deleted_count, days)
        return deleted_count
    except Exception as e:
        logger.error("Failed to purge old records: %s", e)
        return 0
