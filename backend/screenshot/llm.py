import os
import sys
from groq import Groq

llmgroqapi = os.environ.get("GROQ_API_KEY", "")
if not llmgroqapi:
    print("[llm.py] WARNING: GROQ_API_KEY not set in environment.", file=sys.stderr)

client = Groq(api_key=llmgroqapi)


def process_news(text: str) -> tuple[str, str]:
    prompt = (
        "You are a text preprocessing assistant for a news search system. "
        "Given raw news text (Arabic or English), do the following:\n"
        "1. Clean and correct the text:\n"
        "   - Fix OCR errors, typos, and garbled characters\n"
        "   - Remove dates, timestamps, publication metadata, and author names\n"
        "   - Remove stray numbers or symbols that are not part of the actual news content\n"
        "   - Keep the core news headline/body intact\n"
        "2. If the text is Arabic, translate the cleaned text to English. "
        "If the text is already in English, set ENGLISH to the same as CLEANED.\n\n"
        "Return your response in exactly this format (no extra text):\n"
        "CLEANED: <cleaned text>\n"
        "ENGLISH: <english translation or same as cleaned if already English>\n\n"
        f"Raw text:\n{text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    output = response.choices[0].message.content.strip()

    result = ""
    english_result = ""
    for line in output.splitlines():
        if line.startswith("CLEANED:"):
            result = line[len("CLEANED:"):].strip()
        elif line.startswith("ENGLISH:"):
            english_result = line[len("ENGLISH:"):].strip()

    return result, english_result


