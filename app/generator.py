from google import genai
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()  # loads the environment variables from the .env file

def clean_response(text: str):
    """Remove code fences and safely parse JSON."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    cleaned = cleaned.strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback: wrap in dict if not valid JSON
        return {"raw_output": cleaned}

def generate_testcases(context: str):
    """Send parsed PRD to Gemini and get structured test cases."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # More focused, structured prompt
    prompt = f"""
You are an expert QA engineer. Generate 10 comprehensive test cases for this PRD.

**REQUIREMENTS:**
- Output: Valid JSON only
- Test cases: 10 total
- Format: Follow exact structure below

**TEST CASE STRUCTURE:**
{{
  "testcases": [
    {{
      "id": "TC001",
      "title": "Clear, specific title",
      "preconditions": "Essential setup only",
      "steps": ["Actionable step 1", "Actionable step 2"],
      "expected_result": "Measurable outcome"
    }}
  ]
}}

**QUALITY STANDARDS:**
- Each test case must be independently executable
- Steps must be specific and unambiguous
- Include edge cases and negative scenarios
- Cover security, performance, and accessibility
- Focus on real user workflows

**PRD CONTENT:**
{context}

Generate test cases now. Output JSON only.
"""

    qa_best_practices = """
**QA BEST PRACTICES TO FOLLOW:**
- Test both positive and negative paths
- Include boundary value testing
- Test error handling and edge cases
- Consider security implications
- Test accessibility features
- Include performance considerations
- Test integration points
- Validate data persistence
"""

    prompt += qa_best_practices

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return clean_response(response.text)
