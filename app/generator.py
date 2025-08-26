from google import genai
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()  # loads the environment variables from the .env file


# generator.py

SCHEMA = {
    "type": "object",
    "properties": {
        "testcases": {
            "type": "array",
            # If your client/version complains about minItems/maxItems,
            # you can remove these two lines — they are optional.
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "preconditions": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "expected_result": {"type": "string"}
                },
                "required": ["id", "title", "preconditions", "steps", "expected_result"]
            }
        }
    },
    "required": ["testcases"]
}

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
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SCHEMA,
            "temperature": 0.2,
        },
    )
    # In structured mode this should already be valid JSON text
    return json.loads(response.text)
    # return clean_response(response.text)


def generate_testcases_multimodal(context: dict) -> dict:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    full_context = context.get("text_content","")
    images = context.get("image_content") or []
    if images:
        full_context += "\n\nVISUAL CONTENT:\n" + "\n".join(images)

    prompt = f"""You are an expert QA engineer. Generate exactly 10 test cases for this PRD using the text and visuals.
Return JSON only, matching the schema. Do not include extra fields.


**QUALITY STANDARDS:**
- Each test case must be independently executable
- Steps must be specific and unambiguous
- Include edge cases and negative scenarios
- Cover security, performance, and accessibility
- Focus on real user workflows
- Every test case should start with "Open [Application]..." when applicable

PRD (TEXT + VISUALS):
{full_context}"""

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
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SCHEMA,
            "temperature": 0.2,
        },
    )
    return json.loads(response.text)
