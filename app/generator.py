from google import genai
import os
import re
import json
from dotenv import load_dotenv
from app.rag_setup import get_qa_index
from llama_index.llms.gemini import Gemini

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



def generate_testcases_with_rag(context: dict) -> dict:
    """Generate test cases using RAG-enhanced context."""
    
    try:
        # Get RAG knowledge (cached after first call)
        qa_index = get_qa_index()
        llm = Gemini(model_name="gemini-2.5-pro")
        query_engine = qa_index.as_query_engine(
            llm=llm,
            response_mode="compact",  # Faster responses
            streaming=False  # Disable streaming for speed
        )
        
        # Extract content and entities from the context
        text_content = context.get("text_content", "")
        extracted_entities = context.get("extracted_entities")

        if not text_content or text_content.strip() == "":
            print("Warning: No text content found, falling back to standard generation")
            return generate_testcases_multimodal(context)

        # Build a hyper-specific RAG query if we have extracted entities
        if extracted_entities:
            features = ", ".join(extracted_entities.get("main_features", []))
            requirements = ", ".join(extracted_entities.get("key_requirements", []))
            roles = ", ".join(extracted_entities.get("user_roles", []))
            rag_query = (
                f"Provide expert testing strategies, potential pitfalls, and security considerations for an application with these features: [{features}]. "
                f"It must meet these requirements: [{requirements}]. "
                f"Consider the following user roles: [{roles}]."
            )
            # Also, add the structured data to the final prompt for the generator LLM
            structured_data_for_prompt = json.dumps(extracted_entities, indent=2)
        else:
            # Fallback to the old, generic query if LangExtract failed or was skipped
            print("No extracted entities found. Using generic RAG query.")
            rag_query = f"""Based on this PRD content, provide specific testing patterns, test case examples, and quality standards for:
1. Functional testing approaches
2. Security testing considerations  
3. Performance testing scenarios
4. User interface testing patterns
5. Integration testing strategies

PRD Content: {text_content[:500]}..."""
            structured_data_for_prompt = "(Not available)"

        print(f"Executing RAG query: {rag_query[:150]}...")
        
        # Quick RAG query
        qa_knowledge = query_engine.query(rag_query)
        print(f"RAG response received: {str(qa_knowledge)[:200]}...")
        
        # Build full context for the final generator full_context = text_content
        
        # Enhanced prompt with RAG context using your better structure
        enhanced_prompt = f"""You are a world-class QA automation engineer. Your task is to generate a comprehensive suite of 10 test cases based on the provided Product Requirements Document (PRD). You are meticulous, creative, and an expert at finding critical bugs.

**INSTRUCTIONS:**
1.  **Analyze the PRD:** Carefully read the PRD content and the structured summary provided below.
2.  **Consult QA Knowledge:** Review the retrieved QA knowledge. This is expert guidance to inspire your test cases. **Do not copy it literally.** Adapt these patterns to the specific features in the PRD.
3.  **Generate 10 Test Cases:** Create exactly 10 test cases that are specific, actionable, and independently executable.
4.  **Prioritize High-Impact Scenarios:** Focus on real user workflows, security vulnerabilities, performance bottlenecks, and data integrity. At least 3 of your test cases must specifically target edge cases, boundary conditions, or potential failure modes.
5.  **Adhere to JSON Schema:** The output must be a single, valid JSON object that strictly follows the provided test case structure. Do not include any extra text or explanations outside of the JSON.

**PRD STRUCTURED SUMMARY (from LangExtract):**
{structured_data_for_prompt}

**FULL PRD CONTENT:**
{full_context[:4000]}

**RETRIEVED QA KNOWLEDGE (for inspiration):**
{qa_knowledge}

**TEST CASE JSON STRUCTURE:**
{{
  "testcases": [
    {{
      "id": "TC001",
      "title": "A clear, specific title describing the test scenario and its objective.",
      "preconditions": "The essential state the system must be in before the test begins. Be concise.",
      "steps": ["Step 1: A clear, actionable step.", "Step 2: Another clear, actionable step."],
      "expected_result": "A specific, measurable, and unambiguous outcome."
    }}
  ]
}}

Begin generating the test cases now."""
        
        print("Generating test cases with enhanced prompt...")
        
        # Generate with enhanced prompt
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=enhanced_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": SCHEMA,
                "temperature": 0.1,  # Lower temperature for more consistent output
            },
        )
        
        result = json.loads(response.text)
        print(f"Successfully generated {len(result.get('testcases', []))} test cases")
        return result
        
    except Exception as e:
        print(f"RAG generation failed with error: {e}")
        print("Falling back to standard generation...")
        # Fallback to standard generation if RAG fails
        return generate_testcases_multimodal(context)




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
