import asyncio
import json     
import langextract as lx
from llama_parse import LlamaParse
from langextract import extract
from google.auth import default as google_auth

# This helper function will be needed to get the default project
def get_gcp_project():
    try:
        _, project_id = google_auth()
        return project_id
    except Exception:
        return None

def extraction_to_dict(obj):
    if hasattr(obj, 'extraction_class'):
        return {
            "extraction_class": getattr(obj, 'extraction_class', None),
            "extraction_text": getattr(obj, 'extraction_text', None),
            "attributes": getattr(obj, 'attributes', {}),
        }
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    # Return string instead of raising error for better debugging
    return str(obj)

# Define the schema for the information we want to extract from the PRD. This tells LangExtract what our final JSON object should look like.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "main_features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of the primary features or user stories described in the document."
        },
        "key_requirements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of critical non-functional requirements, such as performance, security, or data integrity constraints."
        },
        "user_roles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of the user roles or personas mentioned (e.g., 'admin', 'guest user')."
        }
    }
}

# The prompt instructs the LLM on how to perform the extraction based on the schema.
EXTRACTOR_PROMPT = """
Extract the main features, key requirements, and user roles from the provided
Product Requirements Document (PRD). Focus on the most critical aspects that
would be essential for a QA engineer to design test cases.
"""

# Providing high-quality, few-shot examples dramatically improves the accuracy
# and reliability of the extraction.
FEW_SHOT_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "**Feature: User Profile Page**\n"
            "Users must be able to create a profile page with a display name and avatar. "
            "The system must ensure that all avatar uploads are processed in under 2 seconds "
            "and are scanned for malware. The page should be accessible to all users, "
            "but only the owning user or an administrator can edit the profile."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="main_features",
                extraction_text="User Profile Page with display name and avatar",
                attributes={"type": "user_interface"}
            ),
            lx.data.Extraction(
                extraction_class="key_requirements", 
                extraction_text="Avatar uploads must process in under 2 seconds",
                attributes={"type": "performance"}
            ),
            lx.data.Extraction(
                extraction_class="key_requirements",
                extraction_text="Avatars must be scanned for malware", 
                attributes={"type": "security"}
            ),
            lx.data.Extraction(
                extraction_class="user_roles",
                extraction_text="user",
                attributes={"permissions": "profile_owner"}
            ),
            lx.data.Extraction(
                extraction_class="user_roles", 
                extraction_text="administrator",
                attributes={"permissions": "edit_any_profile"}
            )
        ]
    ),
    lx.data.ExampleData(
        text=(
            "**Story: Shopping Cart Checkout**\n"
            "As a guest, I want to be able to purchase items in my cart. The checkout flow "
            "must support both credit card and PayPal. All payment transactions must be "
            "logged for auditing. The final confirmation screen must display an order number."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="main_features",
                extraction_text="Shopping Cart Checkout for guests",
                attributes={"type": "e_commerce"}
            ),
            lx.data.Extraction(
                extraction_class="main_features",
                extraction_text="Support for Credit Card and PayPal payments", 
                attributes={"type": "payment"}
            ),
            lx.data.Extraction(
                extraction_class="key_requirements",
                extraction_text="All payment transactions must be logged for auditing",
                attributes={"type": "compliance"}
            ),
            lx.data.Extraction(
                extraction_class="key_requirements",
                extraction_text="Confirmation screen must display an order number", 
                attributes={"type": "user_experience"}
            ),
            lx.data.Extraction(
                extraction_class="user_roles",
                extraction_text="guest",
                attributes={"permissions": "purchase_only"}
            )
        ]
    )
]


async def parse_pdf_with_images_async(pdf_path: str) -> dict:
    """
    Parses a PDF using LlamaParse and then extracts structured data using LangExtract.
    """
    # Step 1: Use LlamaParse to get clean markdown from the PDF
    try:
        parser = LlamaParse(result_type="markdown", verbose=True, language="en")
        print("Starting PDF parsing with LlamaParse...")
        documents = await parser.aload_data(pdf_path)
        print("LlamaParse finished parsing.")
        
        if not documents:
            raise ValueError("LlamaParse returned no content.")
        
        full_text = "".join([doc.get_content() for doc in documents])
        print(f"Text extracted from llamaparse (first 500 chars): {full_text[:500]}...")
        
    except Exception as e:
        print(f"Error during LlamaParse processing: {e}")
        return {"text_content": f"Failed to parse document. Error: {e}", "extracted_entities": None}

    # Step 2: Use LangExtract with schema-based extraction
    try:
        print("Starting entity extraction with LangExtract...")
        
        # For schema-based extraction, you need to modify your approach
        result = lx.extract(
            text_or_documents=full_text,
            prompt_description=EXTRACTOR_PROMPT,
            examples=FEW_SHOT_EXAMPLES,
            model_id="gemini-1.5-flash",  # Use correct model name
            # Note: Don't pass schema parameter, instead structure your examples properly
        )
        
        print(f"LangExtract result type: {type(result)}")
        print(f"LangExtract result: {result}")
        
        # Handle the response based on the actual structure
        if hasattr(result, 'extractions'):
            # Entity-based extraction - returns result object with extractions
            extracted_data = result.extractions
        elif isinstance(result, dict):
            # Schema-based extraction - returns dict directly
            extracted_data = result
        elif isinstance(result, list) and len(result) > 0:
            # List of results
            extracted_data = result[0] if hasattr(result[0], 'extractions') else result
        else:
            extracted_data = result
            
        print(f"Final extracted data: {json.dumps(extracted_data, default=extraction_to_dict, indent=2)}")

        extracted_data = extraction_to_dict(result)
        
        return {
            "text_content": full_text,
            "extracted_entities": extracted_data
        }
        
    except Exception as e:
        print(f"Error during LangExtract processing: {e}")
        import traceback
        traceback.print_exc()
        return {"text_content": full_text, "extracted_entities": None}



def parse_pdf_with_images(pdf_path: str) -> dict:
    """Synchronous wrapper for LlamaParse."""
    try:
        return asyncio.run(parse_pdf_with_images_async(pdf_path))
    except RuntimeError:
        # asyncio.run() cannot be called from a running event loop.
        # In that case, just return a placeholder or handle differently.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(parse_pdf_with_images_async(pdf_path))
