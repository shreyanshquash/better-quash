import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import tempfile

from app.parser import parse_pdf_with_images
from app.generator import generate_testcases, generate_testcases_multimodal


load_dotenv() #loads the environment variables from the .env file

app = FastAPI() #instantiates the FastAPI app

@app.post("/generate-testcases") #generate-testcases is the endpoint name
async def upload_and_generate(file: UploadFile = File(...)):   #takes the mandatory file from the user
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: #creates a temporary file
        contents = await file.read() #reads the file from the user
        tmp.write(contents) #writes the file to the temporary file

        tmp_path = tmp.name #gets the path of the temporary file

        try:
            # Parse PDF with images
            parsed_content = parse_pdf_with_images(tmp_path)
            
            # Generate test cases with multimodal support
            testcases = generate_testcases_multimodal(parsed_content)
            
            return JSONResponse(content={
                "testcases": testcases,
                "content_analysis": {
                    "text_extracted": len(parsed_content["text_content"]) > 0,
                    "images_found": parsed_content["total_images"],
                    "ocr_successful": len(parsed_content["image_content"]) > 0
                }
            })
            # # Parse and generate testcases
            # text = parse_pdf(tmp_path) #parses the pdf file from parser.py
            # testcases = generate_testcases(text) #generates the testcases from generator.py

            # return JSONResponse(content={"testcases": testcases}) #returns the testcases
        finally:
            # Clean up the temporary file
            os.remove(tmp_path)



   


