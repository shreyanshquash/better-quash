import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import tempfile

from app.parser import parse_pdf
from app.generator import generate_testcases


load_dotenv() #loads the environment variables from the .env file

app = FastAPI() #instantiates the FastAPI app



@app.post("/generate-testcases") #generate-testcases is the endpoint name
async def generate_testcases(file: UploadFile = File(...)):   #takes the mandatory file from the user
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: #creates a temporary file
        contents = await file.read() #reads the file from the user
        tmp.write(contents) #writes the file to the temporary file

        tmp_path = tmp.name #gets the path of the temporary file

        # Parse and generate testcases
        text = parse_pdf(tmp_path)
        testcases = generate_testcases(text)

        # Clean up the temporary file
        os.remove(tmp_path)

        return JSONResponse(content={"testcases": testcases}) #returns the testcases


   


