import os
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import pandas as pd
import spacy
from spacy.matcher import Matcher
from scipy.spatial.distance import euclidean, cosine

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# Load a more robust pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Folder paths
resume_folder = 'C:/Users/ashish.choudhary/jupyternotebook/ML_NLP_HF/NLP_ML_HF/ResumeScreening/resumes/'
jd_folder = 'C:/Users/ashish.choudhary/jupyternotebook/ML_NLP_HF/NLP_ML_HF/ResumeScreening/jds/'

# Function to read .docx files
def read_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Function to read .pdf files
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

# Function to read text from either .docx or .pdf files
def read_text_file(file_path):
    if file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Function to read all text files in a folder
def read_text_files_from_folder(folder_path):
    files = os.listdir(folder_path)
    texts = {}
    for file_name in files:
        if file_name.endswith(('.docx', '.pdf')):
            file_path = os.path.join(folder_path, file_name)
            texts[file_name] = read_text_file(file_path)
    return texts

# Function to calculate combined similarity
def calculate_combined_similarity(resume_embedding, jd_embedding):
    euclidean_dist = euclidean(resume_embedding, jd_embedding)
    cosine_dist = cosine(resume_embedding, jd_embedding)
    
    # Convert distances to similarity scores
    euclidean_similarity = 1 / (1 + euclidean_dist)
    cosine_similarity = 1 - cosine_dist
    
    # Combine both similarity scores
    combined_similarity = (euclidean_similarity + cosine_similarity) / 2
    
    return combined_similarity

# Enhanced entity extraction
def extract_information(text):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    
    # Patterns for matching company names (simple heuristic)
    company_pattern = [{"ENT_TYPE": "ORG"}]
    matcher.add("COMPANY", [company_pattern])
    
    matches = matcher(doc)
    companies = set()
    experience = set()
    certificates = set()
    skillsets = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        companies.add(span.text)

    for ent in doc.ents:
        if ent.label_ == "DATE":
            experience.add(ent.text)
        elif "certificate" in ent.text.lower():
            certificates.add(ent.text)
        elif ent.label_ in ["GPE", "PERSON"]:
            skillsets.add(ent.text)

    return {
        "Companies": ', '.join(companies),
        "Experience": ', '.join(experience),
        "Certificates": ', '.join(certificates),
        "Skillsets": ', '.join(skillsets)
    }

# Read resumes and job descriptions
resumes = read_text_files_from_folder(resume_folder)
jds = read_text_files_from_folder(jd_folder)

# List to store results
results = []

# Compare each resume with each JD using combined similarity
for resume_name, resume_text in resumes.items():
    resume_embedding = model.encode(resume_text)
    extracted_info = extract_information(resume_text)

    for jd_name, jd_text in jds.items():
        jd_embedding = model.encode(jd_text)
        similarity_score = calculate_combined_similarity(resume_embedding, jd_embedding)
        
        results.append({
            "Resume": resume_name,
            "Job Description": jd_name,
            "Similarity Score": similarity_score,
            "Companies": extracted_info["Companies"],
            "Experience": extracted_info["Experience"],
            "Certificates": extracted_info["Certificates"],
            "Skillsets": extracted_info["Skillsets"]
        })

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
output_path = 'C:/Users/ashish.choudhary/Documents/similarity_results_with_info.xlsx'
df.to_excel(output_path, index=False)

print(f"Results have been saved to {output_path}")
