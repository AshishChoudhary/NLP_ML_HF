import os
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer, util

# Load the pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Read resumes and job descriptions
resumes = read_text_files_from_folder(resume_folder)
jds = read_text_files_from_folder(jd_folder)

# Compare each resume with each JD
for resume_name, resume_text in resumes.items():
   # print(f"\nResults for {resume_name}:")
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    
    for jd_name, jd_text in jds.items():
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding)
        print(f"  {resume_name}>{jd_name}>{similarity.item():.4f}")
