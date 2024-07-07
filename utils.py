import os
import fitz  
import shutil

def fetch_pdf_chunks(split_chunks_by_string=None, split_chunks_by_number_of_characters=250):

    project_root = os.path.dirname(__file__)
    source_documents_folder = os.path.join(project_root, "SOURCE_DOCUMENTS")
    archive_folder = os.path.join(project_root, "ARCHIVE")

    # Create the ARCHIVE folder if it doesn't exist
    if not os.path.exists(archive_folder):
        os.makedirs(archive_folder)
    
    pdf_chunks = []

    # Iterate over all files in the SOURCE_DOCUMENTS folder
    for filename in os.listdir(source_documents_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(source_documents_folder, filename)
            
            with fitz.open(file_path) as pdf_document:
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Initial split by provided criteria
                    if split_chunks_by_string:
                        page_chunks = page_text.split(split_chunks_by_string)
                        print(f"Page {page_num+1} split into {len(page_chunks)} chunks by string.")
                    elif split_chunks_by_number_of_characters:
                        page_chunks = [page_text[i:i+split_chunks_by_number_of_characters] for i in range(0, len(page_text), split_chunks_by_number_of_characters)]
                        print(f"Page {page_num+1} split into {len(page_chunks)} chunks by number of characters.")
                    else:
                        page_chunks = [page_text]
                        print(f"Page {page_num+1} not split into chunks.")
                    
                    pdf_chunks.extend(page_chunks)
            
            # Move the processed file to the ARCHIVE folder
            shutil.move(file_path, os.path.join(archive_folder, filename))
    
    return pdf_chunks

# print(fetch_pdf_chunks(split_chunks_by_string=None, split_chunks_by_number_of_characters=250))