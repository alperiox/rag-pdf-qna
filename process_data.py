import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def process_pdf(file) -> list[Document]:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file.getvalue())
        loader = PyMuPDFLoader(f.name, extract_images=True)
    
    data = loader.load()
    return data
