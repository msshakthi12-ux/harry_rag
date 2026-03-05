import logging

logging.basicConfig(level=logging.INFO)

from ingestion.pdf_loader import PDFLoader
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import Chunker
from ingestion.embed_store import EmbedStore

loader  = PDFLoader("data/pdf_books")
cleaner = TextCleaner()
chunker = Chunker()
store   = EmbedStore()

docs = []
for page in loader.load_all():
    cleaned = cleaner.clean(page.text)
    docs.extend(chunker.chunk(cleaned, page.metadata))

store.add_documents(docs)
print(f"✅ Stored {len(docs)} chunks into ChromaDB.")