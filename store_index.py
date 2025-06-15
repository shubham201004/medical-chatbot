from src.helper import load_pdf, text_chunk, hugging_face_embedding
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangChain
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY =os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data= load_pdf("data/")
text_chunks= text_chunk(extracted_data)

embedding = hugging_face_embedding()

from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangChain
import os
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")
index_name ="medical-chatbot"
doc_search = PineconeLangChain.from_texts(
    texts=[x.page_content for x in text_chunks],
    embedding=embedding,
    index_name=index_name,
)

# Now you can perform similarity searches
# results = doc_search.similarity_search("your query", k=3)  # Get top 3 results