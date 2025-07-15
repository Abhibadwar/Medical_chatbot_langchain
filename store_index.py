from src.helper import load_pdf_file,text_split,download_huggingface_embeddings
from pinecone import  ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os



PINECONE_API_KEY =os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data="C:/Users/91787/Desktop/Medicalcatbot/Data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()



pc = Pinecone(api_key="pcsk_M1r9i_PtKirwxSqhQRRuUAzeeucCozyYBZF7re87uQ73ND1JHY8ugtxS3Nm1DA9zYj7fD")

index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    )
)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
