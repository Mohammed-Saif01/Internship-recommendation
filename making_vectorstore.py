from data2 import internships, candidates
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from time import sleep

load_dotenv()

class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
            sleep(0.2) 
        print(f"Embedded {len(texts)} documents")
        return embeddings


def internship_to_string(internship):
    s = ""
    for key,values in internship.items():
        if key in ['title', 'field', 'location', 'skills_required', 'description', 'candidate_expectations']:
            s += f"{key}: {', '.join(values) if isinstance(values, list) else values}"
            s += ","
    return s

if __name__ == "__main__":

    emb = CustomGoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001', google_api_key='AIzaSyAgTTDHqbdm5bymFwHO7jqDKm3hg4xhRT0')


    vector_store = Chroma(
        collection_name='internship_recommendations',
        embedding_function=emb, 
        persist_directory='internship_recommendation_db',
    )
    # # print(internship_to_string(internships[0]))

    uuids = [internship['job_id'] for internship in internships]

    docs = [Document(page_content=internship_to_string(internship), metadata={'job_id': internship['job_id'], 'stipend':internship['stipend_num'], 'location': internship['location']}) for internship in internships]

    vector_store.add_documents(docs, uuids=uuids)

    # print(internship_to_string(internships[0]))