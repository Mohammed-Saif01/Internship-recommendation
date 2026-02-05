import pandas as pd
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
import ast
from making_user_interaction_matrix import customEmbeddings
from making_vectorstore import internship_to_string

load_dotenv()

emb_new = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')


def candidate_to_string(candidate):
    s = ""
    for key,values in candidate.items():
        if key in ['education', 'skills', 'location', 'preferred_locations', 'preferred_fields', 'avoid_fields', 'resume_text','experience_level','preferred_mode','preferred_stipend']:
            s += f"{key}: {', '.join(values) if isinstance(values, str) and ',' in values else values}"
            s += ","
    return s


def get_candidate_by_id(candidate_id):
    candidate_df = pd.read_csv('candidates.csv')
    try:
        candidate_row = candidate_df[candidate_df['candidate_id'] == candidate_id]
        if candidate_row.empty:
            return None
        return candidate_row.iloc[0].to_dict()
    except:
        return None

# get_candidate_by_id('candidate_1')

def get_internship_recommendations_for_candidate(candidate_id, k=5, filter_by_stipend=None, filter_by_location=None):
    internships_vector_store = Chroma(
        collection_name='internship_recommendations',
        embedding_function=emb_new, 
        persist_directory='internship_recommendation_db'
    )
    candidate = get_candidate_by_id(candidate_id)
    if not candidate:
        print(f"(get_internship_recommendations_for_candidate) No candidate found with id: {candidate_id}")
        return []

    query = candidate_to_string(candidate)

    if filter_by_stipend is None and filter_by_location is None:
        results = internships_vector_store.similarity_search_with_score(
            query=query,
            k=k
        )

    elif filter_by_location is not None and filter_by_stipend is None:
        results = internships_vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter = {'location': {'$regex': filter_by_location}}
        )

    elif filter_by_stipend is not None and filter_by_location is None:
        results = internships_vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter = {'stipend_num': {'$gte': filter_by_stipend}}
    )
        
    else:
        results = internships_vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter = {
                'stipend_num': {'$gte': filter_by_stipend},
                'location': {'$in': [filter_by_location]}
            }
        )
    return results




def get_internship_by_id(job_id):
    internship_df = pd.read_csv('internships.csv')
    internship_row = internship_df[internship_df['job_id'] == job_id]
    if internship_row.empty:
        return None
    return internship_row.iloc[0].to_dict()

emb = customEmbeddings()


def get_similar_internships(job_id, k=5):

    internships_vector_store = Chroma(
        collection_name='internship_recommendations',
        embedding_function=emb_new, 
        persist_directory='internship_recommendation_db'
    )

    internship_row = get_internship_by_id(job_id)

    results = internships_vector_store.similarity_search_with_score(
        query=internship_to_string(internship_row),
        k=k
    )
    return results

def collaborative_filtering_recommendations(candidate_id, k=5, filter_by_location=None, filter_by_stipend=None):
    vector_store_cf = Chroma(
        collection_name='candidates_interaction_matrix',
        embedding_function=emb, 
        persist_directory='candidates_interaction_matrix_db',
    )

    results = vector_store_cf.similarity_search_with_score(
        query=candidate_id,
        k=30
    )

    internship_df = pd.read_csv('internships.csv')

    print('recommended_candidates with score:', [(result[0].page_content, result[1]) for result in results][1:])
    recommended_candidate_ids = [res[0].page_content for res in results if res[0].page_content != candidate_id]
    print(f"Recommended candidate ids : ",recommended_candidate_ids)

    recommended_internships = []

    for rec_cand_id in recommended_candidate_ids:
        rec_candidate = get_candidate_by_id(rec_cand_id)
        # print(rec_candidate)
        if 'applied_jobs' in rec_candidate:
            applied_internships = ast.literal_eval(rec_candidate['applied_jobs'])
            recommended_internships.extend(applied_internships)
            saved_internships = ast.literal_eval(rec_candidate['saved_jobs'])
            recommended_internships += [jobs for jobs in saved_internships if jobs not in recommended_internships]

    # recommended_internships = list(set(recommended_internships))

    # df = internship_df[internship_df['job_id'].isin(recommended_internships)]
    # df = df[df['stipend_num'] >= filter_by_stipend] if filter_by_stipend is not None else df
    # df = df[df['location'].apply(lambda loc: filter_by_location in loc)] if filter_by_location is not None else df
    # return df.to_dict(orient='records')
    return recommended_internships

if __name__ == '__main__':

    # print("Top 5 internship recommendations for candidate", get_candidate_by_id('candidate_1'))
    # print("-----------------------------------------------------")
    # results = get_internship_recommendations_for_candidate('candidate_33')
    # print(results)
    # for i, res in enumerate(results):
    #     internship = get_internship_by_id(res[0].metadata['job_id'])
    #     print(f"Score: {res[1]}")
    #     print(f"{i+1}. Internship details: {internship}")
    #     print("\n\n")
    # initialize_candidates_interaction_vector_store()
    # print(customEmbeddings().embed_documents(['candidate_1']))
    rcms = collaborative_filtering_recommendations('candidate_32')
    print(rcms)
    # for r in rcms:
        # print(r['job_id'])
