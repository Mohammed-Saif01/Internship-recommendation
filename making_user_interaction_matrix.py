import pandas as pd
import ast
from langchain_chroma import Chroma
from langchain_core.documents import Document
import numpy as np
import json
import os

if os.path.exists('labels_list.json'):
    with open('labels_list.json','r') as f:
        labels_list = json.load(f)
else:
    labels_list = []

def initialize_candidates_interaction_vector_store():

    candidates_df = pd.read_csv('candidates.csv')

    all_genres = set()
    all_locations = set()
    candidates_df['genre_time_spent'] = candidates_df['genre_time_spent'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
    candidates_df['genre_clicks'] = candidates_df['genre_clicks'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
    candidates_df['location_time_spent'] = candidates_df['location_time_spent'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
    candidates_df['location_clicks'] = candidates_df['location_clicks'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})

    for r in candidates_df.itertuples():
        # print(type(r.genre_time_spent), type(r.genre_clicks))
        # print(r.genre_time_spent, r.genre_clicks)
        # break
        genre_time_spent = list(r.genre_time_spent.keys())
        genre_clicks = list(r.genre_clicks.keys())
        all_genres.update(genre_time_spent)
        all_genres.update(genre_clicks)
        location_time_spent = list(r.location_time_spent.keys())
        location_clicks = list(r.location_clicks.keys())
        all_locations.update(location_time_spent)
        all_locations.update(location_clicks)

    print("All Genres:", all_genres)
    print("All Locations:", all_locations)

    candidates_interaction_matrix = pd.DataFrame(0.0, index=candidates_df['candidate_id'], columns=sorted(list(all_genres)) + sorted(list(all_locations)))

    global labels_list
    labels_list = sorted(list(all_genres)) + sorted(list(all_locations))
    with open('labels_list.json', 'w') as f:
        json.dump(labels_list, f)

    print(candidates_interaction_matrix.head())

    for r in candidates_df.itertuples():

        for key,value in r.genre_time_spent.items():
            candidates_interaction_matrix.at[r.candidate_id, key] += value
        for key,value in r.genre_clicks.items():
            candidates_interaction_matrix.at[r.candidate_id, key] += value
        for key,value in r.location_time_spent.items():
            candidates_interaction_matrix.at[r.candidate_id, key] += value
        for key,value in r.location_clicks.items():
            candidates_interaction_matrix.at[r.candidate_id, key] += value

    def normalize(x):
        # total = x.sum()
        norm = np.linalg.norm(x)
        if norm > 0:
            return x / norm
        else:
            return x

    candidates_interaction_matrix = candidates_interaction_matrix.apply(normalize, axis=1)
    candidates_interaction_matrix.to_csv('candidates_interaction_matrix.csv')

    print("Normalized Successfully")
    print(candidates_interaction_matrix.head())

    # Storing to vector store
    class customEmbeddings:
        def embed_documents(self,texts):
            result = []
            for t in texts:
                vec = candidates_interaction_matrix.loc[t].values.tolist()
                result.append(vec)
            return result
        
    emb = customEmbeddings()
    vector_store = Chroma(
        collection_name='candidates_interaction_matrix',
        embedding_function=emb, 
        persist_directory='candidates_interaction_matrix_db',
    )

    docs = [Document(page_content=candidate_id) for candidate_id in candidates_interaction_matrix.index]

    vector_store.add_documents(docs, uuids=candidates_interaction_matrix.index.tolist())


    print("Successfully created candidates interaction matrix and saved to candidates_interaction_matrix_db")

# Mini class for embedding
class customEmbeddings:
            
    def embed_documents(self, candidate_ids):
        result = []
        df = pd.read_csv('candidates_interaction_matrix.csv', index_col=0)

        for candidate_id in candidate_ids:
            if candidate_id not in df.index:
                raise ValueError(f"Candidate '{candidate_id}' not found in interaction matrix!")

            vec = df.loc[candidate_id].values.tolist()
            result.append(vec)

        return result

    
    def embed_query(self, candidate_id):
        vec = self.embed_documents([candidate_id])[0]
        return vec
    
emb =  customEmbeddings()

def add_new_candidate_to_vectorStore(candidate_id):
    global labels_list
    candidates_df = pd.read_csv('candidates.csv')
    candidate_row = candidates_df[candidates_df['candidate_id'] == candidate_id]
    print(f"storing candidate into vector store : {candidate_id}")
    # print(f"printing candidate row : {candidate_row}")
    if candidate_row.empty:
        print(f"(add_new_candidate_to_vectorStore) No candidate found with id: {candidate_id}")
        return
    else:
        print(f"Found candidate with id: {candidate_id}")


    class customMiniEmbeddings:
            
        def embed_documents(self,candidate_ids):
            print("embed documents called")
            print(f"candidates Ids = {candidate_ids}")
            if not candidate_ids:
                print("embed_documents called with EMPTY LIST")
                return []

            result = []
            candidate_id = candidate_ids[0]
            candidate_row = candidates_df[candidates_df['candidate_id'] == candidate_id]
            candidate_details = candidate_row.iloc[0].to_dict()
            
            genre_time_spent = ast.literal_eval(candidate_details['genre_time_spent']) if pd.notna(candidate_details['genre_time_spent']) else {}
            genre_clicks = ast.literal_eval(candidate_details['genre_clicks']) if pd.notna(candidate_details['genre_clicks']) else {}
            location_time_spent = ast.literal_eval(candidate_details['location_time_spent']) if pd.notna(candidate_details['location_time_spent']) else {}
            location_clicks = ast.literal_eval(candidate_details['location_clicks']) if pd.notna(candidate_details['location_clicks']) else {}
            interaction_vector = []

            new_row = pd.Series(0.0, index=labels_list, name=candidate_id)
            print(f"(add_new_candidate_to_vectorStore) printing new row : ",new_row)
            for key,value in genre_time_spent.items():
                if key in labels_list:
                    new_row.at[key] += value
                else:
                    print(f"Genre '{key}' from candidate '{candidate_id}' not in labels_list.")
                    print("Generating complete vector store again to include new genres/locations.")
                    initialize_candidates_interaction_vector_store()
                    return self.embed_documents([candidate_id])
                
            for key,value in genre_clicks.items():
                if key in labels_list:
                    new_row.at[key] += value
                else:
                    print(f"Genre '{key}' from candidate '{candidate_id}' not in labels_list.")
                    print("Generating complete vector store again to include new genres/locations.")
                    initialize_candidates_interaction_vector_store()
                    return self.embed_documents([candidate_id])
                
            for key,value in location_time_spent.items():
                if key in labels_list:
                    new_row.at[key] += value
                else:
                    print(f"Location '{key}' from candidate '{candidate_id}' not in labels_list.")
                    print("Generating complete vector store again to include new genres/locations.")
                    initialize_candidates_interaction_vector_store()
                    return self.embed_documents([candidate_id])
                
            for key,value in location_clicks.items():
                if key in labels_list:
                    new_row.at[key] += value
                else:
                    print(f"Location '{key}' from candidate '{candidate_id}' not in labels_list.")
                    print("Generating complete vector store again to include new genres/locations.")
                    initialize_candidates_interaction_vector_store()
                    return self.embed_documents([candidate_id])
                
            norm = np.linalg.norm(new_row.values)
            
            if norm > 0:
                new_row = new_row / norm 


            vec = new_row.values.tolist()
            
            df = pd.read_csv('candidates_interaction_matrix.csv',index_col=0)

            # (Optional but recommended) ensure same columns and order
            # new_row = new_row.reindex(df.columns, fill_value=0.0)

            # Add as a new row; new_row.name will become the index label
            df = pd.concat([df, new_row.to_frame().T], axis=0)

            # Don’t forget to save back if you actually want it persisted
            df.to_csv('candidates_interaction_matrix.csv')  # or True if you use index

            result.append(vec)
            print(f"returning embeddings : {result}")
            return result
        
        def embed_query(self, candidate_id):
            vec = self.embed_documents([candidate_id])[0]
            return vec

    # Adding to vector Store
    vector_store = Chroma(
        embedding_function=customMiniEmbeddings(), 
        persist_directory='candidates_interaction_matrix_db',
    )

    # Add to vector store
    doc = Document(page_content=candidate_id)
    vector_store.add_documents([doc], uuids=[candidate_id])

    print(f"Added new candidate {candidate_id} to interaction matrix vector store.")

if __name__ == '__main__':
    # print(emb.embed_documents(['candidate_1']))
    initialize_candidates_interaction_vector_store()