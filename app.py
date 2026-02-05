import streamlit as st
import os
from add_credentials import add_new, validate_user
import time
from extract_resume_info import extract_links_from_pdf, extract_text_from_pdf,resumeDataToProfile, customState, extract_resume_info, checkIfAllFilled, askDetails, routing
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from add_new import ResumeExtractionData
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI, OpenAI
from add_new import add_new_user
import pandas as pd
from find_internship import get_internship_recommendations_for_candidate, get_internship_by_id, collaborative_filtering_recommendations

st.header('Recommendation Engine Demo')

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if 'password' not in st.session_state:
    st.session_state.password = None

if 'page' not in st.session_state:
    st.session_state.page = 'login'

if 'user_input' not in st.session_state:
    st.session_state.user_input = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

if 'candidate_id' not in st.session_state:
    st.session_state.candidate_id = None

if 'chat_disabled' not in st.session_state:
    st.session_state.chat_disabled = True

if 'rerun_for_chatinput' not in st.session_state:
    st.session_state.rerun_for_chatinput = False

# Login Page

if st.session_state.page == 'login':

    st.session_state.user_id = st.text_input('Enter User ID')
    st.session_state.password = st.text_input("Enter password")
    if st.session_state.user_id and st.session_state.password:
        submit = st.button("LOGIN")
        register = st.button("REGISTER")
        if submit:
            response = validate_user(st.session_state.user_id, st.session_state.password)
            if response['status']:
                st.success(response['message'])
                df = pd.read_csv('credentials.csv')
                # if df[(df['user_id'] == st.session_state.user_id) & (df['password']==st.session_state.password)]['candidate_id'].empty:
                #     st.session_state.page = 'upload'
                # else:
                #     st.session_state.candidate_id = df[(df['user_id'] == str(st.session_state.user_id)) & (str(df['password'])==str(st.session_state.password))]['candidate_id']
                #     st.session_state.page = 'recommendation'
                user_mask = (
                    (df['user_id'].astype(str) == str(st.session_state.user_id)) &
                    (df['password'].astype(str) == str(st.session_state.password))
                )

                candidate_ids = df.loc[user_mask, 'candidate_id']

                if candidate_ids.empty or candidate_ids.isna().all() or (candidate_ids.astype(str).str.strip() == '').all():
                    st.session_state.page = 'upload'
                else:
                    st.session_state.candidate_id = candidate_ids.iloc[0]
                    st.session_state.page = 'recommendation'
                time.sleep(3)
                st.rerun()
            else:
                st.error(response['message'])

        if register:
            response = add_new(str(st.session_state.user_id), str(st.session_state.password))
            if response['status'] :
                st.success(response['message'])
            else:
                st.error(response['message'])
            st.session_state.page = 'login'


llm = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
# llm = ChatGroq(model="openai/gpt-oss-120b")
# llm = ChatOpenAI(model='openai/gpt-oss-120b:free')
# llm = ChatOpenAI(model="x-ai/grok-4.1-fast:free")
# llm = OpenAI(model='openai/gpt-4.1-nano')
# llm = ChatOpenAI(model='qwen/qwen3-coder:free')

# llm = ChatOpenAI(model='deepseek/deepseek-r1-distill-llama-70b:free')
# llm = OpenAI(model='deepseek/deepseek-r1-distill-llama-70b:free')

def inputNode(state : customState):
    llm_question = state['messages'][-1].content
    if not st.session_state.rerun_for_chatinput:
        st.session_state.rerun_for_chatinput = True
        st.rerun()
    if st.session_state.user_input is None:
        st.stop()
    else:
        with st.chat_message('human'):
            st.markdown(st.session_state.user_input)
        extraction_system_prompt = """You are a Data Integrity Specialist. 
Your task is to update the 'resume_info' JSON based strictly on the User's answer to the Interviewer's question.

### INPUT CONTEXT
1. **Current JSON**: The existing data (DO NOT delete any existing fields like 'name' or 'education').
2. **Question Asked**: The specific attribute the interviewer asked about.
3. **User Answer**: The raw text input from the user.

### FIELD MAPPING RULES (Apply these logic transformations)

1. **preferred_stipend** (Target Type: Float)
   - Convert "5k", "5000", "5 thousand" -> 5000.0
   - Convert "10LPA" -> Calculate monthly equivalent (approx 83333.0)
   - If user says "Any", "Negotiable", or "Market standards" -> Set to 0.0

2. **preferred_mode** (Target Type: Enum ['onsite', 'remote', 'hybrid'])
   - Map "Work from home", "wfh", "remote" -> "remote"
   - Map "Office", "On-site", "WFO" -> "onsite"
   - Map "Both", "Flexible" -> "hybrid"

3. **preferred_locations** (Target Type: List[str])
   - Extract cities/regions into a clean list.
   - Example: "I prefer Blr or Pune" -> ["Bangalore", "Pune"]
   - If user says "Anywhere" or "Open to relocation" -> ["Any"]

4. **preferred_fields** / **avoid_fields** (Target Type: List[str])
   - Extract standard industry terms (e.g., "Web Dev", "ML", "Testing").

### CRITICAL CONSTRAINTS
- **Partial Updates**: ONLY update the field relevant to the user's answer. Keep all other fields exactly as they are.
- **Negative Answers**: If the user says "I don't know", "Skip", or "No preference", leave the field as None or null.
- **No Hallucinations**: Do not infer gender, age, or unstated skills.
"""
        messages = [
            {"role":"system", "content":extraction_system_prompt},
            {"role":"user", "content":"Here are all attributes : {resume_info} . \n Missing fields : {missing_fields}"},
            {"role":"assistant","content":llm_question},
            {"role":"user","content":st.session_state.user_input}
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        class output(TypedDict):
            resume_info : ResumeExtractionData
            all_attributes_filled : bool 
            missing_fields : list[str]

        structured_llm = llm.with_structured_output(output)

        response = structured_llm.invoke(prompt.format_messages(resume_info=state['resume_info'], missing_fields=state['missing_fields']))

        userinput = st.session_state.user_input
        st.session_state.user_input = None
        st.session_state.messages.append(HumanMessage(content=userinput))
        return {'resume_info': response['resume_info'], 'all_attributes_filled': response['all_attributes_filled'], 'missing_fields':response['missing_fields'], 'messages':[HumanMessage(content=userinput)]}


graph = StateGraph(customState)

graph.add_node('eri',extract_resume_info)
graph.add_node('chk', checkIfAllFilled)
graph.add_node('ask',askDetails)
graph.add_node('in',inputNode)

graph.add_edge(START, 'eri')
graph.add_edge('eri', 'chk')
graph.add_conditional_edges('chk', routing, {'END':END, 'ask':'ask'})
graph.add_edge('ask','in')
graph.add_edge('in','chk')

checkpointer = InMemorySaver()

if 'myGraph' not in st.session_state:
    st.session_state.myGraph = graph.compile(checkpointer=checkpointer)

config1 = {"configurable":{"thread_id":"1"}}
# Upload Page

if st.session_state.page == 'upload':
    file = st.file_uploader('Upload your resume here', type=['.pdf'])
    if file:
        st.success('File Uploaded Successfully')
        if st.session_state.file_uploaded is False:
            pdf = file.getbuffer()
            with open('temp.pdf', 'wb') as f:
                f.write(pdf)

            resume_text = extract_text_from_pdf('temp.pdf')
            links = extract_links_from_pdf('temp.pdf')

        if st.session_state.messages is not None:
            for message in st.session_state.messages:
                if isinstance(message, AIMessage):
                    with st.chat_message('assistant'):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message('human'):
                        st.markdown(message.content)

        st.session_state.user_input = st.chat_input("answer here : ", disabled=st.session_state.chat_disabled)

        if st.session_state.messages == []:

            initial_state = {
                "raw_resume_text": resume_text,
                "extracted_links": links,
            }

            state = st.session_state.myGraph.invoke(initial_state, config=config1)
        
        else:
            state = st.session_state.myGraph.invoke(None,config=config1)

        profile = resumeDataToProfile(state['resume_info'])
        profile = add_new_user(profile)
        print(f"printing profile returned from add_new_user(profile) : ",profile)
        if profile is not None:
            # df = pd.read_csv('credentials.csv')
            # df[(df['user_id'] == st.session_state.user_id) & (df['password'] == st.session_state.password)]['candidate_id'] = profile['candidate_id']
            # df.to_csv('credentials.csv')
            # 1. Read the CSV
            df = pd.read_csv('credentials.csv')

            # 2. Create a "Mask" (the condition)
            # Note: Ensure data types match (e.g., strings vs integers)
            mask = (df['user_id'] == st.session_state.user_id) & (df['password'] == st.session_state.password)

            # 3. Update using .loc[row_condition, column_name]
            if mask.any(): # Check if the user actually exists
                df.loc[mask, 'candidate_id'] = profile['candidate_id']
                
                # 4. Save to CSV
                # IMPORTANT: Use index=False to avoid adding an "Unnamed: 0" column
                df.to_csv('credentials.csv', index=False)
            else:
                st.error("User not found or password incorrect.")
                
            st.session_state.candidate_id = profile['candidate_id']
            st.success('Candidate Profile Created successfuly')
        time.sleep(3)
        st.session_state.page = 'recommendation'
        st.rerun()

        # print(profile)

if st.session_state.page == 'recommendation':
    st.header('Recommendations')

    recommendations = get_internship_recommendations_for_candidate(st.session_state.candidate_id)
    internship_ids = []
    for internships in recommendations:
        internship_ids.append(internships[0].metadata['job_id'])
    
    for ids in internship_ids:
        with st.container(key=ids, border=True):
            internship = get_internship_by_id(ids)
            cont1 = st.container(key=f"{ids}_title_and_company")
            with cont1:
                col1, col2 = st.columns(2, gap='medium', border=True)
                with col1 :
                    st.markdown(f"Role : {internship['title']}")
                with col2:
                    st.markdown(f"Company : {internship['company_name']}")
            
            cont2 = st.container(key=f"{ids}_description", border=True)
            with cont2:
                st.markdown(internship['description'])

    collaborative_recommendations = collaborative_filtering_recommendations(st.session_state.candidate_id)
    st.markdown("Internships you may like (Based on other users similar to you)")
    for ids in collaborative_recommendations:
        internship = get_internship_by_id(ids)
        st.space(size="medium")
        with st.container(key=ids, border=True):
            cont1 = st.container(key=f"{ids}_title_and_company_col")
            with cont1:
                col1, col2 = st.columns(2, gap='medium', border=True)
                with col1 :
                    st.markdown(f"Role : {internship['title']}")
                with col2:
                    st.markdown(f"Company : {internship['company_name']}")
            
            cont2 = st.container(key=f"{ids}_description_col", border=True)
            with cont2:
                st.markdown(internship['description'])

# if st.session_state.page == 'recommendation':
    
#     st.markdown("<h2 style='text-align:center;color:#4A90E2;'>✨ Recommended Internships For You ✨</h2>", 
#                 unsafe_allow_html=True)
#     st.write("")

#     recommendations = get_internship_recommendations_for_candidate(st.session_state.candidate_id)
#     internship_ids = [intern[0].metadata['job_id'] for intern in recommendations]

#     for ids in internship_ids:
#         internship = get_internship_by_id(ids)

#         st.markdown("""
#         <div style='
#             background-color: #FFFFFF11; 
#             border-radius: 12px; 
#             padding: 20px; 
#             margin-bottom: 20px;
#             backdrop-filter: blur(6px);
#             border: 1px solid rgba(255,255,255,0.17);
#         '>
#         """, unsafe_allow_html=True)

#         col1, col2 = st.columns([3, 1])
#         with col1:
#             st.markdown(f"<h4 style='color:#FFFFFF;'>{internship['title']}</h4>", unsafe_allow_html=True)
#             st.markdown(f"<p style='color:#9EC9FF; font-size:14px;'>🏢 {internship['company_name']}</p>", 
#                         unsafe_allow_html=True)

#         with col2:
#             st.button("Apply 🚀", key=f"apply_{ids}")

#         st.markdown(
#             f"<p style='color:#DDDDDD; font-size:14px; text-align:justify;'>{internship['description']}</p>",
#             unsafe_allow_html=True
#         )

#         st.markdown("</div>", unsafe_allow_html=True)

#     # ---- Collaborative Recommendations Section ----
#     st.write("---")
#     st.markdown("<h3 style='color:#5DA9E9;'>🔎 Internships You May Also Like</h3>", unsafe_allow_html=True)
#     st.caption("Based on similar users to you")

#     collaborative_recommendations = collaborative_filtering_recommendations(st.session_state.candidate_id)
    
#     for ids in collaborative_recommendations[:5]:
#         internship = get_internship_by_id(ids)

#         st.markdown("""
#         <div style='
#             background-color: #1E1E1E; 
#             border-radius: 12px; 
#             padding: 20px; 
#             margin-bottom: 20px;
#             border: 1px solid #333333;
#         '>
#         """, unsafe_allow_html=True)

#         col1, col2 = st.columns([3, 1])
#         with col1:
#             st.markdown(f"<h4 style='color:#FFFFFF;'>{internship['title']}</h4>", unsafe_allow_html=True)
#             st.markdown(f"<p style='color:#9EC9FF; font-size:14px;'>🏢 {internship['company_name']}</p>", 
#                         unsafe_allow_html=True)

#         with col2:
#             st.button("Apply 🚀", key=f"col_apply_{ids}")

#         st.markdown(
#             f"<p style='color:#CFCFCF; font-size:14px; text-align:justify;'>{internship['description']}</p>",
#             unsafe_allow_html=True
#         )

#         st.markdown("</div>", unsafe_allow_html=True)