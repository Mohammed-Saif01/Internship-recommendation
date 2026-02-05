import streamlit as st
import os
import time
import pandas as pd

# --- Custom Imports ---
from add_credentials import add_new, validate_user
from extract_resume_info import extract_links_from_pdf, extract_text_from_pdf, resumeDataToProfile, customState, extract_resume_info, checkIfAllFilled, askDetails, routing
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from add_new import ResumeExtractionData, add_new_user
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI, OpenAI

# UPDATE: Added get_similar_internships to the import
from find_internship import get_internship_recommendations_for_candidate, get_internship_by_id, collaborative_filtering_recommendations, get_similar_internships

# ==========================================
# 🎨 UI/UX DESIGN CONFIGURATION
# ==========================================

st.set_page_config(page_title="AI Career Architect", page_icon="🚀", layout="wide")

def load_custom_css():
    st.markdown("""
    <style>
        /* IMPORT GOOGLE FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        /* GLOBAL STYLES */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* BACKGROUND GRADIENT */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            background-attachment: fixed;
        }

        /* CUSTOM SCROLLBAR */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #1e1e2f;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 5px;
        }

        /* HEADERS */
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(0,0,0,0.3);
        }

        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: rgba(30, 30, 50, 0.5);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* INPUT FIELDS */
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.05);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }

        /* BUTTONS (Neon Gradient) */
        div.stButton > button {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        }
        div.stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
            color: white;
        }

        /* CARD STYLING */
        .job-card {
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px;
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            color: white;
            position: relative;
            overflow: hidden;
        }
        
        .job-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, #00d2ff, #928DAB);
        }

        .job-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(255, 255, 255, 0.3);
        }

        .company-badge {
            background: rgba(0, 210, 255, 0.15);
            color: #00d2ff;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 10px;
        }
        
        .role-title {
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 8px;
            background: -webkit-linear-gradient(#fff, #ccc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* DETAIL PAGE SPECIFIC */
        .metric-box {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .metric-label {
            font-size: 0.8em;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 1.1em;
            font-weight: 600;
            color: #fff;
            margin-top: 5px;
        }
        .skill-chip {
            display: inline-block;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 5px 15px;
            margin: 5px;
            font-size: 0.85em;
            color: #eee;
        }
        .section-header {
            font-size: 1.2em;
            color: #00d2ff;
            margin-top: 25px;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ==========================================
# STATE INITIALIZATION
# ==========================================

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
if 'selected_internship_id' not in st.session_state:
    st.session_state.selected_internship_id = None

# ==========================================
# SIDEBAR LOGIC
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("👤 User Profile")
        st.markdown(f"**Logged in as:** `{st.session_state.user_id}`")
        st.markdown("---")
        st.subheader("⚙️ Options")
        st.markdown("Want to analyze a different resume?")
        if st.button("📄 Upload New Resume", use_container_width=True, disabled=True):
            st.session_state.page = 'upload'
            st.session_state.file_uploaded = False
            st.session_state.messages = []
            st.session_state.user_input = None
            st.rerun()
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ==========================================
# LOGIN PAGE
# ==========================================
# ==========================================
# LOGIN PAGE
# ==========================================

if st.session_state.page == 'login':
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🔐 Access Portal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a0a0a0;'>Welcome to the Next-Gen Recommendation Engine</p>", unsafe_allow_html=True)
        
        with st.container(border=True): 
            st.session_state.user_id = st.text_input('User ID', placeholder="Enter your ID")
            st.session_state.password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                submit = st.button("LOGIN", use_container_width=True)
            with b_col2:
                register = st.button("REGISTER", use_container_width=True)

            if submit:
                if st.session_state.user_id and st.session_state.password:
                    response = validate_user(st.session_state.user_id, st.session_state.password)
                    if response['status']:
                        st.success(response['message'])
                        df = pd.read_csv('credentials.csv')
                        user_mask = ((df['user_id'].astype(str) == str(st.session_state.user_id)) & (df['password'].astype(str) == str(st.session_state.password)))
                        candidate_ids = df.loc[user_mask, 'candidate_id']

                        if candidate_ids.empty or candidate_ids.isna().all() or (candidate_ids.astype(str).str.strip() == '').all():
                            st.session_state.page = 'upload'
                        else:
                            st.session_state.candidate_id = candidate_ids.iloc[0]
                            st.session_state.page = 'recommendation'
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(response['message'])
                else:
                    st.warning("Please fill in all fields.")

            # --- FIXED REGISTER LOGIC ---
            if register:
                if st.session_state.user_id and st.session_state.password:
                    response = add_new(str(st.session_state.user_id), str(st.session_state.password))
                    
                    if response['status']:
                        # SUCCESS: Show message, wait, then refresh
                        st.success(f"✅ {response['message']}")
                        time.sleep(2)  # Give user 2 seconds to see the message
                        st.rerun()
                    else:
                        # ERROR: Show message, but DO NOT rerun immediately (so user can see it)
                        st.error(f"❌ {response['message']}")
                else:
                    st.warning("Please fill in fields to register.")
# ==========================================
# LLM SETUP
# ==========================================

llm = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

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
        extraction_system_prompt = """You are a Data Integrity Specialist...""" 
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

# ==========================================
# UPLOAD PAGE
# ==========================================

if st.session_state.page == 'upload':
    st.markdown("""
        <div style="text-align:center; margin-bottom: 20px;">
            <h1 style="font-size: 3rem;">📄 Resume Analysis</h1>
            <p style="color: #ccc;">Upload your PDF to begin the profile matching process</p>
        </div>
    """, unsafe_allow_html=True)

    file = st.file_uploader('Upload your resume here', type=['.pdf'])
    
    if file:
        if not st.session_state.file_uploaded:
             st.markdown('<div style="background-color:rgba(0,255,0,0.1); padding:10px; border-radius:5px; color:#4caf50; text-align:center; margin-bottom:10px;">✅ File Uploaded Successfully</div>', unsafe_allow_html=True)
             st.session_state.file_uploaded = True
        
        pdf = file.getbuffer()
        with open('temp.pdf', 'wb') as f:
            f.write(pdf)

        resume_text = extract_text_from_pdf('temp.pdf')
        links = extract_links_from_pdf('temp.pdf')

        with st.container():
            if st.session_state.messages is not None:
                for message in st.session_state.messages:
                    if isinstance(message, AIMessage):
                        with st.chat_message('assistant'):
                            st.markdown(message.content)
                    elif isinstance(message, HumanMessage):
                        with st.chat_message('human'):
                            st.markdown(message.content)

        st.session_state.user_input = st.chat_input("Type your answer here...", disabled=st.session_state.chat_disabled)

        if st.session_state.messages == []:
            initial_state = { "raw_resume_text": resume_text, "extracted_links": links }
            state = st.session_state.myGraph.invoke(initial_state, config=config1)
        else:
            state = st.session_state.myGraph.invoke(None,config=config1)

        profile = resumeDataToProfile(state['resume_info'])
        profile = add_new_user(profile)
        
        if profile is not None:
            df = pd.read_csv('credentials.csv')
            mask = (df['user_id'].astype(str) == str(st.session_state.user_id)) & (df['password'].astype(str) == str(st.session_state.password))

            if mask.any():
                df.loc[mask, 'candidate_id'] = profile['candidate_id']
                df.to_csv('credentials.csv', index=False)
            else:
                st.error("User not found or password incorrect.")
                
            st.session_state.candidate_id = profile['candidate_id']
            st.markdown('<div style="background-color:rgba(0, 210, 255, 0.2); padding:15px; border-radius:10px; color:white; text-align:center;">🎉 Candidate Profile Created Successfully! Redirecting...</div>', unsafe_allow_html=True)
            time.sleep(2)
            st.session_state.page = 'recommendation'
            st.rerun()

# ==========================================
# RECOMMENDATION PAGE
# ==========================================

if st.session_state.page == 'recommendation':
    render_sidebar()
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 40px;">
            <h1>✨ Personalized Recommendations</h1>
            <p style="font-size: 1.1em; opacity: 0.8;">AI-curated opportunities matching your unique profile</p>
        </div>
    """, unsafe_allow_html=True)

    recommendations = get_internship_recommendations_for_candidate(st.session_state.candidate_id)
    internship_ids = []
    for internships in recommendations:
        internship_ids.append(internships[0].metadata['job_id'])
    
    # ---------------- Direct Recommendations (ROW WISE) ----------------
    
    for i, ids in enumerate(internship_ids):
        internship = get_internship_by_id(ids)
        
        card_html = f"""
        <div class="job-card">
            <span class="company-badge">🏢 {internship.get('company_name', 'Company')}</span>
            <div class="role-title">{internship.get('title', 'Role')}</div>
            <p style="color: #ccc; font-size: 0.95em; line-height: 1.5; height: 75px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">
                {internship.get('description', '')}
            </p>
            <div style="margin-top: 10px; margin-bottom: 10px;">
                <span style="color: #bbb; font-size: 0.85em;">📍 {internship.get('location', 'Remote')} &nbsp; | &nbsp; 💰 {internship.get('stipend', 'N/A')}</span>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        btn_c1, btn_c2, btn_spacer = st.columns([1, 1, 3]) 
        with btn_c1:
            if st.button(f"View Details 👁️", key=f"view_{ids}_{i}", use_container_width=True):
                st.session_state.selected_internship_id = ids
                st.session_state.page = 'internship_details'
                st.rerun()
        with btn_c2:
            st.button(f"Apply 🚀", key=f"apply_{ids}_{i}", use_container_width=True)
            
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    # ---------------- Collaborative Recommendations ----------------
    
    st.markdown("---")
    st.markdown("""
        <div style="margin-top: 40px; margin-bottom: 20px;">
            <h3>👥 People Like You Also Viewed</h3>
        </div>
    """, unsafe_allow_html=True)

    collaborative_recommendations = collaborative_filtering_recommendations(st.session_state.candidate_id)[:5]
    
    cols = st.columns(3)
    
    for idx, ids in enumerate(collaborative_recommendations):
        internship = get_internship_by_id(ids)
        col_idx = idx % 3
        
        with cols[col_idx]:
             st.markdown(f"""
            <div class="job-card" style="padding: 15px; border-color: rgba(255,255,255,0.05);">
                <span class="company-badge" style="background: rgba(255,255,255,0.1); color: white;">{internship.get('company_name')}</span>
                <div style="font-weight:700; font-size: 1.1em; color:white; margin: 10px 0;">{internship.get('title')}</div>
                <p style="font-size: 0.8em; color: #999;">{internship.get('location', '')}</p>
            </div>
            """, unsafe_allow_html=True)
             
             if st.button("View Details", key=f"col_view_{ids}_{idx}", use_container_width=True):
                 st.session_state.selected_internship_id = ids
                 st.session_state.page = 'internship_details'
                 st.rerun()

# ==========================================
# INTERNSHIP DETAIL PAGE
# ==========================================

if st.session_state.page == 'internship_details':
    render_sidebar()
    
    if st.session_state.selected_internship_id:
        data = get_internship_by_id(st.session_state.selected_internship_id)
        
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.page = 'recommendation'
                st.rerun()
        
        st.markdown(f"""
        <div class="job-card" style="margin-top: 20px; border-top: 5px solid #00d2ff;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 style="margin-bottom: 5px; font-size: 2.5em;">{data.get('title', 'Role')}</h1>
                    <h3 style="color: #00d2ff; opacity: 0.9;">🏢 {data.get('company_name', 'Company Name')}</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-box"><div class="metric-label">Stipend</div><div class="metric-value">{data.get('stipend', 'N/A')}</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box"><div class="metric-label">Location</div><div class="metric-value">{data.get('location', 'Remote')}</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-box"><div class="metric-label">Duration</div><div class="metric-value">{data.get('duration', 'N/A')}</div></div>""", unsafe_allow_html=True)
        with m4:
             st.markdown(f"""<div class="metric-box"><div class="metric-label">Field</div><div class="metric-value">{data.get('field', 'General')}</div></div>""", unsafe_allow_html=True)

        c_left, c_right = st.columns([2, 1], gap="large")
        
        with c_left:
            st.markdown('<div class="section-header">📝 Job Description</div>', unsafe_allow_html=True)
            st.write(data.get('description', 'No description available.'))
            
            st.markdown('<div class="section-header">🎯 Candidate Expectations</div>', unsafe_allow_html=True)
            st.write(data.get('candidate_expectations', 'No specific expectations listed.'))

        with c_right:
            st.markdown('<div class="section-header">🛠️ Skills Required</div>', unsafe_allow_html=True)
            skills = data.get('skills_required', [])
            if isinstance(skills, list):
                html_skills = "".join([f'<span class="skill-chip">{skill}</span>' for skill in skills])
                st.markdown(f"<div>{html_skills}</div>", unsafe_allow_html=True)
            else:
                st.info("No specific skills listed.")
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.button("Apply for this Position 🚀", key="detail_apply_btn", use_container_width=True)

        # ---------------- SIMILAR INTERNSHIPS SECTION ----------------
        
        st.markdown("---")
        st.markdown("### 🔄 Similar Opportunities")
        st.markdown("<br>", unsafe_allow_html=True)

        # Retrieve similar internships based on the CURRENT viewed ID
        # Note: k=5 allows us to filter out the current job and still show ~4 items
        similar_results = get_similar_internships(st.session_state.selected_internship_id, k=5)
        
        for idx, (doc, score) in enumerate(similar_results):
            sim_id = doc.metadata['job_id']
            
            # Skip if the similar result is the exact same job we are viewing
            if sim_id == st.session_state.selected_internship_id:
                continue
                
            sim_data = get_internship_by_id(sim_id)

            # Reusing the Row-Wise Card Design
            card_html = f"""
            <div class="job-card" style="border-left: 5px solid #a0a0a0;">
                <span class="company-badge">🏢 {sim_data.get('company_name', 'Company')}</span>
                <div class="role-title">{sim_data.get('title', 'Role')}</div>
                <p style="color: #ccc; font-size: 0.95em; line-height: 1.5; height: 60px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">
                    {sim_data.get('description', '')}
                </p>
                <div style="margin-top: 5px;">
                    <span style="color: #bbb; font-size: 0.85em;">📍 {sim_data.get('location', 'Remote')}</span>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Action Buttons for Similar Jobs
            # IMPORTANT: Unique keys using 'sim_{idx}' to prevent key conflicts
            btn_s1, btn_s2, btn_spacer = st.columns([1, 1, 3]) 
            with btn_s1:
                if st.button(f"View Details 👁️", key=f"sim_view_{sim_id}_{idx}", use_container_width=True):
                    st.session_state.selected_internship_id = sim_id
                    st.rerun() # Rerun to refresh the details page with the new ID
            with btn_s2:
                st.button(f"Apply 🚀", key=f"sim_apply_{sim_id}_{idx}", use_container_width=True)
            
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    else:
        st.error("No Internship Selected")
        if st.button("Go Back"):
            st.session_state.page = 'recommendation'
            st.rerun()