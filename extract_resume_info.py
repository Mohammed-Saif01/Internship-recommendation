import fitz
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from add_new import CandidateProfile, ResumeExtractionData
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAI

load_dotenv()

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.
    Returns:
        str: The extracted text.    

    """
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_links_from_pdf(file_path):
    """
    Extracts hyperlinks from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of extracted hyperlinks.
    """

    links = []
    with fitz.open(file_path) as doc:
        for page in doc:
            link_dicts = page.get_links()
            for link in link_dicts:
                if 'uri' in link:
                    links.append(link['uri'])
    return links

class customState(TypedDict):
    resume_info : ResumeExtractionData
    raw_resume_text : str
    extracted_links : list[str]
    all_attributes_filled : bool
    missing_fields : list[str]
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
# llm = ChatGroq(model="openai/gpt-oss-120b")
# llm = ChatOpenAI(model='openai/gpt-oss-120b:free')
# llm = ChatOpenAI(model='x-ai/grok-4.1-fast:free')
# llm = OpenAI(model='nousresearch/hermes-3-llama-3.1-405b:free')
# llm = ChatOpenAI(model='qwen/qwen3-coder:free')
# llm = OpenAI(model='deepseek/deepseek-r1-distill-llama-70b:free')
# llm = OpenAI(model='openai/gpt-4.1-nano')




def extract_resume_info(state : customState):

    messages = [
        {"role": "system", "content": "You are an expert resume parser. Extract relevant information from resumes. Don't provide json where needed, use dictionary format."},
        {"role": "user", "content": """Extract relevant information from the following resume text:\n\n{raw_resume_text}\n\nAlso, consider the following links extracted from the resume for additional context: {extracted_links}."""}
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    structured_llm = llm.with_structured_output(ResumeExtractionData)

    response = structured_llm.invoke(prompt.format_messages(raw_resume_text=state.get("raw_resume_text", ""), extracted_links=state.get("extracted_links", [])))

    state['resume_info'] = response

    print(f"Resume info is extracted")

    return state


# def checkIfAllFilled(state : customState):

#     messages = [
#         {"role":"system", "content":"Check if all attributes are filled or not. Find the missing details"},
#         {"role":"user", "content":"Here are all attributes : {resume_info}"}
#     ]

#     prompt = ChatPromptTemplate.from_messages(messages)

#     class output(TypedDict):
#         all_filled : bool
#         missing_fields : list[str]

#     structured_llm = llm.with_structured_output(output)

#     response = structured_llm.invoke(prompt.format_messages(resume_info=state['resume_info']))

#     state['all_attributes_filled'] = response['all_filled']
#     state['missing_fields'] = response['missing_fields']
#     # print("missing fields : ",response['missing_fields'])
#     return state

def checkIfAllFilled(state : customState):
    print("--- Checking All Missing Fields ---")
    
    # We get the list of all keys from your Pydantic model definition
    # This ensures the LLM knows exactly which fields exist in the schema
    # NEW (Pydantic V1 Syntax - Compatible with LangChain)
    all_fields = list(ResumeExtractionData.__fields__.keys())    
    system_prompt = f"""You are a quality assurance checker.
    
    Your task is to check the 'resume_info' JSON for ANY missing values.
    
    The complete list of required fields is: {all_fields}
    
    RULES for "Missing":
    1. Value is None or null.
    2. Value is an empty list [].
    3. Value is 0.0 or 0 (specifically for numeric fields like stipend).
    4. Value is an empty string "".
    
    OUTPUT:
    - Set 'all_filled' to False if ANY required field (except exclusions) is missing.
    - Return the list of keys that are missing.
    """

    messages = [
        ("system", system_prompt),
        ("user", "Here is the current resume_info: {resume_info}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    class CheckOutput(TypedDict):
        all_filled : bool
        missing_fields : list[str]

    structured_llm = llm.with_structured_output(CheckOutput)
    
    # We invoke the chain
    chain = prompt | structured_llm
    response = chain.invoke({"resume_info": state['resume_info']})

    return {
        "all_attributes_filled": response['all_filled'],
        "missing_fields": response['missing_fields']
    }



def askDetails(state : customState):

    # ask_system_prompt = """You are a recruiter acting as an interviewer. 
    # Your goal is to fill the following missing fields: {missing_fields}.
 
    # CRITICAL INSTRUCTIONS : 
    # 1. You must be asking questions to fill these important fields : preferred location, preferred fields, preferred stipend, avoid fields
    # 2. Your goal is to fill all the attributes in resume_info. If all important fields filled then try to ask questions to fill missing fields.
    # 3. Ask question one by one
    # 4. You must ask these questions (Avoid if already asked): 
    #         1. About preferred location
    #         2. About preferred fields
    #         3. About preferred Stipend
    #         4. Abour the fields candidate wants to avoid"""

    ask_system_prompt = """
You are a recruiter interviewing a candidate.

Your PRIMARY GOAL:
Collect the following four fields in this order:
1. preferred_locations
2. preferred_fields
3. preferred_stipend
4. avoid_fields

Only when ALL four fields are filled and valid,
then you may ask about any remaining missing fields: {missing_fields}

RULES:
- Ask ONLY ONE question at a time.
- Ask in the exact order shown above.
- If a field is already filled or valid, skip it and ask the next one.
- Keep each question short and specific to the field.
- Do not make assumptions or insert values into fields.
- Do not summarize anything.
- Only output the QUESTION to the user.

"""

    if state['messages']  == []:
        messages = [
            {"role":"system", "content":ask_system_prompt},
            {"role":"user","content":"current resume info : {resume_info}"}
        ]

    else:

        messages = [
            {"role":"system", "content":ask_system_prompt},
            MessagesPlaceholder(variable_name="chat_history"),
            {"role":"user","content":"current resume info : {resume_info}"}
        ]

    


    prompt = ChatPromptTemplate.from_messages(messages)

    class output(TypedDict):
        resume_info : ResumeExtractionData
        all_attributes_filled : bool 
        missing_fields : list[str]

    parser = StrOutputParser()
    question = llm | parser
    llm_question = question.invoke(prompt.format_messages(resume_info = state['resume_info'], missing_fields=state['missing_fields'], chat_history=state['messages']))

    st.session_state.messages.append(AIMessage(content=llm_question))

    with st.chat_message('assistant'):
        st.markdown(llm_question)
    
    st.session_state.chat_disabled = False

    return {'messages':[AIMessage(content=llm_question)]}

    # if user_input:
    #     with st.chat_message('human'):
    #         st.markdown(user_input)
    # user_input = input(f"Assistant : {llm_question} : \n")

    # extraction_system_prompt = "You are a data entry expert. Update the 'resume_info' based on the User's latest answer."

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

    updates_messages = [
        {"role":"system", "content":extraction_system_prompt},
        {"role":"user", "content":"Here are all attributes : {resume_info} . \n Missing fields : {missing_fields}"},
        {"role":"assistant","content":llm_question},
        {"role":"user","content":user_input}
    ]

    new_messages = [
        AIMessage(content=llm_question), 
        HumanMessage(content=user_input)
    ]

    new_prompt = ChatPromptTemplate.from_messages(updates_messages)

    # print(new_prompt.format_messages(resume_info = state['resume_info'], missing_fields=state['missing_fields']))

    structured_llm = llm.with_structured_output(output)

    response = structured_llm.invoke(new_prompt.format_messages(resume_info = state['resume_info'], missing_fields=state['missing_fields']))

    return {
        "resume_info": response['resume_info'],
        "all_attributes_filled": response['all_attributes_filled'],
        "missing_fields": response['missing_fields'],
        "messages": new_messages
    }

# def user_input(state : customState):

    


def routing(state : customState):
    if state['all_attributes_filled'] is True:
        return 'END'
    else :
        return 'ask'

graph = StateGraph(customState)

graph.add_node('eri',extract_resume_info)
graph.add_node('chk',checkIfAllFilled)
graph.add_node('ask',askDetails)

graph.add_edge(START, 'eri')
graph.add_edge('eri', 'chk')
graph.add_conditional_edges('chk', routing, {'ask':'ask','END':END})
graph.add_edge('ask','chk')

myGraph = graph.compile()

def resumeDataToProfile(state : ResumeExtractionData):
    profile = {
        'name' : state['name'],
        'education' : state['education'],
        'skills' : [skill['name'] for skill in state['skills']],
        'skill_proficiency' : {skill['name'] : skill['proficiency'] for skill in state['skills']},
        'contact' : state['contact'],
        'location' : state['location'],
        'preferred_locations' : state['preferred_locations'],
        'preferred_fields' : state['preferred_fields'],
        'avoid_fields' : state['avoid_fields'],
        'preferred_mode' : state['preferred_mode'],
        'preferred_stipend' : state['preferred_stipend'],
        'resume_text': state['resume_summary'],
        'projects': state['projects']
    }
    return profile

if __name__ == "__main__":

    resume_text = extract_text_from_pdf('RAJAT JAIN 22EJICS126.pdf')
    # print(resume_text)
    links = extract_links_from_pdf('RAJAT JAIN 22EJICS126.pdf')
    # print(links)

    initial_state = {
        "raw_resume_text": resume_text,
        "extracted_links": links,
    }
    
    state = myGraph.invoke(initial_state)
    # print(state)

    # print(state['resume_info'])

    profile = resumeDataToProfile(state['resume_info'])
    print(profile)

