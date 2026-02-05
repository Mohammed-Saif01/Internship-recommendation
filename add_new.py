import ast
import pandas as pd
from pydantic.v1 import BaseModel, Field
from typing import List, Dict, Literal, Optional
from datetime import datetime
from making_user_interaction_matrix import add_new_candidate_to_vectorStore
from making_vectorstore import internship_to_string, CustomGoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class ContactInfo(BaseModel):
    """
    Model representing the candidate's contact details.
    """
    email: str = Field(
        ..., 
        description="The candidate's primary email address for communication."
    )
    phone: str = Field(
        ..., 
        description="The candidate's 10-digit mobile phone number."
    )

class Skill(BaseModel):
    name: str
    proficiency: float = Field(description="0.0 to 1.0")

class SocialLink(BaseModel):
    platform: str
    url: str

class CandidateProfile(BaseModel):
    """
    Comprehensive profile model for a job candidate, including personal info,
    preferences, and platform behavioral analytics.
    """
    
    # --- Personal Information ---
    name: str = Field(
        ..., 
        description="Full legal name of the candidate."
    )
    education: str = Field(
        ..., 
        description="Current educational qualification or degree status (e.g., 'B.Com Accounting, Final Year')."
    )
    
    skills: List[str] = Field(
        ..., 
        description="List of technical or soft skills claimed by the candidate."
    )

    skill_proficiency: Dict[str, float] = Field(
        ..., 
        description="Dictionary mapping skill names to a proficiency score (0.0 to 1.0) based on assessments. eg. {'Tally': 0.85, 'MS Excel': 0.9}"
    )
    
    contact: ContactInfo = Field(
        ..., 
        description="Nested object containing email and phone details."
    )
    # social_profiles: Dict[str, str] = Field(
    #     ..., 
    #     description="Dictionary mapping platform names (e.g., 'linkedin', 'github') to profile URLs. eg. {'linkedin': 'https://linkedin.com/in/username'}"
    # )
    location: str = Field(
        ..., 
        description="The candidate's current residential city."
    )

    # --- Job Preferences ---
    preferred_locations: List[str] = Field(
        ..., 
        description="List of cities or regions where the candidate is willing to work."
    )
    preferred_fields: List[str] = Field(
        ..., 
        description="List of job domains the candidate is interested in (e.g., 'Finance', 'Accounting')."
    )
    avoid_fields: List[str] = Field(
        ..., 
        description="List of job domains the candidate specifically does not want to pursue."
    )
    preferred_mode: Literal['onsite', 'remote', 'hybrid'] = Field(
        ..., 
        description="Preferred working arrangement: 'onsite', 'remote', or 'hybrid'."
    )
    preferred_stipend: float = Field(
        ..., 
        description="Minimum expected monthly stipend value."
    )
    projects: List[str] = Field(
        ..., 
        description="List of significant project titles or short descriptions worked on by the candidate."
    )
    experience_level: str = Field(
        default='fresher', 
        description="Self-designated experience tier (e.g., 'fresher', 'intermediate', 'expert')."
    )
    resume_text: str = Field(
        ..., 
        description="Full text extracted from the candidate's uploaded resume/CV for NLP processing."
    )
    
    # --- Job Interaction Lists (Behavioral) ---
    saved_jobs: List[str] = Field(
        default_factory=list, 
        description="List of Job IDs the candidate has bookmarked/saved."
    )
    applied_jobs: List[str] = Field(
        default_factory=list, 
        description="List of Job IDs the candidate has actively applied to."
    )
    rejected_jobs: List[str] = Field(
        default_factory=list, 
        description="List of Job IDs where the candidate's application was rejected."
    )
    interviewed_jobs: List[str] = Field(
        default_factory=list, 
        description="List of Job IDs where the candidate reached the interview stage."
    )
    
    # --- Analytics / Metrics (Computed) ---
    genre_time_spent: Dict[str, float] = Field(
        default_factory=dict, 
        description="Time spent (in minutes) browsing specific job categories (e.g., {'finance': 14.2})."
    )
    genre_clicks: Dict[str, int] = Field(
        default_factory=dict, 
        description="Count of clicks on job cards belonging to specific categories."
    )
    location_time_spent: Dict[str, float] = Field(
        default_factory=dict, 
        description="Time spent (in minutes) browsing jobs in specific locations."
    )
    location_clicks: Dict[str, int] = Field(
        default_factory=dict, 
        description="Count of clicks on job cards for specific locations."
    )
    
    # --- Meta / Scoring ---
    career_goal: str = Field(
        default="", 
        description="A short statement describing the candidate's long-term professional aspirations."
    )
    last_active: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Timestamp of the user's most recent interaction with the platform."
    )
    activity_score: float = Field(
        default=0.0, 
        description="Computed score (0.0 to 1.0) representing user engagement and platform usage frequency."
    )

    candidate_id: str = Field(
        default=None, 
        description="Unique identifier for the candidate profile. Leave empty to auto-generate."
    )




# --- MODEL A: The Extraction Schema (Input from LLM) ---
class ResumeExtractionData(BaseModel):
    name: Optional[str] = Field(default=None, description="Full legal name. Fill None if not found.")
    education: Optional[str] = Field(default=None, description="Latest degree/qualification. Fill None if not found.")
    location: Optional[str] = Field(None, description="Current city. Fill None if not found.")
    
    # We use Lists here because LLMs handle them better than Dicts
    skills: Optional[List[Skill]] = Field(default_factory=list)
    social_links: Optional[List[SocialLink]] = Field(default_factory=list)
    
    contact: ContactInfo = Field(default=None)
    
    projects: Optional[List[str]] = Field(default_factory=list)
    experience_level: Optional[str] = Field("fresher", description="fresher, intermediate, or expert")
    
    # Preferences (Optional because they might not be in the resume)
    preferred_mode: Optional[Literal['onsite', 'remote', 'hybrid']] = Field(default=None)

    preferred_locations: Optional[List[str]] = Field(
        ..., 
        description="List of cities or regions where the candidate is willing to work."
    )
    preferred_fields: List[str] = Field(
        ..., 
        description="List of job domains the candidate is interested in (e.g., 'Finance', 'Accounting')."
    )
    avoid_fields: List[str] = Field(
        ..., 
        description="List of job domains the candidate specifically does not want to pursue."
    )
    preferred_mode: Literal['onsite', 'remote', 'hybrid'] = Field(
        ..., 
        description="Preferred working arrangement: 'onsite', 'remote', or 'hybrid'."
    )
    preferred_stipend: float = Field(
        ..., 
        description="Minimum expected monthly stipend value."
    )

    resume_summary : str = Field(
        ...,
        description="Summary of whole resume"
    )

class InternshipJob(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the job posting")
    title: str = Field(..., description="The designation of the internship role")
    company_name: str = Field(..., description="Name of the company offering the internship")
    field: str = Field(..., description="The industry or domain (e.g., Logistics/Railways)")
    location: str = Field(..., description="City and State of the internship")
    duration: str = Field(..., description="Duration of the internship (e.g., '4 months')")
    
    skills_required: List[str] = Field(
        ..., 
        description="List of technical or soft skills required"
    )
    
    stipend: str = Field(..., description="String representation of stipend (e.g., '₹15,000/month')")
    stipend_num: int = Field(..., description="Numeric value of the stipend for filtering/sorting")
    
    description: str = Field(..., description="Detailed responsibilities and tasks")
    candidate_expectations: str = Field(..., description="What the company expects from candidates")

def add_new_user(candidate_data: dict):
    """
    Function to add a new candidate profile to the system.
    Validates input data against the CandidateProfile model.
    """
    try:
        candidate_profile = CandidateProfile(**candidate_data)
        # Here, you would typically save the candidate_profile to a database or vector store.
        candidate_id = len(pd.read_csv('candidates.csv')['candidate_id'].tolist())
        candidate_profile.candidate_id = f"candidate_{candidate_id + 1}"
        candidate_profile = candidate_profile.dict()

        new_row = pd.Series(candidate_profile)

        # Save the updated candidate profile to the CSV file
        df = pd.read_csv('candidates.csv')
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        df.to_csv('candidates.csv', index=False)
        print(f"Successfully added candidate: {candidate_profile['candidate_id']}")

        print(new_row.to_frame().T)
        # Adding candidate to vector store
        add_new_candidate_to_vectorStore(candidate_id=candidate_profile['candidate_id'])
        
        return candidate_profile
    
    except Exception as e:
        print("Some Exception occur returning None")
        print(f"Error adding candidate: {e}")
        return None


def add_new_internship(internship_data: dict):
    """
    Function to add a new internship job posting to the system.
    Validates input data against the InternshipJob model.
    """
    try:
        internship_job = InternshipJob(**internship_data)
        # Here, you would typically save the internship_job to a database or vector store.

        internship_job.job_id = f"JOB_{len(pd.read_csv('internships.csv')['job_id'].tolist())+1}"

        print(f"Validated internship job data for job_id: {internship_job.job_id}")

        print("Preparing to add internship job to vector store...")

        print("Initializing embedding model...")

        emb = CustomGoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001', google_api_key='AIzaSyAgTTDHqbdm5bymFwHO7jqDKm3hg4xhRT0')

        print("Embedding model initialized.")

        print("Creating vector store...")

        vector_store = Chroma(
            embedding_function=emb,
            persist_directory='internship_recommendation_db',
        )

        print("Vector store created.")

        doc = Document(page_content=internship_to_string(internship_job.model_dump()), metadata={
            'job_id': internship_job.job_id,
            'stipend': internship_job.stipend_num,
        })

        print("Adding documents to vector store...")

        vector_store.add_documents([doc])
        print(f"Added new internship job {internship_job.job_id} to vector store.")


        print("Saving the new internship job to the CSV file...")
        # Save the new internship job to the CSV file
        df = pd.read_csv('internships.csv')
        new_row = pd.Series(internship_job.model_dump())
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        df.to_csv('internships.csv', index=False)
        print(f"Successfully added internship job: {internship_job.job_id}")
        
        return internship_job
    
    except Exception as e:
        print(f"Error adding internship job: {e}")
        return None

def save_job(candidate_id: str, job_id: str):
    """
    Function to save a job for a candidate.
    Updates the candidate's saved_jobs list in the CSV file.
    """
    try:
        df = pd.read_csv('candidates.csv')
        candidate_index = df.index[df['candidate_id'] == candidate_id].tolist()
        if not candidate_index:
            print(f"No candidate found with id: {candidate_id}")
            return None
        candidate_index = candidate_index[0]
        
        saved_jobs = ast.literal_eval(df.at[candidate_index, 'saved_jobs']) if pd.notna(df.at[candidate_index, 'saved_jobs']) else []
        if job_id not in saved_jobs:
            saved_jobs.append(job_id)
            df.at[candidate_index, 'saved_jobs'] = str(saved_jobs)
            df.to_csv('candidates.csv', index=False)
            print(f"Job {job_id} saved for candidate {candidate_id}.")
        else:
            print(f"Job {job_id} is already saved for candidate {candidate_id}.")
        
        return saved_jobs

    except Exception as e:
        print(f"Error saving job: {e}")
        return None

def save_applied_job(candidate_id: str, job_id: str):
    """
    Function to mark a job as applied for a candidate.
    Updates the candidate's applied_jobs list in the CSV file.
    """
    try:
        df = pd.read_csv('candidates.csv')
        candidate_index = df.index[df['candidate_id'] == candidate_id].tolist()
        if not candidate_index:
            print(f"No candidate found with id: {candidate_id}")
            return None
        candidate_index = candidate_index[0]
        
        applied_jobs = ast.literal_eval(df.at[candidate_index, 'applied_jobs']) if pd.notna(df.at[candidate_index, 'applied_jobs']) else []
        if job_id not in applied_jobs:
            applied_jobs.append(job_id)
            df.at[candidate_index, 'applied_jobs'] = str(applied_jobs)
            df.to_csv('candidates.csv', index=False)
            print(f"Job {job_id} marked as applied for candidate {candidate_id}.")
        else:
            print(f"Job {job_id} is already marked as applied for candidate {candidate_id}.")
        
        return applied_jobs

    except Exception as e:
        print(f"Error marking job as applied: {e}")
        return None



if __name__ == "__main__":


    # data = {
    #     "name": "Sarthak Jain",
    #     "education": "B.Com Accounting, Final Year",
    #     "skills": ["Tally", "MS Excel", "Bookkeeping", "GST Filing"],
    #     "skill_proficiency": {"Tally": 0.88, "MS Excel": 0.92, "Bookkeeping": 0.89},
    #     "contact": {"email": "sarthak.jain@example.com", "phone": "9812345678"},
    #     "social_profiles": {"linkedin": "https://linkedin.com/in/sarthakj", "github": ""},
    #     "location": "Jaipur",
    #     "preferred_locations": ["Jaipur", "Gurgaon"],
    #     "preferred_fields": ["Accounting", "Finance"],
    #     "avoid_fields": ["Marketing"],
    #     "preferred_mode": "onsite",
    #     "preferred_stipend": "10000",
    #     "projects": ["Automated Financial Ledger in Excel"],
    #     "experience_level": "intermediate",
    #     "resume_text": "Detail-oriented commerce student...",
    #     "saved_jobs": ["JOB_015", "JOB_007"],
    #     "applied_jobs": ["JOB_015"],
    #     "rejected_jobs": [],
    #     "interviewed_jobs": ["JOB_015"],
    #     "genre_time_spent": {"finance": 14.2},
    #     "genre_clicks": {"finance": 11},
    #     "location_time_spent": {"Jaipur": 5.4},
    #     "location_clicks": {"Jaipur": 4},
    #     "career_goal": "Work as a Financial Analyst",
    #     "last_active": "2025-01-19T09:15:00",
    #     "activity_score": 0.72,
    #     "candidate_id": ""
    # }

    # add_new_user(data)
    # print("Candidate addition process completed.")

    # data = {
    #     "job_id": "JOB_140",
    #     "title": "Railway Operations Intern",
    #     "company_name": "SpeedRail Logistics",
    #     "field": "Logistics/Railways",
    #     "location": "Bilaspur, Chhattisgarh",
    #     "duration": "4 months",
    #     "skills_required": ["Parcel Management", "Rake Planning", "FOIS", "Safety"],
    #     "stipend": "₹15,000/month",
    #     "stipend_num": 15000,
    #     "description": "Assist in parcel and freight train operations, track rakes, coordinate loading/unloading, and learn railway systems.",
    #     "candidate_expectations": "We seek disciplined individuals fascinated by railways. Attention to timetables, safety consciousness, and coordination skills are required."
    # }
    # add_new_internship(data)
    # print("Internship addition process completed.")
    pass