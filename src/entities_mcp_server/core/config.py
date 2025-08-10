from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    QDRANT_URL: str = "qdrant"
    QDRANT_COLLECTION_NAME: str = "job-postings-collection-hybrid-search"
    EMBEDDING_MODEL: str 
    EMBEDDING_MODEL_PROVIDER: str

    model_config = SettingsConfigDict(env_file=".env")

config = Config()


ENTITY_LABELS = {
    "ORG",      # Company names
    "GPE",      # Locations
    "LOC",      # Additional location info
    # "PERSON",   # Recruiter/contact name
    "DATE",     # Posting dealine etc.
    "MONEY",    # Salary info
    "PRODUCT",  # Tools, software products
    "LANGUAGE", # Programming/spoken language
    "NORP",     # Nationalities, religios, political groups
    "WORK_OF_ART"  # May hold job titles or certifications
}


CUSTOM_PATTERNS = [
    # Emails (simple pattern)
    {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]},

    # URLs (simple regex match using pattern of tokens starting with https?:// or www)
    {"label": "URL", "pattern": [{"TEXT": {"REGEX": r"https?://[^\s]+"}}]},
    {"label": "URL", "pattern": [{"TEXT": {"REGEX": r"www\.[^\s]+"}}]},

    # Phone numbers (simple patterns for digits and separators)
    {"label": "PHONE_NUMBER", "pattern": [
        {"TEXT": {"REGEX": r"\+?\d{1,3}"}}, {"TEXT": {"REGEX": r"[-.\s]?"}}, 
        {"TEXT": {"REGEX": r"\(?\d{1,4}\)?"}}, {"TEXT": {"REGEX": r"[-.\s]?"}}, 
        {"TEXT": {"REGEX": r"\d{1,4}"}}, {"TEXT": {"REGEX": r"[-.\s]?"}}, 
        {"TEXT": {"REGEX": r"\d{1,9}"}}
    ]},

    # Skills (examples, add or extend this list)
    {"label": "SKILL", "pattern": "python"},
    {"label": "SKILL", "pattern": "excel"},
    {"label": "SKILL", "pattern": "project management"},
    {"label": "SKILL", "pattern": "c++"},
    {"label": "SKILL", "pattern": "javascript"},

    # Certifications (examples)
    {"label": "CERTIFICATION", "pattern": "PMP"},
    {"label": "CERTIFICATION", "pattern": "AWS Certified"},
    {"label": "CERTIFICATION", "pattern": "Cisco Certified Network Associate"},

    # Job Titles (examples)
    {"label": "JOB_TITLE", "pattern": "Data Analyst"},
    {"label": "JOB_TITLE", "pattern": "Sales Manager"},
    {"label": "JOB_TITLE", "pattern": "Software Engineer"},
    {"label": "JOB_TITLE", "pattern": "Project Manager"}
]