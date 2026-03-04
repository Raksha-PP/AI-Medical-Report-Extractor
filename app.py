import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

# -----------------------------
# Initialize Ollama Model
# -----------------------------
llm = ChatOllama(
    model="llama3",
    temperature=0,
    format="json",  # Forces JSON mode in Ollama
)

st.set_page_config(page_title="AI Medical Report Extractor", layout="wide")

st.title("🩺 AI Medical Report Extractor")
st.write("Extract structured medical insights using Local LLM (Ollama)")

# -----------------------------
# User Input
# -----------------------------
medical_report = st.text_area(
    "📄 Paste Medical Report Here:",
    height=300,
)

if st.button("🔍 Analyze Report"):

    if not medical_report.strip():
        st.warning("Please enter a medical report.")
        st.stop()

    # ==============================
    # FEATURE 1: Patient Info
    # ==============================

    class PatientInfo(BaseModel):
        patient_name: str = Field(description="Full name of the patient")
        age: int = Field(description="Age of the patient in years")
        gender: str = Field(description="Gender of the patient")
        diagnosis: str = Field(description="Main medical diagnosis")
        prescribed_medications: List[str] = Field(description="List of prescribed medications")

    patient_parser = PydanticOutputParser(pydantic_object=PatientInfo)

    patient_prompt = PromptTemplate(
        template="""
You are a medical information extraction system.

Extract patient details from the medical report below.

Medical Report:
{report}

IMPORTANT:
Return ONLY valid JSON.
Do NOT add explanation.
Do NOT wrap in markdown.

{format_instructions}
""",
        input_variables=["report"],
        partial_variables={"format_instructions": patient_parser.get_format_instructions()},
    )

    patient_chain = patient_prompt | llm | patient_parser

    # ==============================
    # FEATURE 2: Risk Assessment
    # ==============================

    class RiskAssessment(BaseModel):
        severity_level: str = Field(description="Severity level: Mild, Moderate, or Severe")
        critical_findings: List[str] = Field(description="Important critical findings")
        recommended_actions: List[str] = Field(description="Recommended medical actions")

    risk_parser = PydanticOutputParser(pydantic_object=RiskAssessment)

    risk_prompt = PromptTemplate(
        template="""
You are a medical risk assessment AI.

Based on the medical report below, determine:

- Severity level (Mild, Moderate, Severe)
- Critical findings
- Recommended actions

Medical Report:
{report}

IMPORTANT:
Return ONLY valid JSON.
Do NOT add explanation.
Do NOT wrap in markdown.

{format_instructions}
""",
        input_variables=["report"],
        partial_variables={"format_instructions": risk_parser.get_format_instructions()},
    )

    risk_chain = risk_prompt | llm | risk_parser

    # ==============================
    # FEATURE 3: Doctor Summary
    # ==============================

    summary_prompt = PromptTemplate(
        template="""
You are a professional medical assistant.

Provide a concise doctor-friendly summary of the following medical report.

Medical Report:
{report}

Return plain text only.
""",
        input_variables=["report"],
    )

    summary_parser = StrOutputParser()
    summary_chain = summary_prompt | llm | summary_parser

    # ==============================
    # Execute
    # ==============================

    try:
        with st.spinner("Analyzing report..."):

            patient_info = patient_chain.invoke({"report": medical_report})
            risk_info = risk_chain.invoke({"report": medical_report})
            doctor_summary = summary_chain.invoke({"report": medical_report})

        # -----------------------------
        # Display Results
        # -----------------------------

        st.success("Analysis Complete ✅")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Patient Information")
            st.json(patient_info.model_dump())

        with col2:
            st.subheader("⚠️ Risk Assessment")
            st.json(risk_info.model_dump())

        st.subheader("🩻 Doctor-Friendly Summary")
        st.write(doctor_summary)

    except Exception as e:
        st.error("Error parsing model output. Try again.")
        st.exception(e)