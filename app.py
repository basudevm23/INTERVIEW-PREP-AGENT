import os
import sys
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(__file__))

from utils import GeminiClient, LocalLoRAModel

st.set_page_config(page_title="Interview Prep Agent", layout="wide")
st.title("💼 Interview Prep Agent — Gemini + TinyLlama LoRA")

# ====== Clients ======
gemini = GeminiClient()
local_lora = None

with st.sidebar:
    st.header("Session Configuration")
    role = st.text_input("Target job role", value="Software Engineer")
    mode = st.selectbox("Mode", ["Gemini: Generate Questions", "Interactive Interview (Gemini + LoRA Score)"])
    n_q = st.slider("Number of questions", 3, 10, 5)
    model_dir = st.text_input("Local base model dir (TinyLlama)", value=os.getenv("LLM_LOCAL_MODEL", "./tiny_llama"))
    adapter_dir = st.text_input("LoRA adapter dir", value=os.getenv("LORA_MODEL_DIR", "./adapters/checkpoint-1"))

st.markdown("---")

# ====== Generate Questions ======
if mode == "Gemini: Generate Questions":
    st.subheader("Generate questions from resume or custom prompt")

    resume_file = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"])
    custom_prompt = st.text_input("Or write a prompt instead (optional):", "")

    if st.button("Generate Questions"):
        if resume_file and not custom_prompt:
            if resume_file.type == "application/pdf":
                import PyPDF2
                try:
                    reader = PyPDF2.PdfReader(resume_file)
                    resume_text = "\n".join([p.extract_text() or "" for p in reader.pages])
                except Exception:
                    resume_text = resume_file.getvalue().decode("utf-8", errors="ignore")
            else:
                resume_text = resume_file.getvalue().decode("utf-8", errors="ignore")

            prompt = f"You are an expert interviewer. Given this resume and role, create {n_q} tailored interview questions.\n\nRole: {role}\n\nResume:\n{resume_text}\n\nReturn a numbered list."
        elif custom_prompt:
            prompt = custom_prompt
        else:
            st.warning("Provide a resume or a prompt.")
            prompt = None

        if prompt:
            with st.spinner("Generating questions via Gemini..."):
                output = gemini.ask(prompt)
            st.success("Questions generated")
            st.text_area("Gemini output", output, height=300)

# ====== Interactive Interview ======
# ====== Interactive Interview ======
elif mode == "Interactive Interview (Gemini + LoRA Score)":
    st.subheader("Interactive interview — Gemini asks, you answer, LoRA scores")

    topic = st.text_input("What would you like to practice?", "Data Analyst: SQL + Pandas")
    start = st.button("Start Interview")

    if start:
        with st.spinner("Generating questions..."):
            qprompt = f"Generate {n_q} interview questions for: {topic}. Return a numbered list."
            output = gemini.ask(qprompt)
        # questions = [ln.strip() for ln in output.splitlines() if ln.strip()]
        # st.session_state["questions"] = questions[:n_q]
        raw_lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        if raw_lines:
          # Print the first line as introductory text
          st.markdown(raw_lines[0])
        # Then set the remaining lines as actual questions
        questions = raw_lines[1:]
        # Make sure exactly n_q questions are stored
        st.session_state["questions"] = questions[:n_q]

    if "questions" in st.session_state:
        questions = st.session_state["questions"]

        if local_lora is None:
            try:
                local_lora = LocalLoRAModel(model_dir, adapter_dir)
            except Exception as e:
                st.error(f"Failed to load LoRA model: {e}")

        for i, q in enumerate(questions, 1):
            st.markdown(f"### Q{i}. {q}")
            ans_key = f"ans_{i}"
            answer = st.text_area(f"Your answer for Q{i}:", key=ans_key, height=120)

            if st.button(f"Score Q{i}", key=f"score_{i}"):
                if not answer.strip():
                    st.warning("Please write an answer before scoring.")
                elif local_lora:
                    with st.spinner("Scoring with LoRA..."):
                        score = local_lora.score_answer(q, answer)
                    st.success(f"LoRA Score: {score}/10")

                    with st.spinner("Fetching model feedback from Gemini..."):
                        feedback = gemini.ask(
                            f"Provide a concise feedback and model answer for this interview question:\n{q}\nAnswer:\n{answer}"
                        )
                    st.markdown("**Model Answer & Feedback:**")
                    st.write(feedback)
                else:
                    st.error("LoRA model not loaded. Check your adapter path.")


st.markdown("---")
st.caption("App version: Gemini + TinyLlama LoRA (Fixed API version)")
