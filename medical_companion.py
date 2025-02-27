import streamlit as st
from googletrans import Translator
from openai import OpenAI
import os
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import re

load_dotenv()

nltk.download('punkt')

# ✅ Set Streamlit page config at the very top
st.set_page_config(page_title="AI Medical Translator", layout="wide")

# ✅ Hardcoded OpenAI API Key (Replace with your actual API key)
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# ✅ Initialize Translator
translator = Translator()

# ✅ Initialize variables at the top level
formatted_transcript = ""
summary = ""

# ✅ Streamlit UI
st.title("🩺 AI-Powered Medical Transcription & Translation")

# ✅ Upload or select audio file
st.subheader("1️⃣ Choose or Upload a Doctor-Patient Conversation Audio File")
st.info("🔹 Choose or upload a doctor-patient conversation audio file to get a transcription, AI visit summary in your chosen language, and key medical term explanations.")
audio_source = st.radio("Select audio source", ["Choose from examples", "Upload your own"])


if audio_source == "Choose from examples":
    audio_file_name = st.selectbox(
        "Select an example audio file",
        ["Sample_Audio_1.mp3"],
        index=0
    )
    proceed_button = st.button("Proceed with selected file")
    audio_file = f"{audio_file_name}" if proceed_button and audio_file_name != "Select an option..." else None
else:
    audio_file = st.file_uploader("Upload an MP3/WAV file", type=["mp3", "wav"])

if audio_file:
    # Check file size
    if isinstance(audio_file, str):
        file_size = os.path.getsize(audio_file)
    else:
        file_size = audio_file.size
        
    if file_size > 25 * 1024 * 1024:  # 25MB in bytes
        st.error("File size exceeds 25MB limit. Please upload a smaller file.")
        st.stop()
        
    # For uploaded files, save to temporary location
    if isinstance(audio_file, str):
        temp_audio_path = audio_file  # Using direct path for example files
    else:
        # Save uploaded file to a temporary location
        temp_audio_path = f"/tmp/{audio_file.name}"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

    st.success(f"✅ Audio file selected: {audio_file if isinstance(audio_file, str) else audio_file.name}")

    # Add an audio player
    st.audio(audio_file)

    # ✅ Transcribe the audio & detect speakers
    with st.spinner("Transcribing... This may take a few seconds."):
        try:
            with open(temp_audio_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            transcript = result.text

            # Simulate speaker detection (since Whisper API doesn't provide speaker detection)
            segments = result.segments
            doctor_speaking = True  # Start with doctor
            formatted_transcript = ""

            for segment in segments:
                speaker_label = "Doctor" if doctor_speaking else "Patient"
                # Clean up the text by removing "Doctor:" and "Patient:" from the beginning
                cleaned_text = segment.text
                cleaned_text = cleaned_text.replace("Doctor,", "Doctor:").replace("Patient,", "Patient:")
                cleaned_text = cleaned_text.strip()
                
                formatted_transcript += f"{cleaned_text} "
                doctor_speaking = not doctor_speaking
            
            # Make sure every instance of "Doctor:" and "Patient:" starts on a new line
            formatted_transcript = re.sub(r'([^\n])(Doctor:|Patient:)', r'\1\n\2', formatted_transcript)
            # Handle case where "Doctor:" or "Patient:" is at the start of the text
            formatted_transcript = re.sub(r'^(Doctor:|Patient:)', r'\1', formatted_transcript)
            # Remove any potential double newlines
            formatted_transcript = re.sub(r'\n\s*\n', '\n', formatted_transcript)

            st.subheader("📄 Transcribed Conversation (Speaker-Separated)")
            st.text_area("Doctor-Patient Conversation", formatted_transcript, height=200)

            # Only proceed with summary generation if we have a transcript
            if formatted_transcript:
                # ✅ Choose target language for report
                st.subheader("🌍 Choose Report Language")
                target_language = st.selectbox("Select language", ["English", "Chinese", "Spanish", "French"])

                # ✅ AI-Powered Doctor's Report Directly in Chosen Language
                st.subheader(f"📑 AI-Generated Patient Visit Report ({target_language})")

                try:
                    summary_prompt = f"""
                    From the following doctor-patient conversation, extract and summarize:
                    - The doctor's **diagnosis** for the patient.
                    - Any **medical advice, prescriptions, or lifestyle recommendations** the doctor gives.
                    - Present the summary in a clear and structured format like a **medical visit report**.

                    Here is the conversation:
                    {formatted_transcript}

                    Format the response as:
                    ---
                    **📝 Patient Visit Report**
                    **🩺 Diagnosis:** [Doctor's diagnosis]
                    **💊 Recommended Treatment:** [Prescriptions or medical treatment]
                    **🍏 Lifestyle & Health Tips:** [Doctor's advice to improve health]
                    ---
                    
                    The report should be in {target_language}.
                    """

                    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",  # Updated model name from gpt-4o-mini
                        messages=[
                            {"role": "system", "content": "You are a medical assistant summarizing doctor-patient conversations."},
                            {"role": "user", "content": summary_prompt}
                        ]
                    )
                    summary = response.choices[0].message.content

                    st.markdown(f"📄 **Patient Visit Report in {target_language}**\n")
                    st.markdown(summary)

                except Exception as e:
                    st.error(f"Summary generation error: {str(e)}")

        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            formatted_transcript = ""  # Reset on error

    # ✅ AI-Generated Key Medical Term Explanations in User's Language
    st.subheader(f"🔍 Key Medical Terms & Explanations ({target_language})")
    try:
        terms_prompt = f"""
        Extract and explain the **medical terms** from the following doctor's **diagnosis and recommendations**.
        Provide a simple explanation that a patient can understand.
        Highlight any conditions that need special attention.

        Here is the diagnosis and recommendations:
        {summary}

        The explanation should be in {target_language}.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical assistant helping patients understand medical terms."},
                {"role": "user", "content": terms_prompt}
            ]
        )
        medical_explanation = response.choices[0].message.content

        st.markdown(f"📑 **Medical Terms & Explanations in {target_language}**\n\n{medical_explanation}")

    except Exception as e:
        st.error(f"Medical term explanation error: {str(e)}")

    # ✅ Allow users to download the AI summary
    if summary:
        st.subheader("📥 Download Your Medical Summary")
        st.download_button(label="Download Report", data=summary, file_name="patient_visit_report.txt", mime="text/plain")
