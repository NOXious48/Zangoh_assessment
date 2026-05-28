# Audio Customer Support Agent - Video Demo Script

**Target Length:** ~5 minutes
**Preparation before recording:** 
- Have your API server running in one terminal (`python -m src.api.server`).
- Have Streamlit running in another terminal (`streamlit run streamlit_app.py`).
- Have your code editor open (showing `pipeline.py` or `server.py`).
- Open your browser to the Streamlit UI (`http://localhost:8501`).

---

## Part 1: Introduction & Architecture (0:00 - 0:45)
**Visual:** Show an Architecture Diagram (if you have one) or your Code Editor showing the project structure.

**Speaker:**
> "Hi, welcome to my demonstration of the Audio Customer Support Agent.
> 
> My architecture is built on a modular pipeline orchestrated by FastAPI. It consists of three core components:
> First, **Speech-to-Text (STT)**, where I utilized a local Whisper model to transcribe user audio.
> Next, the **LLM Agent**, powered by a LangChain ReAct framework integrated with ChromaDB. This acts as our Retrieval-Augmented Generation (RAG) system, pulling from 16 predefined customer support documents to ground the AI's answers.
> Finally, **Text-to-Speech (TTS)** synthesizes the agent's text response back into audio.
> All of this is tied together seamlessly using Python's `asyncio` for non-blocking performance."

---

## Part 2: Mid-Session Requirement Explanation (0:45 - 1:45)
**Visual:** Switch to your code editor. Show `src/pipeline.py` (specifically `process_audio_with_transcript`) and then `src/api/server.py` (showing the Pydantic models).

**Speaker:**
> "For the mid-session requirement, I was tasked with enhancing the pipeline to return both audio and textual transcripts, along with tracking processing times.
>
> To implement this, I updated my `pipeline.py` by adding a new `process_audio_with_transcript` method. This method captures the exact `user_input` from STT, the `agent_response` from the LLM, and calculates the total `processing_time_ms` using Python's `time` module.
>
> I then updated `server.py` by creating new Pydantic models like `EnhancedAudioResponse` and `TranscriptData`. I modified the `/chat/audio` endpoint so that instead of returning raw audio bytes, it encodes the audio in Base64 and returns a structured JSON payload containing the audio string, the transcript dictionary, and the processing time. Finally, I updated the Streamlit UI to parse this JSON and display the transcript and processing times side-by-side with the audio player."

---

## Part 3: Live System Demonstration (1:45 - 4:45)
**Visual:** Switch to the Streamlit UI in your browser. Navigate to the **"Audio Chat Interface"** tab. 

**Speaker:**
> "Now, let's see the system in action with a few test queries using our Streamlit interface. Because I updated the UI for the mid-session requirement, you'll see a dual-layout with audio controls on the left and the newly added live transcript on the right."

*(Note: During this part, click "Record Audio" and speak the prompts clearly. Wait for the processing to finish and the audio response to play back).*

### Test Query 1: Basic Return Policy
**Action:** Record yourself saying: *"What is your return policy?"*
**Speaker:**
> "For our first query, I'll ask a basic policy question to test the RAG retrieval.
> *(Wait for processing...)*
> As you can see on the right, the transcript successfully captured my speech, and the agent accurately retrieved our 30-day return policy. The total processing time is displayed below."

### Test Query 2: Specific Constraints
**Action:** Record yourself saying: *"Can I return opened cosmetics?"*
**Speaker:**
> "Next, let's test a more specific constraint from our knowledge base regarding non-returnable items.
> *(Wait for processing...)*
> Perfect. The agent correctly retrieved the health and safety regulations policy stating that opened cosmetics cannot be returned."

### Test Query 3: Shipping Times
**Action:** Record yourself saying: *"How long does standard shipping take?"*
**Speaker:**
> "For our third query, let's switch contexts to shipping and logistics.
> *(Wait for processing...)*
> The system correctly queried the shipping documents and informed us that standard shipping takes 5-7 business days."

### Test Query 4: Tech Support / Warranty
**Action:** Record yourself saying: *"My electronics are broken, do I get free technical support?"*
**Speaker:**
> "Finally, let's ask a multi-part question regarding electronics and technical support.
> *(Wait for processing...)*
> The agent successfully cross-referenced the technical support documents, verifying that we do offer free technical support for electronic products."

---

## Part 4: Conclusion (4:45 - 5:00)
**Visual:** Switch to the "Health Monitor" tab in Streamlit to show all components are healthy and running.

**Speaker:**
> "To wrap up, here is the Health Monitor showing that all endpoints and components remained stable throughout the interactions. The complete audio-in to audio-out pipeline, complete with transcript generation, works end-to-end.
>
> Thank you for watching!"
