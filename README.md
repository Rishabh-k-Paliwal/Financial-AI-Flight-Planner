
## Financial AI System

An intelligent, multi-agent financial command center powered by Google's **Gemini LLM, LangChain, LangGraph, Streamlit, and ChromaDB RAG**. 

It takes your personal financial context and unleashes 4 targeted AI specialist agents to build a highly personalized "Financial Flight Plan".

---

## 🛠 Features
- **Multi-Agent Workflow (`agents.py`)**: 
  -  **Savings Architect**: Calculates emergency fund gaps.
  -  **Debt Shredder**: Targets high-interest debt with Avalanche/Snowball.
  -  **Insurance Expert**: Evaluates coverage and calculates precise risks.
  -  **Investment Scout**: Aligns portfolio with the 100-Age Rule.
  -  **Final Planner**: Merges agent advice into a prioritized, budgeted timeline.
- **Dynamic Local RAG (`rag_setup.py`)**: Vector retrieval of PDFs and Text files locally via ChromaDB so agents stay highly knowledgeable.
- **Terminal Rate-Limit Handler**: Includes an automatic suspension wrapper that intercepts Gemini Free-Tier `429 RESOURCE_EXHAUSTED` quotas and waits dynamically instead of crashing.
- **Streamlit UI (`app.py`)**: A single-page dashboard with spider-charts, dynamic health scoring (`scoring_engine.py`), and a conversational memory interface.

---

