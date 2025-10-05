# Sentiment-Controlled Text Generation

This Streamlit application provides a robust tool for generating text output that is precisely aligned with a specific emotional tone (sentiment). The project uses a hybrid architecture to maximize both performance and output quality.

## Key Features and Technical Approach

The application is structured into a two-stage pipeline, leveraging the strengths of both local open-source models and a powerful cloud API.

### 1. Local Sentiment Processing
- **Model:** distilbert-base-uncased-finetuned-sst-2-english
- **Role:** Rapidly analyzes the user's input prompt and assigns a sentiment label (POSITIVE, NEGATIVE, or NEUTRAL).
- **Advantage:** This computationally inexpensive step is executed locally using the Hugging Face pipeline, ensuring minimal latency and zero external API costs for the sentiment detection phase.

### 2. Cloud Text Generation
- **Model:** gpt-4o-mini (OpenAI)
- **Role:** Generates the final, detailed paragraph or short essay based on the required style.
- **Input Strategy:** We enforce the desired tone using a robust instruction injection method: The user's input is combined with a precise System Prompt (e.g., "You are a professional content writer who strictly adheres to the requested tone") and a Sentiment Prefix (e.g., "Write a deeply pessimistic paragraph about...") before being sent to the LLM.
- **API:** Communication relies on the official openai Python library, using the high-performance gpt-4o-mini endpoint.

## Project Development Rationale (Reflections)

During development, we faced a critical decision regarding the choice of the primary text generation model.

- **The Initial Hurdle (Open-Source Models):**  
  We initially experimented with several highly-rated open-source Large Language Models (LLMs) served via common inference platforms. Specifically, models such as Qwen/Qwen3-0.6B were tested. The main challenge was poor contextual relevance.

  While these open-source models successfully adopted the requested tone (e.g., pessimism), the resulting text was often not relevant to the user's specific input prompt. They struggled to maintain focus on the core topic, leading to generic or off-topic output that failed to meet the functional requirements of the application.

- **The Final Decision (GPT-4o mini):**  
  To ensure consistent, high-quality, and reliable adherence to the nuanced sentiment instructions, we transitioned to the OpenAI GPT-4o mini model. This proprietary model demonstrates superior capability in honoring complex stylistic and behavioral instructions set via the system prompt, guaranteeing that the generated text truly aligns with the required emotional state. This decision prioritizes functional quality and reliability over the use of a completely open-source solution for the generation step.

## Setup and Execution

To run this application, ensure you have a standard Python environment (Python 3.8+) configured.

### 1. Installation
Install the necessary Python dependencies via pip:

```bash
pip install streamlit torch transformers openai
```

### 2. API Key Configuration (Required for Generation)
The application requires an OpenAI API Key.
- **Account Requirement:** You must have an OpenAI account with a valid API key and active billing (even if using free trial credits).
- **Key Input:** When running the Streamlit application, enter your secret key into the dedicated input field in the sidebar.

### 3. Running the Application
1. Save the project code as `streamlit_app.py`.
2. Open your terminal or command prompt in the directory containing the file.
3. Execute the application:

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser.