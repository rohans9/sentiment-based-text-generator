import streamlit as st
import torch
from transformers import pipeline
import time # Just here in case we want to add an exponential backoff later for API calls
import openai

# --- Configuration and Setup ---

# The model we use for checking the sentiment of the user's prompt
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" 
# The workhorse model for generating high-quality, relevant text
GPT_MODEL_NAME = "gpt-4o-mini" 

# We create prompt prefixes here to force the GPT model into the desired tone.
SENTIMENT_PROMPT_MAPPING = {
    "POSITIVE": "Write a highly enthusiastic, uplifting, and optimistic paragraph about the following topic: ",
    "NEGATIVE": "Write a deeply pessimistic, skeptical, and critical paragraph about the following topic: ",
    "NEUTRAL": "Write a balanced, purely factual, and objective paragraph about the following topic: "
}

@st.cache_resource
def load_sentiment_model():
    """
    Grabs the local sentiment analysis model and caches it so it only loads once.
    """
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=SENTIMENT_MODEL_NAME, 
            device=0 if torch.cuda.is_available() else -1
        )
        return sentiment_analyzer
    except Exception as e:
        st.error(f"Hmm, ran into an error loading the local sentiment model. Make sure 'transformers' and 'torch' are installed correctly. Error: {e}")
        return None

sentiment_analyzer = load_sentiment_model()

# --- The Brains: Handling the OpenAI API Call ---

def generate_text_with_openai(prompt: str, max_tokens: int, api_key: str):
    """
    Sends the fully-engineered prompt off to the OpenAI API for text generation.
    """

    openai.api_key = api_key
    
    response = openai.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[
            # System message acts as the instruction for the model's persona
            {"role": "system", "content": "You are a professional content writer who strictly adheres to the requested tone and sentiment."},
            # User message contains the actual topic and sentiment prefix
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.9,
    )

    return response.choices[0].message.content.strip()
    

def generate_sentiment_aligned_text(prompt, target_sentiment, max_tokens, api_key):
    """
    This orchestrator function checks the sentiment (auto or manual) and
    then calls the OpenAI generator with the appropriate stylistic prompt.
    """
    try:

        if target_sentiment == "Auto-Detect":
            if sentiment_analyzer is None:
                st.error("Can't run auto-detect, the sentiment model didn't load properly.")
                return None, None

            # Get the sentiment score from the local model
            analysis_result = sentiment_analyzer(prompt)[0]
            detected_sentiment = analysis_result['label']
            st.info(f"The local model detected the sentiment as: **{detected_sentiment}** (Confidence Score: {analysis_result['score']:.4f})")
        else:
            # Use the user's manual override choice
            detected_sentiment = target_sentiment.upper()
            st.info(f"We are using the manual override: **{detected_sentiment}**")

        # Step 2: Build the complete prompt with the style prefix
        style_prefix = SENTIMENT_PROMPT_MAPPING.get(detected_sentiment, SENTIMENT_PROMPT_MAPPING['NEUTRAL'])
        full_generation_prompt = f"{style_prefix}{prompt}"
        
        # Step 3: Call the paid API service to generate the final text
        generated_text = generate_text_with_openai(
            full_generation_prompt,
            max_tokens=max_tokens,
            api_key=api_key
        )

        # We trust the OpenAI response is the final text we want.
        return generated_text, detected_sentiment

    # Handle common API errors
    except openai.AuthenticationError as e:
        st.error("Ouch, Authentication Error! The API Key you entered seems invalid. Please double-check it in the sidebar.")
        return None, None
    except openai.APIError as e:
        st.error(f"Something went wrong with the OpenAI request: {e.message}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error popped up during the generation process: {e}")
        return None, None

# --- Streamlit Interface (How the User Sees It) ---

st.set_page_config(page_title="AI Sentiment Text Generator (OpenAI)", layout="centered")

st.title("AI Text Generator: Sentiment-Driven Content")
st.markdown("This app combines two powerful tools: a fast, local **Sentiment Model** determines the tone, while the **OpenAI API** (gpt-4o-mini) generates the content.")

# Settings go into the sidebar to keep the main view clean
with st.sidebar:
    st.header("OpenAI Configuration")
    
    # We ask the user for their key since we can't store secrets here
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="This key is essential for the text generation model to work (it uses your credits)."
    )
    
    st.markdown("---")
    st.header("Generation Settings")
    
    sentiment_options = ["Auto-Detect", "Positive", "Negative", "Neutral"]
    selected_sentiment = st.selectbox(
        "Sentiment Override:",
        sentiment_options,
        index=0,
        help="Force the AI to write with a specific emotion, or let it guess the emotion from your topic."
    )
    
    max_tokens = st.slider(
        "Max Output Length (tokens):",
        min_value=50,
        max_value=500, # Increased max tokens for better GPT output
        value=200,
        step=10,
        help="Controls the approximate size of the generated paragraph or essay."
    )
    
    st.markdown("---")
    st.caption(f"Sentiment Model: `{SENTIMENT_MODEL_NAME}`")
    st.caption(f"Generation Model: `{GPT_MODEL_NAME}` (OpenAI)")


# The main input box for the user's topic
prompt_input = st.text_area(
    "Enter your prompt or topic here:",
    "Self-driving cars are coming soon and will change everything.",
    height=100
)

# The button that kicks off the whole process
if st.button("Generate Sentiment-Aligned Text", type="primary"):
    if not prompt_input or len(prompt_input.strip()) < 5:
        st.warning("Please give the AI something meaningful to write about (at least 5 characters).")
    elif not api_key_input:
        st.error("Whoops! You need to put your OpenAI API Key in the sidebar first.")
    elif sentiment_analyzer is None:
        st.error("The local sentiment model failed to load.")
    else:
        st.subheader("Results")
        with st.spinner("Analyzing sentiment locally and then asking the OpenAI cloud to generate text..."):
            generated_text, final_sentiment = generate_sentiment_aligned_text(
                prompt_input, 
                selected_sentiment, 
                max_tokens,
                api_key_input
            )

        if generated_text:
            st.markdown(f"#### Generated Text (Style: {final_sentiment})")
            st.success(generated_text)
            st.markdown("---")
            st.markdown(f"**Full Prompt Used for Generation:** `{SENTIMENT_PROMPT_MAPPING.get(final_sentiment, '')}{prompt_input}`")
