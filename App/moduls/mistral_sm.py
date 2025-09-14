from mistralai import Mistral
from dotenv import load_dotenv
import os

def prompt_1(text, result1, result2):
    user_prompt = f"""
    You will analyze the following news article and explain its potential reliability and political bias for general readers.

    NEWS TEXT:
    {text}

    MODEL OUTPUT:

    - Fake news probability:
        - Fake: {result2['probabilities']['fake']:.2f}
        - Real: {result2['probabilities']['real']:.2f}

    - Political bias probabilities:
        - Left: {result1['probabilities']['left']:.2f}
        - Leaning-left: {result1['probabilities']['leaning-left']:.2f}
        - Center: {result1['probabilities']['center']:.2f}
        - Leaning-right: {result1['probabilities']['leaning-right']:.2f}
        - Right: {result1['probabilities']['right']:.2f}

    TASK:
    Write three separate and concise paragraphs that fulfill the following roles.
    Each one MUST start with the following heading (exactly as written, including the colons), followed by its content:

    1. **Interpretation paragraph**: Clearly summarize the model results using accessible language. Explain whether the article is more likely to be fake or real, and what political leaning it is most associated with. Stay neutral and factual.

    2. **Justification paragraph**: Elaborate on why the model might have assigned those values. Go beyond tone and vocabulary: consider who the main actors or institutions mentioned are, what political or social issue is being discussed, whether the narrative aligns with typical ideological frames, and how the argument is constructed. If relevant, comment on the emotional charge of the language, presence of sensationalism, one-sided arguments, or omission of context. Try to infer the underlying perspective or agenda conveyed by the article, based on content and form.

    3. **Risk analysis paragraph**: Speak directly to the reader. If the article appears biased or unreliable, explain why it’s important to question its content before accepting it as truth. Caution the reader about how such content might shape their perception, reinforce ideological biases, or mislead them about complex issues. Encourage critical thinking and awareness of how media can influence public opinion and discourse.

    Write clearly and neutrally, aiming for a general audience.
    """
    return user_prompt


def prompt_2(text):
    summary_prompt = f"""
    You will read the following news article and write a clear and concise summary of its main points.

    TEXT:
    {text}

    TASK:
    Summarize the article in one paragraph (6–8 lines), covering the main topic, key facts, people or institutions involved, and any relevant outcomes or context. Avoid personal opinions or speculation. Write in neutral and accessible language, suitable for a general audience.

    Make sure the summary is self-contained and understandable without needing to read the full article.
    """
    return summary_prompt


def call_mistral_api(prompt, role):

    load_dotenv()
    API_KEY = os.getenv("MISTRAL_API_KEY")
    MODEL_NAME = 'mistral-large-latest'

    client = Mistral(api_key=API_KEY)

    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def EXPLAIN(text,result1,result2):
    prompt = prompt_1(text, result1, result2)
    role = "You are a helpful assistant that analyzes news articles to explain their reliability and bias to the general public."
    return call_mistral_api(prompt, role)


def SUMMARY(text):
    prompt = prompt_2(text)
    role = "You are a helpful assistant that summarizes news articles clearly and concisely for a general audience."
    return call_mistral_api(prompt, role)
