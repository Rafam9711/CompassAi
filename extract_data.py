import json
from typing import List, Dict
import time
import re
from ratelimit import limits, sleep_and_retry
from collections import Counter

from openai import OpenAI

# Initialize the AI/ML API client with your API key and base URL
client = OpenAI(
    api_key="7b001294dc5e435bbb7d7b",
    base_url="https://api.aimlapi.com",
)

def rate_limited_api_call(messages):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=messages,
        temperature=0.7,
    )
    return response

def get_model_responses(messages: List[Dict[str, str]], max_retries: int = 3, timeout: int = 60) -> str:
    for attempt in range(max_retries):
        try:
            response = rate_limited_api_call(messages)
            return Markdown(response.choices[0].message.content.strip())
        except Exception as e:
            console.print(f"[bold red] Error in the model response (Attempt {attempt + 1} / {max_retries}): {e} [/bold red]")
            if attempt < max_retries - 1:
                time.sleep(5)
    raise Exception("Failed to get the response after repeated errors")

def generate_custom_marketing_samples(num_samples: int = 5) -> List[str]:
    system_message = f"""You are a specialized marketing AI designed to support the creation of comprehensive and innovative marketing strategies. You will generate {num_samples} marketing scenarios tailored specifically to integrate into the various stages of a marketing campaign workflow.

    For each sample, provide:

    1. Instruction: A specific, challenging marketing task that requires creativity and expertise.
    2. Input: Detailed context, including company info, target audience, constraints, goals, and the stage of the workflow where it will be applied (e.g., strategy, image generation, copywriting, post scheduling).
    3. Response: A highly creative and actionable solution that not only addresses the instruction and input but is also ready for integration into the corresponding workflow stage.

    Guidelines:
    - Cover a wide range of industries and marketing channels.
    - Incorporate modern trends like AI, sustainability, personalization, and social causes.
    - Consider various audience segments and challenging scenarios.
    - Ensure each scenario is adaptable to generate a PDF strategy, produce compelling images, write engaging copy, or schedule posts effectively.

    Format each sample as follows:
    ### Instruction:
    [Concise, specific marketing task]

    ### Input:
    [Detailed context, including workflow stage]

    ### Response:
    [Creative, comprehensive solution]

    Separate each sample with ---

    Remember: The generated content should be versatile and align seamlessly with our app's workflow, from initial strategy creation to final content adaptation and scheduling."""

    user_message = f"Generate {num_samples} cutting-edge, versatile marketing scenarios that align with our AI-powered marketing workflow. Use the specified format, separating each sample with ---."

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = get_model_responses(messages)
    return [sample.strip() for sample in response.split("---") if sample.strip()]

def quality_check(sample: dict) -> bool:
    min_length = 50
    max_repetition = 0.3

    for key in ["instruction", "input", "response"]:
        text = sample[key]
        if len(text) < min_length:
            return False

        words = re.findall(r"\w+", text.lower())
        unique_words = set(words)
        if len(unique_words) / len(words) < (1 - max_repetition):
            return False

    return True

from collections import Counter
from typing import List

# We update the channel keywords to the specific ones we need
industry_keywords = ["tech", "fashion", "food", "finance", "entertainment", "healthcare", "education", "travel"]
channel_keywords = ["facebook", "twitter", "instagram"]
objective_keywords = ["brand awareness", "lead generation", "customer retention", "product launch", "crisis management"]

# Function to track diversity in the dataset
def track_diversity(dataset: List[dict]) -> None:
    industries = Counter()
    channels = Counter()
    objectives = Counter()

    for sample in dataset:
# We convert the text of the samples to lowercase and analyze its content
        text = " ".join([sample["instruction"], sample["input"], sample["response"]]).lower()

# We update the counters based on the presence of keywords
        industries.update(word for word in industry_keywords if word in text)
        channels.update(word for word in channel_keywords if word in text)
        objectives.update(word for word in objective_keywords if word in text)

    # We print the diversity of the dataset
    print("Diversidad del Dataset:")
    print(f"Industrias: {dict(industries)}")
    print(f"Canales: {dict(channels)}")
    print(f"Objetivos: {dict(objectives)}")

def create_finetuning_dataset(target_samples: int, output_file: str):
    console.print(
        Panel(
            f"Creating fine-tuning dataset for [bold]Marketing & Social Media Content[/bold]",
            expand=False,
        )
    )

    dataset = []
    samples_per_call = 50  # Reduced to respect token limits
    calls_made = 0
    max_calls = 200  # Increased to allow for more total samples

    with console.status("[bold green]Generating samples...") as status:
        while len(dataset) < target_samples and calls_made < max_calls:
            calls_made += 1
            status.update(
                f"API call {calls_made} (Dataset size: {len(dataset)}/{target_samples})"
            )

            samples = generate_multiple_marketing_samples(samples_per_call)

            for sample in samples:
                if len(dataset) >= target_samples:
                    break

                parts = sample.split("###")
                if len(parts) == 4:
                    instruction = parts[1].replace("Instruction:", "").strip()
                    input_text = parts[2].replace("Input:", "").strip()
                    response = parts[3].replace("Response:", "").strip()

                    sample_dict = {
                        "instruction": instruction,
                        "input": input_text,
                        "response": response,
                    }

                    if quality_check(sample_dict):
                        dataset.append(sample_dict)
                    else:
                        console.print("[yellow]Skipped low-quality sample")
                else:
                    console.print(
                        f"[yellow]Skipped malformed sample: {sample[:100]}..."
                    )

            # Save progress after each API call
            with open(output_file, "w") as f:
                json.dump(dataset, f, indent=2)

    console.print(
        f"[bold green]Dataset creation complete! Total samples: {len(dataset)}"
    )
    track_diversity(dataset)

create_finetuning_dataset(
    target_samples=1000,
    output_file="marketing_social_media_dataset_v1.json",
)
