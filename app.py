import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from unsloth import FastLanguageModel
import torch

# Load the model and tokenizer
model_path = "/home/roser97/MarketAI/lora_model"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=800,  # Adjust according to your needs
    load_in_4bit=True,
)

# Configure the model for inference
FastLanguageModel.for_inference(model)

def generate_marketing_content(instruction, input_context):
    inputs = tokenizer(
        [f"### Instruction:\\n{instruction}\\n### Input:\\n{input_context}\\n### Response:"],
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    output = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    st.set_page_config(page_title="Compass AI", layout="wide")
    st.title("Compass AI")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Generate Content", "Data"])

    if page == "Generate Content":
        # Content generation logic
        st.header("Generate Marketing Content")
        instruction = st.text_area("Enter your instruction")
        input_context = st.text_area("Enter the context")
        if st.button("Generate"):
            result = generate_marketing_content(instruction, input_context)
            st.write(result)
    elif page == "Data":
        # Data analysis logic
        st.header("Upload Data")
        file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
        if file is not None:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            st.dataframe(df)

if __name__ == "__main__":
    main()

