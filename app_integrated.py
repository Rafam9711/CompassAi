
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from unsloth import FastLanguageModel
import torch

# Cargar el modelo y el tokenizador
model_path = "/home/roser97/MarketAI/lora_model"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=800,  # Ajusta según tus necesidades
    load_in_4bit=True,
)

# Configurar el modelo para inferencia
FastLanguageModel.for_inference(model)

def generate_marketing_content(instruction, input_context):
    inputs = tokenizer(
        [f"### Instruction:\n{instruction}\n### Input:\n{input_context}\n### Response:"],
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    output = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    st.set_page_config(page_title="Compass AI", layout="wide")
    st.title("Compass AI")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Home", "Campaign Creation", "Strategy", "Scheduling", "Analytics"])

    if page == "Home":
        show_home()
    elif page == "Campaign Creation":
        show_campaign_creation()
    elif page == "Strategy":
        show_strategy()
    elif page == "Scheduling":
        show_scheduling()
    elif page == "Analytics":
        show_analytics()

def show_home():
    st.header("Welcome to AI Marketing Campaign Agent")
    st.write("This tool helps you create, manage, and analyze your marketing campaigns using AI.")
    st.write("Use the sidebar to navigate through different features.")

def show_campaign_creation():
    st.header("Campaign Creation")
    
    # Brand Questionnaire
    st.subheader("Brand Questionnaire")
    brand_name = st.text_input("Brand Name")
    industry = st.selectbox("Industry", ["Technology", "Fashion", "Food & Beverage", "Other"])
    target_audience = st.text_area("Describe your target audience")
    campaign_objective = st.selectbox("Campaign Objective", ["Brand Awareness", "Lead Generation", "Sales", "Other"])

    # Content Generation
    st.subheader("Content Generation")
    content_type = st.selectbox("Content Type", ["Social Media Post", "Ad Copy", "Email"])
    content_prompt = st.text_area("Describe the content you want to generate")

    if st.button("Generate Content"):
        with st.spinner("Generating content..."):
            generated_content = generate_marketing_content(content_prompt, f"{brand_name}, {industry}, {target_audience}, {campaign_objective}")
        st.text_area("Generated Content", generated_content, height=200)

def show_strategy():
    st.header("Marketing Strategy")
    
    start_date = st.date_input("Campaign Start Date")
    duration = st.number_input("Campaign Duration (days)", min_value=1, value=30)
    
    if st.button("Generate Strategy"):
        with st.spinner("Generating strategy..."):
            strategy = generate_marketing_content("Generate a marketing strategy", f"Start Date: {start_date}, Duration: {duration} days")
        
        st.subheader("Generated Marketing Strategy")
        st.text(strategy)
        
        if st.button("Generate PDF Proposal"):
            st.write("PDF generation functionality to be implemented.")

def show_scheduling():
    st.header("Content Scheduling")
    
    platforms = st.multiselect("Select Platforms", ["Facebook", "Instagram", "Twitter"])
    post_content = st.text_area("Post Content")
    post_date = st.date_input("Post Date")
    post_time = st.time_input("Post Time")
    
    if st.button("Schedule Post"):
        scheduled_datetime = datetime.combine(post_date, post_time)
        for platform in platforms:
            st.success(f"Post scheduled for {platform} at {scheduled_datetime}")

def show_analytics():
    st.header("Campaign Analytics")
    st.write("This feature is under development. It will show campaign performance metrics and insights.")
    
if __name__ == "__main__":
    main()
