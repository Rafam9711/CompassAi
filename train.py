from unsloth import FastLanguageModel
import torch

# Model configurations
max_seq_length = 800  # Ajusta según tus necesidades
dtype = None  # None para detección automática. Usa Float16 o Bfloat16 si sabes cuál usar.
load_in_4bit = True  # Habilita la cuantificación de 4 bits para reducir el uso de memoria

# Load the pre-trained model with optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",  # Nombre del modelo preentrenado
    max_seq_length=max_seq_length,  # Longitud máxima de secuencia
    dtype=dtype,  # Tipo de dato de la GPU (detección automática si es None)
    load_in_4bit=load_in_4bit,  # Cuantificación de 4 bits para optimizar memoria
)

print("Modelo cargado exitosamente.")

# Applying LoRA adapters to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Adaptadores LoRA aplicados correctamente.")

from datasets import load_dataset


# Define the format template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token

# Function to format the dataset
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["response"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Load and prepare the dataset for fine-tuning
dataset = load_dataset('json', data_files='/content/marketing_social_media_dataset_v1.json', split='train')

# Apply the format to the entire dataset using the map function
dataset = dataset.map(formatting_prompts_func, batched=True)

print("Dataset cargado y formateado correctamente.")

from trl import SFTTrainer
from transformers import TrainingArguments

# Training configuration using SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,  # Deshabilitar fp16
        bf16=False,   # Habilitar bf16, recomendado para GPUs Ampere
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Start the training process
trainer_stats = trainer.train()

print("Entrenamiento completado con éxito.")

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from IPython.display import display, Markdown
import json

# Configure the model for inference
FastLanguageModel.for_inference(model)

# Generate text based on a given instruction
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Best marketing post for sneaker company",  # Instrucción para el modelo
            "",  # Entrada adicional (en este caso, ninguna)
            "",  # Respuesta esperada (en este caso, ninguna)
        )
    ], return_tensors="pt").to("cuda")

# Generate output without using TextStreamer
output = model.generate(**inputs)

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Function to parse the output and convert it into a dictionary
def parse_output_to_dict(output_text):
    result = {}
    current_section = None
    lines = output_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('###'):
            current_section = line.strip('# ').lower().replace(' ', '_')
            result[current_section] = {}
        elif ':' in line:
            key, value = line.split(':', 1)
            key = key.lower().replace(' ', '_').strip()
            result[current_section][key] = value.strip()
        elif line and current_section:
            if 'content' not in result[current_section]:
                result[current_section]['content'] = []
            result[current_section]['content'].append(line)

    return result

# Parse the generated output into a dictionary
parsed_output = parse_output_to_dict(output_text)

# Display the parsed output as a formatted JSON
display(Markdown("## Parsed JSON Output\n\n```json\n" + json.dumps(parsed_output, indent=2) + "\n```"))

# Guardar el modelo ajustado y el tokenizador en un directorio
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("Modelo y tokenizador guardados correctamente en 'lora_model'.")

from unsloth import FastLanguageModel

# Save the fine-tuned model and tokenizer in a directory
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Configure the model for inference
FastLanguageModel.for_inference(model)

print("Modelo y tokenizador recargados correctamente desde 'lora_model'.")

# Generate text based on a new prompt
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Create a marketing campaign to promote the chocolate bar",  # Instrucción
            "Company: Cadbury, target audience: adults/boomers",  # Información de entrada adicional
            "",  # Respuesta esperada (en este caso, ninguna)
        )
    ], return_tensors="pt").to("cuda")

# Generate output (if you're not using TextStreamer, you can remove the corresponding line)
output = model.generate(**inputs, max_new_tokens=128)

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Display the generated output
print("Salida Generada por el Modelo:")
print(output_text)

from google.colab import files
import shutil

# Create a zip file of the model
shutil.make_archive("lora_model", 'zip', "lora_model")

# Download the zip file
files.download("lora_model.zip")

