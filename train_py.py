from unsloth import FastLanguageModel
import torch

# Configuraciones del modelo
max_seq_length = 800  # Ajusta según tus necesidades
dtype = None  # None para detección automática. Usa Float16 o Bfloat16 si sabes cuál usar.
load_in_4bit = True  # Habilita la cuantificación de 4 bits para reducir el uso de memoria

# Carga el modelo preentrenado con las optimizaciones
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",  # Nombre del modelo preentrenado
    max_seq_length=max_seq_length,  # Longitud máxima de secuencia
    dtype=dtype,  # Tipo de dato de la GPU (detección automática si es None)
    load_in_4bit=load_in_4bit,  # Cuantificación de 4 bits para optimizar memoria
)

print("Modelo cargado exitosamente.")

# Aplicación de adaptadores LoRA al modelo
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

# Paso 8: Dar formato al conjunto de datos para el entrenamiento
# Definir la plantilla de formato
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token

# Función para formatear el conjunto de datos
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["response"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Cargar y preparar el dataset para el ajuste fino
dataset = load_dataset('json', data_files='/content/marketing_social_media_dataset_v1.json', split='train')

# Aplicar el formato a todo el dataset usando la función map
dataset = dataset.map(formatting_prompts_func, batched=True)

print("Dataset cargado y formateado correctamente.")

from trl import SFTTrainer
from transformers import TrainingArguments

# Configuración del entrenamiento utilizando SFTTrainer
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

# Iniciar el proceso de entrenamiento
trainer_stats = trainer.train()

print("Entrenamiento completado con éxito.")

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from IPython.display import display, Markdown
import json

# Configurar el modelo para la inferencia
FastLanguageModel.for_inference(model)

# Generar texto basado en una instrucción dada
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Best marketing post for sneaker company",  # Instrucción para el modelo
            "",  # Entrada adicional (en este caso, ninguna)
            "",  # Respuesta esperada (en este caso, ninguna)
        )
    ], return_tensors="pt").to("cuda")

# Generar la salida sin el uso de TextStreamer
output = model.generate(**inputs)

# Decodificar la salida
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Función para analizar la salida y convertirla en un diccionario
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

# Analizar la salida generada en un diccionario
parsed_output = parse_output_to_dict(output_text)

# Mostrar la salida analizada como un JSON formateado
display(Markdown("## Parsed JSON Output\n\n```json\n" + json.dumps(parsed_output, indent=2) + "\n```"))

# Guardar el modelo ajustado y el tokenizador en un directorio
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("Modelo y tokenizador guardados correctamente en 'lora_model'.")

from unsloth import FastLanguageModel

# Recargar el modelo y el tokenizador desde el directorio guardado
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Configurar el modelo para inferencia
FastLanguageModel.for_inference(model)

print("Modelo y tokenizador recargados correctamente desde 'lora_model'.")

# Generar texto basado en una nueva indicación
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Create a marketing campaign to promote the chocolate bar",  # Instrucción
            "Company: Cadbury, target audience: adults/boomers",  # Información de entrada adicional
            "",  # Respuesta esperada (en este caso, ninguna)
        )
    ], return_tensors="pt").to("cuda")

# Generar salida (si no usas TextStreamer, puedes quitar la línea correspondiente)
output = model.generate(**inputs, max_new_tokens=128)

# Decodificar la salida
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Mostrar la salida generada
print("Salida Generada por el Modelo:")
print(output_text)

from google.colab import files
import shutil

# Crear un archivo zip del modelo
shutil.make_archive("lora_model", 'zip', "lora_model")

# Descargar el archivo zip
files.download("lora_model.zip")

