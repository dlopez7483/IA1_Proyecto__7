import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_movie_lines(file_path):
    """
    Carga las líneas desde el archivo movie_lines.txt.
    Devuelve un diccionario con los IDs de las líneas como claves y el texto como valores.
    """
    lines = {}
    with open(file_path, encoding='UTF-8', errors='ignore') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 3:  # Verificar que tiene los campos necesarios
                line_id = parts[0]
                text = parts[2]
                lines[line_id] = text
    return lines

def load_movie_conversations(file_path):
    """
    Carga las conversaciones desde el archivo movie_conversations.txt.
    Devuelve una lista de listas, donde cada sublista contiene los IDs de las líneas en una conversación.
    """
    conversations = []
    with open(file_path, encoding='UTF-8', errors='ignore') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 2:  # Verificar que tiene los campos necesarios
                line_ids = parts[1][1:-1].replace("'", "").split(", ")  # Convertir a lista
                conversations.append(line_ids)
    return conversations

def preprocess_data(lines, conversations):
    """
    Procesa los datos de líneas y conversaciones para generar pares de preguntas y respuestas.
    También prepara los datos para el entrenamiento (tokenización y padding).
    """
    questions = []
    answers = []

    # Extraer pares de preguntas y respuestas
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            question_id = conversation[i]
            answer_id = conversation[i + 1]
            if question_id in lines and answer_id in lines:
                questions.append(lines[question_id])
                answers.append(lines[answer_id])

    # Limpiar el texto
    def clean_text(text):
        return text.lower().strip()

    questions_clean = [clean_text(q) for q in questions]
    answers_clean = [clean_text(a) for a in answers]

    # Tokenización
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions_clean + answers_clean)

    # Convertir texto a secuencias numéricas
    questions_seq = tokenizer.texts_to_sequences(questions_clean)
    answers_seq = tokenizer.texts_to_sequences(answers_clean)

    # Padding para que todas las secuencias tengan la misma longitud
    max_len_input = max(len(seq) for seq in questions_seq)
    max_len_output = max(len(seq) for seq in answers_seq)

    questions_seq = pad_sequences(questions_seq, maxlen=max_len_input, padding='post')
    answers_seq = pad_sequences(answers_seq, maxlen=max_len_output, padding='post')

    # Crear el vocabulario
    vocab_size = len(tokenizer.word_index) + 1  # +1 para el token <PAD>

    return questions_seq, answers_seq, tokenizer, max_len_input, max_len_output, vocab_size



def retornar_datos():
    # Definir rutas de los archivos
    lines_path = "movie_lines.txt"
    conversations_path = "movie_conversations.txt"

    # Verificar que los archivos existan
    if not os.path.exists(lines_path) or not os.path.exists(conversations_path):
        print("Error: Verifica las rutas de los archivos de entrada.")
        exit(1)

    # Cargar datos
    print("Cargando datos...")
    lines = load_movie_lines(lines_path)
    conversations = load_movie_conversations(conversations_path)

    # Procesar datos
    print("Procesando datos...")
    questions_seq, answers_seq, tokenizer, max_len_input, max_len_output, vocab_size = preprocess_data(lines, conversations)

    # Mostrar estadísticas de los datos procesados
    print(f"Número de pares de preguntas y respuestas: {len(questions_seq)}")
    print(f"Longitud máxima de entrada: {max_len_input}")
    print(f"Longitud máxima de salida: {max_len_output}")
    print(f"Tamaño del vocabulario: {vocab_size}")
    return questions_seq, answers_seq, tokenizer, max_len_input, max_len_output, vocab_size



