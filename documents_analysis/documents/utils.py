import pytesseract
from PIL import Image
from docx import Document as DocxDocument
from io import BytesIO
from transformers import pipeline
from PyPDF2 import PdfReader
from pathlib import Path
from deeppavlov import configs, build_model

# Загрузка модели NER для русского языка
ner_model = build_model(configs.ner.ner_rus_bert, download=True)



def handle_uploaded_file(f):
    # Открываем PDF-файл из байтового потока
    pdf_reader = PdfReader(BytesIO(f.read()))
    text = ""

    # Перебираем страницы и извлекаем текст
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Сначала пробуем извлечь текст, или добавляем пустую строку, если текста нет

    return text


def ocr_image(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text


# Инициализация модели для Named Entity Recognition (NER)
nlp_ner = pipeline("ner", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")


def extract_summary_with_ner(text):
    # Применение модели для извлечения именованных сущностей
    ner_results = ner_model([text])[0]
    entities = {}
    for entity, tag in zip(*ner_results):
        if tag != 'O':  # 'O' означает отсутствие сущности
            tag = tag[2:]  # Убираем префикс 'B-' или 'I-'
            if tag not in entities:
                entities[tag] = []
            entities[tag].append(entity)
    return entities


def save_file(text, filename):
    doc = DocxDocument()

    # Сохранение .docx в памяти
    doc.add_paragraph(text)
    file_name = Path(filename).stem
    doc.save(f'{file_name}.docx')
