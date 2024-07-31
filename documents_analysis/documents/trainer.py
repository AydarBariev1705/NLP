from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch

# Токенайзер и модель
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=8)

# Текст и метки
text = ("Наименование заказчика - ПАО «Ростелеком»."
        " Проект - Создание СЗПДн."
        " Адрес - г.Москва, Ярославская ул., д.13А."
        " Срок выполнения - 40 календарных дней."
        " Работы включают: Предпроектное обследование, разработка ТЗ, тех-рабочее проектирование. "
        "Требования: Защита данных на всех уровнях. "
        "Объект защиты: Сетевые устройства и серверы.")
labels = [
    0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 0, 3, 3, 3, 0, 0, 0,
    4, 4, 4, 4, 0, 0, 0, 5, 0, 5, 5, 5, 0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 6, 6, 6, 6, 6, 6, 0, 0,
    7, 7, 7, 7, 0, 7, 7,
]

# Токенизация текста
encodings = tokenizer(text, return_offsets_mapping=True)
encoded_labels = [labels[i] for i, offset in enumerate(encodings.offset_mapping) if offset[0] != offset[1]]
text2 = """
Наименование заказчика: ПАО «Салехарднефтегаз».
Наименование проекта: 
Проектирование системы защиты информации для обеспечения непрерывности технологических процессов добычи и первичной переработки нефти и газа.
Адрес(-а) расположения защищаемых объектов заказчика:
РФ, ЯНАО, г. Салехард, ул. Нефтегазовая, корп. 5.
Сроки выполнения проекта / этапов проекта:
Старт проекта - 01.06.2024 г,
Завершение проекта – 30.12.2024 г.
этап 1 – 01.06.2024 – 31.07.2024
этап 2 – 01.08.2024 – 30.09.2024
этап 3 – 01.10.2024 – 30.12.2024.
Перечень выполняемых работ:
•Аудит (обследование) объектов защиты
•Разработка моделей угроз безопасности информации
•Разработка регламентов и организационно-распорядительной документации
•Разработка основных технических решений (ОТР)
•Разработка рабочей документации
•Разработка программы и методики испытаний.
Перечень требований по функциям проектируемой системы защиты информации:
•защита от несанкционированного доступа к ресурсам объекта защиты, включая идентификацию, аутентификацию, управление доступом;
•обеспечение антивирусной защиты;
•обеспечение межсетевого экранирования и защиты от сетевых атак;
•обнаружение вторжений
•криптографическая защита информации, в том числе информации, передаваемой между площадками объекта защиты и информации, передаваемой между системой и пользователями по протоколу HTTPS.
•анализ защищенности и выявление уязвимостей объекта защиты
Информация о объекте(-ах) защиты:
Общее количество объектов защиты(систем) – 12
Тип: объекты КИИ - 8, ИСПДн - 4
Состав оборудования и ПО:
•серверы – есть, 34 шт.,
•АРМ – есть, 67 шт.,
•сетевое оборудование – есть, 23 шт.,
•операционные системы – есть, MS Windows 7, MS Windows Server 2012, Ubuntu,
•виртуализация – есть, VMware ESXi 6
"""

labels2 = [
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0,
    0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0,
    5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5,
    5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 0, 0, 6, 6, 6, 6,
    6, 6, 0, 0, 6, 6, 6, 6, 6, 0, 0, 6, 6, 6, 6, 0, 0, 6, 6, 6, 6, 0, 0, 6, 6, 6, 6, 0, 0, 6, 6, 6, 6,
    0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 7, 7, 7, 7, 0, 0, 7, 7, 7, 7, 0, 0, 0, 7, 7, 7, 0, 0, 0, 7,
    7, 7, 7, 0, 0, 0, 7, 7, 7, 7, 0, 0, 7, 7, 7, 7, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0
]

# Токенизация нового текста
new_encodings2 = tokenizer(text2, return_offsets_mapping=True, padding=True, truncation=True)
new_encoded_labels2 = [labels2[i] for i, offset in enumerate(new_encodings2.offset_mapping) if offset[0] != offset[1]]

# Объединение данных
combined_texts = [text, text2]
combined_labels = [encoded_labels, new_encoded_labels2]


# Подготовка токенов и меток для всех текстов
all_encodings = tokenizer(combined_texts, truncation=True, padding=True, return_tensors="pt")
all_labels = []

# Объединение меток
for labels in combined_labels:
    all_labels.append(labels)

# Преобразование меток в тензоры
all_labels_tensors = [torch.tensor(label) for label in all_labels]

# Создание объекта Dataset
class TNDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TNDataset(all_encodings, all_labels_tensors)
# Настройка параметров тренировки
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Создание объекта Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Тренировка модели
trainer.train()

# Сохранение модели
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")


# Извлечение

# Загрузка модели и токенайзера
tokenizer = BertTokenizer.from_pretrained("./model")
model = BertForTokenClassification.from_pretrained("./model")

# Пайплайн для извлечения именованных сущностей
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Новый текст для анализа
new_text = ("Наименование заказчика - ПАО «Ростелеком». "
            "Проект - Создание СЗПДн."
            " Адрес - г.Москва, Ярославская ул., д.13А."
            " Срок выполнения - 40 календарных дней. "
            "Работы включают: Предпроектное обследование, разработка ТЗ, тех-рабочее проектирование."
            " Требования: Защита данных на всех уровнях. Объект защиты: Сетевые устройства и серверы.")

# Извлечение сущностей
entities = nlp(new_text)
print(entities)
