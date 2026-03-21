import sqlite3
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)

import pymorphy2


# Функция отображения структуры базы данных
def show_database_structure(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Получаем список всех таблиц
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("\nСписок таблиц:")
    for i, table in enumerate(tables):
        print(f"{i + 1}. Таблица: {table[0]}")

    # Получаем описание каждой таблицы
    for table_name in tables:
        cursor.execute(f"PRAGMA table_info('{table_name[0]}');")
        columns = cursor.fetchall()

        print(f"\nОписание таблицы '{table_name[0]}':")
        for column in columns:
            print(f"- Имя колонки: {column[1]}, Тип данных: {column[2]}")

    conn.close()


# Отображаем структуру базы данных
show_database_structure('articles.db')


# 1. Загрузка данных из SQLite базы данных
def load_data_from_db(db_file, table_name):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(f'SELECT description FROM {table_name}', conn)
    # Объединяем строки столбца в единый список предложений
    sentences = []
    for row in df['description']:
        # Пропускаем None значения
        if row is not None:
            sentences.extend(re.split(r'[.!?]+\s+', str(row)))

    # Фильтруем пустые строки
    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
    return sentences


sentences = load_data_from_db('articles.db', 'articles')
print(f"Всего предложений: {len(sentences)}")


# 2. Функция для приведения слова к нормальной форме
def normalize_word(word, pos, morph_analyzer):
    """
    Приводит слово к нормальной форме:
    - для глаголов - к инфинитиву
    - для существительных - к именительному падежу
    """
    if not word:
        return None

    try:
        parsed = morph_analyzer.parse(word)[0]

        # Определяем часть речи из POS тега
        pos_lower = pos.lower() if pos else ''

        # Если слово - глагол, приводим к инфинитиву
        if 'VERB' in pos_lower or 'INFN' in pos_lower:
            # Ищем форму инфинитива
            for form in parsed.lexeme:
                if 'infn' in str(form.tag):
                    return form.word
            return parsed.normal_form

        # Для существительных приводим к именительному падежу
        elif 'NOUN' in pos_lower:
            return parsed.normal_form

        # Для остальных частей речи возвращаем нормальную форму
        else:
            return parsed.normal_form
    except Exception as e:
        # В случае ошибки возвращаем исходное слово
        return word


# 3. Синтаксический разбор и выделение подлежащих и сказуемых с нормализацией
def extract_subject_and_predicate(sentence, segmenter, morph_tagger, syntax_parser, morph_analyzer):
    try:
        doc = Doc(sentence)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)

        subject = None
        predicate = None
        subject_pos = None
        predicate_pos = None

        if not doc.sents:
            return None, None

        for token in doc.sents[0].tokens:
            if token.rel == 'nsubj':
                subject = token.text.lower()
                # Получаем часть речи для подлежащего
                if token.pos:
                    subject_pos = token.pos
            elif token.rel == 'root':
                predicate = token.text.lower()
                # Получаем часть речи для сказуемого
                if token.pos:
                    predicate_pos = token.pos

        # Приводим слова к нормальной форме
        if subject and subject_pos:
            subject = normalize_word(subject, subject_pos, morph_analyzer)

        if predicate and predicate_pos:
            predicate = normalize_word(predicate, predicate_pos, morph_analyzer)

        return subject, predicate
    except Exception as e:
        # В случае ошибки пропускаем предложение
        return None, None


# 4. Инициализация инструментов
print("Инициализация инструментов...")
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
morph_analyzer = pymorphy2.MorphAnalyzer()

# 5. Получение и фильтрация пар подлежащее-сказуемое
print("Выполняется синтаксический разбор предложений...")
pairs = []
total_sentences = len(sentences)

for i, sent in enumerate(sentences):
    if i % 1000 == 0:
        print(f"Обработано {i} из {total_sentences} предложений ({i / total_sentences * 100:.1f}%)")

    # Ограничиваем длину предложения для производительности
    if len(sent) > 500:
        continue

    subj, pred = extract_subject_and_predicate(sent, segmenter, morph_tagger, syntax_parser, morph_analyzer)
    if subj and pred:
        pairs.append((subj, pred))

print(f"Всего пар подлежащее-сказуемое: {len(pairs)}")

# 6. Подсчет частот встречаемости
co_occurrences = Counter(pairs)

# Вывод TOP-100 сочетаний
print("\nТОП-20 наиболее частых сочетаний подлежащих и сказуемых:")
for i, ((subj, pred), freq) in enumerate(co_occurrences.most_common(100), 1):
    print(f"{i}. {subj} — {pred}: {freq} раз(а)")


# 7. Визуализация TOP-N сочетаний
def plot_top_pairs(co_occurrences, top_n=20):
    if not co_occurrences:
        print("Нет данных для визуализации")
        return

    top_pairs = co_occurrences.most_common(top_n)
    subjects_predicates = [f"{sub} — {pred}" for sub, pred in top_pairs]
    frequencies = [freq for _, freq in top_pairs]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(subjects_predicates, frequencies, color='skyblue')
    plt.title(
        f"ТОП-{top_n} наиболее частых сочетаний подлежащих и сказуемых\n(с нормализацией: глаголы в инфинитиве, существительные в им. падеже)")
    plt.xlabel("Частота")
    plt.ylabel("Сочетания подлежащих и сказуемых")
    plt.gca().invert_yaxis()

    # Добавляем значения на столбцы
    for bar, freq in zip(bars, frequencies):
        plt.text(freq + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(freq), va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


# Визуализируем результаты
if co_occurrences:
    plot_top_pairs(co_occurrences, top_n=20)

    # Дополнительная статистика
    print(f"\nСтатистика:")
    print(f"Всего уникальных сочетаний: {len(co_occurrences)}")
    print(f"Всего пар с частотой > 1: {sum(1 for freq in co_occurrences.values() if freq > 1)}")
    print(f"Максимальная частота: {max(co_occurrences.values())}")
    print(f"Средняя частота: {sum(co_occurrences.values()) / len(co_occurrences):.2f}")
else:
    print("Не найдено ни одной пары подлежащее-сказуемое")