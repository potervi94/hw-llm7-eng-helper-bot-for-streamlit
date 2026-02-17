# 🇬🇧 English Helper Bot

Персональний AI-помічник для вивчення англійської мови з базою знань, пам'яттю учнів та автоматичним накопиченням матеріалів.

## 🚀 Демо

**👉 [Відкрити бота](https://hw-llm7-eng-bot.streamlit.app)**

> Якщо бот "спить" — зачекайте ~30 секунд, Streamlit Cloud прокидає неактивні додатки.

---

## ✨ Можливості

- **📝 Переклад слів/фраз** — переклад + 2-3 приклади + синоніми
- **📖 Переклад речень** — переклад + розбір граматики (час, Passive Voice, there is/are тощо)
- **🧠 Пам'ять між сесіями** — бот пам'ятає ім'я, дату візиту, вивчені теми
- **📚 Єдина RAG база знань** — граматика + авто-накопичені теми в одному namespace
- **🔄 Ротація моделей** — 4 безкоштовні Gemini моделі з автоперемиканням
- **👤 Авторизація** — через GitHub/Google (Streamlit Cloud) або ручне введення
- **📊 Статистика** — розмір БД, кількість профілів, візити за день

---

## 🏗️ Архітектура

```
┌──────────────────────────────────────────────┐
│              Streamlit Cloud                 │
│  ┌────────────────────────────────────────┐  │
│  │          hw_llm7_main.py               │  │
│  │                                        │  │
│  │  Sidebar            Chat               │  │
│  │  ┌──────────┐      ┌──────────────┐    │  │
│  │  │Auth      │      │ Chat UI      │    │  │
│  │  │(GitHub / │      │ messages     │    │  │
│  │  │ manual)  │      │ input        │    │  │
│  │  ├──────────┤      └──────┬───────┘    │  │
│  │  │Models    │             │            │  │
│  │  │Stats     │             ▼            │  │
│  │  │Topics    │      ┌──────────────┐    │  │
│  │  └──────────┘      │ RAG + LLM    │    │  │
│  │                    │ with rotation│    │  │
│  │                    └──────┬───────┘    │  │
│  └───────────────────────────┼────────────┘  │
└──────────────────────────────┼───────────────┘
                               │
                               ▼
            ┌──────────────────────────────────┐
            │       Pinecone (1 index)         │
            │                                  │
            │  namespace: knowledge            │
            │  → граматика + авто-теми         │
            │    (єдина база, один пошук)      │
            │                                  │
            │  namespace: profiles             │
            │  → імена, візити, прогрес        │
            │                                  │
            │  namespace: history              │
            │  → усі розмови учнів             │
            └──────────────────────────────────┘
```

---

## 👤 Авторизація

Бот підтримує два способи авторизації:

### Автоматична (Streamlit Cloud + GitHub/Google)

Якщо у Settings → General → Viewer auth увімкнена авторизація,
бот автоматично визначає ім'я через `st.experimental_user.email`.
Учню достатньо натиснути "Почати навчання".

### Ручна (локально або без auth)

Учень вводить ім'я у бічній панелі. Ім'я використовується як ключ
для пошуку профілю та історії у Pinecone.

---

## 📊 Статистика в бічній панелі

Бот показує в реальному часі:

| Метрика | Опис |
|---------|------|
| 📚 Знання | Кількість тем у базі (початкові + авто) |
| 👥 Профілі | Кількість зареєстрованих учнів |
| 💬 Розмови | Загальна кількість збережених діалогів |
| 📦 Всього | Загальна кількість векторів в індексі |
| 🔢 Візити | Порядковий номер візиту та візити за день |

---

## 📋 Quickstart (локально)

### 1. Клонування

```bash
git clone https://github.com/potervi94/hw-llm7-eng-helper-bot-for-streamlit.git
cd hw-llm7-eng-helper-bot-for-streamlit
```

### 2. Середовище

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Залежності

```bash
pip install -r requirements.txt
```

### 4. API ключі

Створіть файл `.env`:

```env
GEMINI_API_KEY=ваш_ключ_gemini
PINECONE_API_KEY=ваш_ключ_pinecone
PINECONE_INDEX_NAME=english-helper
```

| Сервіс | Посилання | Безкоштовно |
|--------|-----------|-------------|
| Google Gemini API | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | ✅ 15 RPM |
| Pinecone | [app.pinecone.io](https://app.pinecone.io) | ✅ 1 індекс, 2GB |

### 5. Запуск

```bash
streamlit run hw_llm7_main.py
```

---

## 🌐 Деплой на Streamlit Cloud

1. Push файли на GitHub (`.env` у `.gitignore`)
2. [share.streamlit.io](https://share.streamlit.io) → New app
3. Repository: `potervi94/hw-llm7-eng-helper-bot-for-streamlit`
4. Branch: `main` | Main file: `hw_llm7_main.py`
5. Advanced settings → Python: `3.11`, Secrets:

```toml
GEMINI_API_KEY = "..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "english-helper"
```

6. **Deploy** → 3-5 хвилин → готово.

### Увімкнення авторизації через GitHub

Settings → General → Viewer authentication →
додайте email-адреси дозволених користувачів.
Бот автоматично визначить ім'я при вході.

---

## ⚠️ Обмеження Streamlit Cloud (безкоштовний план)

| Параметр | Обмеження |
|----------|-----------|
| Додатки | необмежено (тільки public repos) |
| RAM | ~1 GB |
| Сон | після ~7 днів без відвідувань (~30 сек пробудження) |
| Диск | тимчасовий (тому пам'ять у Pinecone) |
| Auth | email-whitelist (GitHub/Google) |

---

## 🔄 Ротація моделей

```
gemini-2.0-flash       (основна)
    ↓ quota exceeded
gemini-2.5-flash-lite  (резерв 1)
    ↓ quota exceeded
gemini-2.0-flash-lite  (резерв 2)
    ↓ quota exceeded
gemini-2.5-flash       (резерв 3)
    ↓ cooldown 60 хв → повертаємось до першої
```

---

## 📚 Автодоповнення бази знань

Після кожного речення (5+ слів):
1. LLM визначає граматичну тему → `"Passive Voice"`
2. Перевірка дублікату в `knowledge`
3. Генерація конспекту → збереження **в той самий namespace**
4. Один `similarity_search` знаходить і початкові, і авто-теми

---

## 📁 Структура проекту

```
├── hw_llm7_main.py            # бот
├── requirements.txt            # залежності
├── data/english_knowledge.txt  # початкова база (17 тем)
├── .env.example                # шаблон ключів
├── .gitignore
└── README.md
```

---

## 🛠️ Технології

**Python 3.11+** · **Streamlit** · **LangChain** · **Google Gemini API** · **Pinecone** · **Google Embedding** (768 dim)

---

📝 Homework project — ITStep AI course, LLM Part 7.
