import warnings
warnings.filterwarnings("ignore")

# Імпорт необхідних бібліотек
import os
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from google.api_core.exceptions import ResourceExhausted

# Завантаження змінних середовища з файлу .env
load_dotenv()


# ============================================================
# Отримання API ключів (з .env або Streamlit Secrets)
# ============================================================
def get_secret(key: str) -> str | None:
    """Отримує секретне значення з .env або Streamlit Secrets."""
    value = os.getenv(key)
    if value:
        return value
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return None


gemini_api_key = get_secret("GEMINI_API_KEY")
pinecone_api_key = get_secret("PINECONE_API_KEY")
INDEX_NAME = get_secret("PINECONE_INDEX_NAME") or "english-helper"

# шлях до файлу бази знань
BASE_DIR = os.path.dirname(__file__) if "__file__" in dir() else "."
DATA_FILE = os.path.join(BASE_DIR, "data", "english_knowledge.txt")


# ============================================================
# Ротація моделей — безкоштовні моделі з окремими квотами
# ============================================================
FREE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]

QUOTA_COOLDOWN = 60 * 60  # 60 хв cooldown


def get_current_model() -> str:
    """Повертає назву поточної активної моделі."""
    return FREE_MODELS[st.session_state.get("current_model_idx", 0)]


def is_model_available(model_name: str) -> bool:
    """Перевіряє чи модель доступна."""
    exhausted = st.session_state.get("exhausted_models", {})
    if model_name not in exhausted:
        return True
    if time.time() - exhausted[model_name] >= QUOTA_COOLDOWN:
        del st.session_state["exhausted_models"][model_name]
        return True
    return False


def rotate_model() -> str | None:
    """Переключається на наступну доступну модель."""
    idx = st.session_state["current_model_idx"]
    st.session_state["exhausted_models"][FREE_MODELS[idx]] = time.time()
    for i in range(1, len(FREE_MODELS)):
        c_idx = (idx + i) % len(FREE_MODELS)
        if is_model_available(FREE_MODELS[c_idx]):
            st.session_state["current_model_idx"] = c_idx
            return FREE_MODELS[c_idx]
    return None


def create_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Створює LLM інстанс для вказаної моделі."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=gemini_api_key,
        temperature=0.7,
    )


def invoke_with_rotation(messages: list, context: str = "", max_retries: int = 4) -> str:
    """Викликає LLM з ротацією моделей та RAG контекстом."""
    for attempt in range(max_retries):
        try:
            llm = create_llm(get_current_model())
            if context:
                enriched = list(messages)
                last = enriched[-1]
                enriched[-1] = HumanMessage(content=(
                    f"{last.content}\n\n"
                    f"--- Контекст з бази знань та історії ---\n{context}\n"
                    f"--- Кінець контексту ---\n\n"
                    f"Використай контекст для точнішої відповіді. "
                    f"Якщо контекст не стосується запиту — ігноруй."
                ))
                return llm.invoke(enriched).content
            return llm.invoke(messages).content
        except (ResourceExhausted, Exception) as e:
            err = str(e).lower()
            is_quota = any(w in err for w in [
                "429", "resource_exhausted", "quota", "rate limit",
                "404", "not_found", "not found", "deprecated",
            ]) or isinstance(e, ResourceExhausted)
            if not is_quota:
                raise
            if rotate_model() is None:
                raise RuntimeError("Всі моделі вичерпали квоту. Спробуйте пізніше.")
    raise RuntimeError("Перевищено кількість спроб ротації.")


# ============================================================
# Pinecone: 3 namespace в одному індексі
#
#   knowledge — граматика + авто-накопичені теми (єдина база)
#   profiles  — профілі учнів (ім'я, візити, прогрес)
#   history   — історія розмов (для контексту навчання)
# ============================================================
@st.cache_resource
def init_pinecone():
    """Ініціалізація Pinecone з 3 namespace (кешується)."""
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key,
        output_dimensionality=768,
    )
    pc = Pinecone(api_key=pinecone_api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(5)

    index = pc.Index(INDEX_NAME)
    vs_knowledge = PineconeVectorStore(index=index, embedding=embedding, namespace="knowledge")
    vs_profiles = PineconeVectorStore(index=index, embedding=embedding, namespace="profiles")
    vs_history = PineconeVectorStore(index=index, embedding=embedding, namespace="history")

    return index, vs_knowledge, vs_profiles, vs_history


def load_knowledge_base(vs_knowledge, index):
    """Завантажує початкову базу знань з файлу (один раз)."""
    stats = index.describe_index_stats()
    count = stats.namespaces.get("knowledge", {}).get("vector_count", 0)
    if count > 0:
        return count

    if not os.path.exists(DATA_FILE):
        return 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("\n\n\n")
    blocks = [b.strip() for b in blocks if b.strip()]

    docs, ids = [], []
    for block in blocks:
        block_name = block.split("\n")[0].strip()
        docs.append(Document(
            page_content=block,
            metadata={
                "block_name": block_name,
                "source": "english_knowledge.txt",
                "type": "grammar",
            },
        ))
        ids.append(str(uuid.uuid4()))

    vs_knowledge.add_documents(documents=docs, ids=ids)
    return len(docs)


def get_db_stats(index) -> dict:
    """Отримує детальну статистику з Pinecone."""
    stats = index.describe_index_stats()
    ns = stats.namespaces

    return {
        "knowledge_count": ns.get("knowledge", {}).get("vector_count", 0),
        "profiles_count": ns.get("profiles", {}).get("vector_count", 0),
        "history_count": ns.get("history", {}).get("vector_count", 0),
        "total_vectors": stats.total_vector_count,
        "dimension": stats.dimension,
    }


# ============================================================
# Пошук у єдиній базі знань
# ============================================================
def search_knowledge(vs_knowledge, query: str, k: int = 3) -> str:
    """Один пошук — знаходить і початкові теми, і авто-накопичені."""
    try:
        results = vs_knowledge.similarity_search(query, k=k)
        if results:
            return "\n\n---\n\n".join([d.page_content for d in results])
    except Exception:
        pass
    return ""


# ============================================================
# Автодоповнення бази знань (в той самий namespace knowledge)
# ============================================================
def extract_and_save_topic(vs_knowledge, user_query: str, bot_response: str):
    """
    Після речення (5+ слів): витягує тему → перевіряє дублікат →
    генерує конспект → зберігає в knowledge.
    """
    if len(user_query.strip().split()) <= 4:
        return

    try:
        # визначаємо тему
        topic_name = invoke_with_rotation([
            SystemMessage(content=(
                "Ти — лінгвістичний класифікатор. "
                "Визнач ОДНУ основну граматичну тему речення. "
                "Відповідай ТІЛЬКИ назвою теми англійською: "
                "Present Perfect, Passive Voice, Conditionals Type 2 тощо. "
                "Нічого більше."
            )),
            HumanMessage(content=user_query),
        ]).strip().strip('"\'.')

        if not topic_name or len(topic_name) > 60:
            return

        # перевіряємо дублікат
        existing = search_knowledge(vs_knowledge, topic_name, k=1)
        if existing and topic_name.lower() in existing.lower():
            return

        # генеруємо конспект
        note = invoke_with_rotation([
            SystemMessage(content=(
                "Створи КОРОТКИЙ конспект граматичної теми українською.\n"
                "Назва теми → Пояснення (1-2 речення) → Формула → "
                "3 приклади (англ + укр) → Маркери/ключові слова.\n"
                "Нічого зайвого."
            )),
            HumanMessage(content=f"Тема: {topic_name}"),
        ])

        if not note or len(note) < 50:
            return

        # зберігаємо в той самий namespace knowledge
        doc = Document(
            page_content=note,
            metadata={
                "block_name": topic_name,
                "source": "auto_generated",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type": "grammar",
            },
        )
        topic_id = make_ascii_id("auto", topic_name)
        vs_knowledge.add_documents(documents=[doc], ids=[topic_id])

        if topic_name not in st.session_state.get("topics_studied", []):
            st.session_state["topics_studied"].append(topic_name)

    except Exception:
        pass


# ============================================================
# Профілі користувачів
# ============================================================
def make_ascii_id(prefix: str, text: str) -> str:
    """Створює ASCII-safe ID з будь-якого тексту (кирилиця, emoji тощо)."""
    import hashlib
    text_hash = hashlib.md5(text.lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{text_hash}"


def save_user_profile(vs_profiles, user_name: str, topics: list[str] = None):
    """Зберігає/оновлює профіль у Pinecone."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    today = datetime.now().strftime("%Y-%m-%d")
    profile_id = make_ascii_id("profile", user_name)

    # лічильник візитів за поточний день
    day_visits = st.session_state.get("day_visits", 0)

    text = f"Користувач {user_name}. Останній візит: {now}."
    if topics:
        text += f" Вивчені теми: {', '.join(topics[-15:])}."

    doc = Document(
        page_content=text,
        metadata={
            "user_name": user_name.lower(),
            "last_visit": now,
            "last_visit_date": today,
            "visit_count": st.session_state.get("visit_count", 1),
            "day_visits": day_visits,
            "type": "profile",
        },
    )
    vs_profiles.add_documents(documents=[doc], ids=[profile_id])


def find_user_profile(vs_profiles, user_name: str) -> dict | None:
    """Шукає профіль користувача."""
    try:
        results = vs_profiles.similarity_search(
            f"користувач {user_name}", k=3,
            filter={"user_name": user_name.lower()},
        )
        if results:
            return results[0].metadata
    except Exception:
        pass
    return None


# ============================================================
# Історія розмов
# ============================================================
def save_conversation_turn(vs_history, user_name: str, user_msg: str, ai_msg: str):
    """Зберігає пару Q&A в історію."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = Document(
        page_content=f"Учень {user_name} запитав: {user_msg}\nВідповідь: {ai_msg[:500]}",
        metadata={
            "user_name": user_name.lower(),
            "timestamp": now,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "user_query": user_msg[:200],
            "type": "conversation",
        },
    )
    vs_history.add_documents(documents=[doc], ids=[str(uuid.uuid4())])


def search_user_history(vs_history, user_name: str, query: str, k: int = 3) -> str:
    """Шукає релевантні минулі розмови учня."""
    try:
        results = vs_history.similarity_search(
            query, k=k, filter={"user_name": user_name.lower()},
        )
        if results:
            return "\n\n".join([
                f"[{d.metadata.get('timestamp', '')}] {d.page_content[:300]}"
                for d in results
            ])
    except Exception:
        pass
    return ""


# ============================================================
# Авторизація: GitHub (Streamlit Cloud) або ручне введення
# ============================================================
def detect_user() -> str | None:
    """
    Спроба автоматично визначити користувача:
    1. Streamlit Cloud з увімкненою авторизацією → email з GitHub/Google
    2. Якщо не вдалося — повертає None (ручне введення)
    """
    try:
        # st.experimental_user доступний на Streamlit Cloud з auth
        user_info = st.experimental_user
        if user_info and hasattr(user_info, "email") and user_info.email:
            return user_info.email.split("@")[0].title()
    except Exception:
        pass
    return None


# ============================================================
# Завдання 1
# Напишіть додаток з чат ботом по допомозі з вивченням
# англійської мови.
#  Якщо користувач просить перекласти слово або
#  фразу, то вивести переклад та приклад використання
#  у реченні
#  Якщо користувач просить перекласти речення, то
#  вивести переклад та пояснення граматики, наприклад
#  структура there is/are, пасивна форма дієслова, тощо
# ============================================================

# Системний промпт
SYSTEM_PROMPT = """Ти — дружній викладач англійської мови на ім'я EnglishBot.
Спілкуєшся українською, допомагаєш учню вивчати англійську.
Маєш базу знань з граматикою та пам'ятаєш історію навчання учня.

ПРАВИЛА:
1. СЛОВО / ФРАЗА (до 3-4 слів):
   - Визнач мову → переклад у зворотному напрямку
   - 2-3 приклади у реченнях (англ + укр)
   - Синоніми, антоніми

2. РЕЧЕННЯ (5+ слів):
   - Визнач мову → переклад у зворотному напрямку
   - Розбір граматики: час, Passive Voice, there is/are, used to тощо
   - Використовуй дані з бази знань

3. ПЕРСОНАЛІЗАЦІЯ:
   - Посилайся на минулі розмови учня
   - Нагадуй вивчені теми
   - Хвали за прогрес, пропонуй повторення

4. Відповідай структуровано, з емодзі.
"""


# ============================================================
# Ініціалізація session_state
# ============================================================
def init_state():
    defaults = {
        "current_model_idx": 0,
        "exhausted_models": {},
        "messages": [],
        "llm_history": [SystemMessage(content=SYSTEM_PROMPT)],
        "user_name": None,
        "user_identified": False,
        "visit_count": 0,
        "day_visits": 0,
        "topics_studied": [],
        "db_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ============================================================
# Streamlit UI — налаштування сторінки
# ============================================================
st.set_page_config(page_title="🇬🇧 English Helper Bot", page_icon="🇬🇧", layout="wide")
st.title("🇬🇧 English Helper Bot")
st.markdown("Персональний помічник для вивчення англійської мови")


# ============================================================
# Pinecone ініціалізація
# ============================================================
vs_knowledge = vs_profiles = vs_history = index = None
db_stats = {}

if pinecone_api_key:
    try:
        index, vs_knowledge, vs_profiles, vs_history = init_pinecone()
        if not st.session_state["db_ready"]:
            load_knowledge_base(vs_knowledge, index)
            st.session_state["db_ready"] = True
        db_stats = get_db_stats(index)
    except Exception as e:
        st.error(f"Помилка Pinecone: {e}")


# ============================================================
# Бічна панель — авторизація
# ============================================================
st.sidebar.header("👤 Авторизація")

# спроба автоматичної авторизації через GitHub/Google (Streamlit Cloud)
auto_name = detect_user()

if not st.session_state["user_identified"]:
    if auto_name:
        # автоматична авторизація через Streamlit Cloud
        st.sidebar.success(f"✅ Авторизовано: **{auto_name}**")
        st.sidebar.caption("Визначено автоматично через Streamlit Cloud")

        if st.sidebar.button("📝 Почати навчання", use_container_width=True):
            st.session_state["user_name"] = auto_name
            st.session_state["user_identified"] = True
            st.rerun()

        st.sidebar.markdown("---")
        other_name = st.sidebar.text_input("Або введіть інше ім'я:")
        if other_name and st.sidebar.button("Увійти", key="other_login"):
            st.session_state["user_name"] = other_name.strip().title()
            st.session_state["user_identified"] = True
            st.rerun()
    else:
        # ручне введення імені
        st.sidebar.caption("Введіть ім'я для збереження прогресу")
        name_input = st.sidebar.text_input("Ваше ім'я:", placeholder="Наприклад: Олена")
        if name_input and st.sidebar.button("🚀 Почати навчання", use_container_width=True):
            st.session_state["user_name"] = name_input.strip().title()
            st.session_state["user_identified"] = True
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.info(
            "💡 **Streamlit Cloud + GitHub:**\n"
            "Увімкніть авторизацію у Settings → "
            "General → Viewer auth, щоб входити "
            "автоматично через GitHub/Google акаунт."
        )
else:
    user_name = st.session_state["user_name"]
    st.sidebar.success(f"✅ **{user_name}** (візит #{st.session_state['visit_count']})")
    topics = st.session_state.get("topics_studied", [])
    if topics:
        with st.sidebar.expander(f"📚 Вивчені теми ({len(topics)})"):
            for t in topics[-10:]:
                st.markdown(f"- {t}")

    if st.sidebar.button("🚪 Вийти", use_container_width=True):
        # зберігаємо профіль перед виходом
        if vs_profiles:
            save_user_profile(vs_profiles, user_name, topics)
        for k in ["user_name", "user_identified", "messages", "llm_history",
                   "visit_count", "day_visits", "topics_studied"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ============================================================
# Бічна панель — моделі
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🔄 Моделі")
st.sidebar.markdown(f"**Активна:** `{get_current_model()}`")
for i, m in enumerate(FREE_MODELS):
    icon = "✅" if i == st.session_state["current_model_idx"] else (
        "⏳" if not is_model_available(m) else "🟢"
    )
    st.sidebar.markdown(f"{icon} `{m}`")


# ============================================================
# Бічна панель — статистика бази даних
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("📊 Статистика БД")

if db_stats:
    col1, col2 = st.sidebar.columns(2)
    col1.metric("📚 Знання", db_stats["knowledge_count"])
    col2.metric("👥 Профілі", db_stats["profiles_count"])
    col1.metric("💬 Розмови", db_stats["history_count"])
    col2.metric("📦 Всього", db_stats["total_vectors"])

    st.sidebar.caption(f"Розмірність: {db_stats['dimension']} | Індекс: `{INDEX_NAME}`")

    # кількість повідомлень за сесію
    session_msgs = len([m for m in st.session_state["messages"] if m["role"] == "human"])
    st.sidebar.markdown(f"**Повідомлень за сесію:** {session_msgs}")

    # візити за день (з профілю)
    if st.session_state.get("day_visits", 0) > 0:
        st.sidebar.markdown(f"**Візитів сьогодні:** {st.session_state['day_visits']}")
else:
    if not pinecone_api_key:
        st.sidebar.warning("⚠️ PINECONE_API_KEY не вказано")
    else:
        st.sidebar.info("Завантаження...")


# ============================================================
# Обробка входу користувача (після авторизації)
# ============================================================
if st.session_state["user_identified"] and st.session_state["visit_count"] == 0:
    user_name = st.session_state["user_name"]

    # шукаємо профіль у Pinecone
    profile = find_user_profile(vs_profiles, user_name) if vs_profiles else None

    if profile:
        # повертається учень
        last_visit = profile.get("last_visit", "невідомо")
        last_date = profile.get("last_visit_date", "")
        visit_count = profile.get("visit_count", 0) + 1
        today = datetime.now().strftime("%Y-%m-%d")

        # рахуємо візити за день
        if last_date == today:
            day_visits = profile.get("day_visits", 0) + 1
        else:
            day_visits = 1

        st.session_state["visit_count"] = visit_count
        st.session_state["day_visits"] = day_visits

        welcome = (
            f"Радий знову тебе бачити, **{user_name}**! 🎉\n\n"
            f"📅 Останній візит: **{last_visit}**\n"
            f"🔢 Це твій **{visit_count}-й** візит "
            f"(**{day_visits}-й** сьогодні)\n\n"
        )

        # підсумок минулих тем
        if vs_history:
            past = search_user_history(vs_history, user_name, "англійська граматика", k=5)
            if past:
                try:
                    summary = invoke_with_rotation([
                        SystemMessage(content=(
                            "Коротко (2-3 речення) підсумуй що учень "
                            "вивчав раніше і запропонуй продовжити або повторити."
                        )),
                        HumanMessage(content=f"Розмови учня:\n{past}"),
                    ])
                    welcome += summary
                except Exception:
                    welcome += "Давай продовжимо вивчення! 📚"
            else:
                welcome += "Давай продовжимо! Що хочеш вивчити? 📚"
        else:
            welcome += "Давай продовжимо! 📚"

        if vs_profiles:
            save_user_profile(vs_profiles, user_name)

    else:
        # новий учень
        st.session_state["visit_count"] = 1
        st.session_state["day_visits"] = 1
        welcome = (
            f"Приємно познайомитися, **{user_name}**! 🤝\n\n"
            f"Я запам'ятаю твій прогрес між сесіями.\n"
            f"Надсилай слово, фразу або речення — "
            f"і я допоможу з перекладом та граматикою! 🚀"
        )
        if vs_profiles:
            save_user_profile(vs_profiles, user_name)

    # зберігаємо привітання
    st.session_state["messages"].append({"role": "ai", "content": welcome})
    st.session_state["llm_history"].append(
        HumanMessage(content=f"Мене звати {user_name}")
    )
    st.session_state["llm_history"].append(AIMessage(content=welcome))
    st.rerun()


# ============================================================
# Привітальне повідомлення (до авторизації)
# ============================================================
if not st.session_state["messages"] and not st.session_state["user_identified"]:
    st.info(
        "👈 **Введіть ваше ім'я** у бічній панелі, щоб почати навчання.\n\n"
        "Бот запам'ятає ваш прогрес і допоможе з вивченням англійської! 🇬🇧"
    )


# ============================================================
# Відображення чату
# ============================================================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# Обробка введення (тільки якщо авторизований)
# ============================================================
if st.session_state["user_identified"]:
    user_input = st.chat_input("Введіть слово, фразу або речення...")
else:
    user_input = None
    st.chat_input("Спочатку введіть ім'я у бічній панелі...", disabled=True)

if user_input:
    user_name = st.session_state["user_name"]

    st.session_state["messages"].append({"role": "human", "content": user_input})
    with st.chat_message("human"):
        st.markdown(user_input)

    with st.chat_message("ai"):
        with st.spinner(f"Думаю... ({get_current_model()})"):
            try:
                st.session_state["llm_history"].append(
                    HumanMessage(content=user_input)
                )

                # --- RAG: один пошук у knowledge ---
                ctx = []
                if vs_knowledge:
                    k = search_knowledge(vs_knowledge, user_input, k=3)
                    if k:
                        ctx.append(f"📚 ГРАМАТИКА:\n{k}")
                if vs_history:
                    h = search_user_history(vs_history, user_name, user_input, k=2)
                    if h:
                        ctx.append(f"🧠 МИНУЛІ РОЗМОВИ:\n{h}")

                context = "\n\n".join(ctx)
                if context:
                    context += (
                        f"\n\nУчня звати {user_name}, "
                        f"візит #{st.session_state['visit_count']}."
                    )

                # --- Виклик LLM ---
                response_text = invoke_with_rotation(
                    st.session_state["llm_history"], context=context
                )

                st.session_state["llm_history"].append(
                    AIMessage(content=response_text)
                )
                st.session_state["messages"].append(
                    {"role": "ai", "content": response_text}
                )
                st.markdown(response_text)

                # --- Зберігаємо в Pinecone ---
                if vs_history:
                    save_conversation_turn(
                        vs_history, user_name, user_input, response_text
                    )

                # --- Автодоповнення бази знань ---
                if vs_knowledge:
                    extract_and_save_topic(vs_knowledge, user_input, response_text)

                # --- Оновлюємо профіль ---
                if vs_profiles:
                    save_user_profile(
                        vs_profiles, user_name,
                        st.session_state["topics_studied"]
                    )

            except RuntimeError as e:
                err = f"❌ {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "ai", "content": err})
            except Exception as e:
                err = f"❌ Помилка: {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "ai", "content": err})
