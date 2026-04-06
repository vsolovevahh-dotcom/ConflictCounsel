import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
APP_TITLE = "NoDramaBot 🔥🧯"
APP_TAGLINE = "Мы гасим возражения, а не бюджеты"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "faq.json"
FEEDBACK_PATH = BASE_DIR / "feedback.json"

PRODUCTS = {
    "segments": {"label": "hh Сегменты", "avatar": "👨‍💻"},
    "clickme": {"label": "Clickme", "avatar": "🎯"},
    "vr": {"label": "Виртуальный рекрутер", "avatar": "🤖"},
    "cobrand": {"label": "Кобрендинг", "avatar": "🌐"},
}

MODES = {
    "rag": "GPT + База (RAG)",
    "gpt": "Только GPT",
    "chat": "Чат (GPT-4o)",
}

DEMO_PROMPTS = {
    "segments": "Клиент пишет раздраженно: «У вас 600 кликов, а лидов ноль. Почему мы сливаем бюджет?»",
    "clickme": "Клиент пишет резко: «Трафик идет, а откликов нет. Зачем мне этот Clickme?»",
    "vr": "Клиент недоволен: «Почему мне приходят нерелевантные отклики? Я не хочу переплачивать!»",
    "cobrand": "Клиент пишет с претензией: «Когда уже будут результаты по кобрендингу? Мы не видим эффекта!»",
}

DEFAULT_SYSTEM_PROMPT = """
Ты — трафик-менеджер hh. Отвечаешь клиентам по продуктам hh и даёшь внутренние рекомендации коллеге.

Продукты hh:

1) hh Сегменты
- Каналы: РСЯ/Яндекс (чаще всего), реже VK, TG Ads, посевы в Телеграм через Яндекс.
- Таргетинги: проф. роли, активность на сайте, ключевые фразы в резюме; возможен look-alike.
- Особенность: в посевах Телеграм через Яндекс доступно только таргетирование по гео, каналы подбираются алгоритмами Яндекса.

2) Clickme
- Механика для привлечения кликов к целевому действию (отклик на вакансии).
- Посадочная всегда hh.ru.
- Трафик идёт как с внешних площадок (Яндекс, VK), так и с внутренних размещений hh.
- Таргетинги: все доступные в Яндекс и VK (ключевые фразы, автотаргетинг и др.).

3) Виртуальный рекрутер
- Массовый автоматизированный найм.
- Оплата идёт за отклики на hh.ru.
- Трафик поступает с внешних площадок (Яндекс, VK, TG Ads, CPA/СРА-сети).

4) Кобрендинг
- Совместные кампании hh и партнёра в разных каналах.
- Цели: рост доверия к бренду, увеличение релевантных откликов.
- Каналы: Яндекс, VK, TG Ads (аналогично продуктам Сегменты и Clickme) + наружная реклама (статичная/динамичная).

Оплата:
- hh Сегменты, Clickme, Кобрендинг → CPC (СРС) или CPM (в TG Ads). Цена клика фиксирована и не обсуждается. Клиент покупает объём кликов/показов по фиксированной цене.
- Виртуальный рекрутер → клиент оплачивает количество откликов.

Трекинг:
- Все кампании по всем продуктам всегда промечены UTM-метками. Это не обсуждается и не требует подтверждения. Клиент может видеть данные в своих системах аналитики, например, в Яндекс Метрике.

Ограничения:
- Нет моделей оплаты за конверсии за пределами продукта Виртуальный рекрутер.
- hh управляет сегментами, таргетингом, настройками, форматами, УТП и позиционированием.

Частая проблема для hh Сегментов:
- Кампания не настроена на оптимизацию по целям → нужно подсветить клиенту:
  1) добавить счётчик Метрики,
  2) выбрать цели для оптимизации.

Правила:
- Всегда отвечай от имени трафик-менеджера hh.
- Не обсуждай изменение стоимости клика/показа — цена фиксирована.
- Не ставь под сомнение наличие UTM.
- Если продукт указан, используй именно его контекст.
- Если продукт не указан в запросе явно, ориентируйся на переданный параметр продукта.
- Если речь о типовой проблеме, используй контекст из базы как опору, но не копируй его дословно.

Формат ответа всегда строго из двух частей:

=== ОТВЕТ КЛИЕНТУ ===
2–4 абзаца простым языком:
- признай переживания клиента,
- объясни ситуацию,
- опиши следующие шаги,
- заверши дружелюбно и профессионально.

=== СОВЕТЫ ТРАФИК-МЕНЕДЖЕРУ ===
2–5 коротких пунктов:
- что проверить,
- что можно оптимизировать,
- какие гипотезы протестировать.

Тон:
- клиенту — спокойный, поддерживающий, профессиональный;
- советы менеджеру — кратко, по делу.
"""

# =========================
# FILE HELPERS
# =========================
def ensure_file(path: Path, default_value: Any) -> None:
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_value, f, ensure_ascii=False, indent=2)

def load_json(path: Path, default_value: Any) -> Any:
    ensure_file(path, default_value)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================
# OPENAI
# =========================
def get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def build_user_prompt(
    query: str,
    product_key: str,
    mode_key: str,
    rag_context: Optional[str] = None,
) -> str:
    product_label = PRODUCTS[product_key]["label"]
    parts = [
        f"Продукт: {product_label}",
        f"Режим: {MODES[mode_key]}",
        f"Запрос клиента:\n{query.strip()}",
    ]
    if rag_context:
        parts.append(f"Контекст из базы кейсов:\n{rag_context}")
    return "\n\n".join(parts)

def ask_gpt(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.3,
        max_tokens=900,
        messages=messages,
    )
    return response.choices[0].message.content or ""

# =========================
# FAQ / SEARCH
# =========================
def case_to_text(case: Dict[str, Any]) -> str:
    fields = [
        case.get("title", ""),
        " ".join(case.get("labels", [])),
        " ".join(case.get("answers", [])),
        " ".join(case.get("tips", [])),
    ]
    return " | ".join([x for x in fields if x])

def filter_cases_by_product(cases: List[Dict[str, Any]], product_key: str) -> List[Dict[str, Any]]:
    return [c for c in cases if c.get("product", "segments") == product_key]

def semantic_search(
    query: str,
    cases: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Tuple[Dict[str, Any], float]]:
    if not cases:
        return []

    corpus = [case_to_text(c) for c in cases]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None
    )
    matrix = vectorizer.fit_transform(corpus + [query])
    query_vec = matrix[-1]
    case_vecs = matrix[:-1]

    sims = cosine_similarity(query_vec, case_vecs).flatten()
    ranked = sorted(
        [(cases[i], float(sims[i])) for i in range(len(cases))],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]

def build_rag_context(matches: List[Tuple[Dict[str, Any], float]]) -> str:
    if not matches:
        return ""
    blocks = []
    for idx, (case, score) in enumerate(matches, start=1):
        answers = "\n- ".join(case.get("answers", [])[:2])
        tips = "\n- ".join(case.get("tips", [])[:3])
        blocks.append(
            f"""Кейс #{idx} (релевантность: {score:.2f})
Заголовок: {case.get("title", "")}
Метки: {", ".join(case.get("labels", []))}
Ответы:
- {answers}
Советы:
- {tips}
"""
        )
    return "\n\n".join(blocks)

# =========================
# ADMIN / FEEDBACK
# =========================
def is_admin_enabled() -> bool:
    return bool(os.getenv("ADMIN_PASSWORD", "").strip())

def check_admin(password: str) -> bool:
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    return bool(admin_password) and password == admin_password

def save_feedback(
    query: str,
    product_key: str,
    mode_key: str,
    answer: str,
    is_helpful: bool
) -> None:
    feedback = load_json(FEEDBACK_PATH, [])
    feedback.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "product": product_key,
        "mode": mode_key,
        "helpful": is_helpful,
        "answer_preview": answer[:500],
    })
    save_json(FEEDBACK_PATH, feedback)

def normalize_multiline(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]

# =========================
# STREAMLIT STATE
# =========================
def init_state() -> None:
    defaults = {
        "history": [],
        "last_answer": "",
        "last_query": "",
        "admin_ok": False,
        "mode": "rag",
        "product": "segments",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =========================
# UI
# =========================
def render_header(product_key: str) -> None:
    avatar = PRODUCTS[product_key]["avatar"]
    product_label = PRODUCTS[product_key]["label"]
    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;border:1px solid #eee;border-radius:12px;background:#fff7e6;">
  <div style="font-size:34px;">🤖🧯</div>
  <div>
    <div style="font-weight:700;">NoDramaBot на линии</div>
    <div style="font-size:14px;color:#555;">Текущий продукт: {avatar} {product_label}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

def render_sidebar() -> Tuple[str, str, bool]:
    st.sidebar.header("Настройки")
    mode_key = st.sidebar.radio(
        "Режим",
        options=list(MODES.keys()),
        format_func=lambda x: MODES[x],
        index=list(MODES.keys()).index(st.session_state["mode"]),
    )
    product_key = st.sidebar.selectbox(
        "Продукт",
        options=list(PRODUCTS.keys()),
        format_func=lambda x: f'{PRODUCTS[x]["avatar"]} {PRODUCTS[x]["label"]}',
        index=list(PRODUCTS.keys()).index(st.session_state["product"]),
    )
    demo_mode = st.sidebar.button("🎭 Разыграть конфликт", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Технологии**")
    st.sidebar.caption("Streamlit • OpenAI GPT-4o mini • TF-IDF semantic search • JSON база кейсов")

    st.session_state["mode"] = mode_key
    st.session_state["product"] = product_key
    return mode_key, product_key, demo_mode

def render_chat_history(product_key: str) -> None:
    if not st.session_state["history"]:
        st.info("Здесь появится история диалога. Начните с вопроса клиента или включите демо-конфликт.")
        return

    avatar = PRODUCTS[product_key]["avatar"]
    for msg in st.session_state["history"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar=avatar):
                st.markdown(content)

def render_feedback_block() -> None:
    if not st.session_state.get("last_answer"):
        return

    st.markdown("### Оценка ответа")
    col1, col2 = st.columns(2)
    if col1.button("👍 Ответ полезный", use_container_width=True):
        save_feedback(
            st.session_state.get("last_query", ""),
            st.session_state.get("product", "segments"),
            st.session_state.get("mode", "rag"),
            st.session_state.get("last_answer", ""),
            True,
        )
        st.success("Спасибо! Сохранили положительный фидбэк.")
    if col2.button("👎 Ответ нужно доработать", use_container_width=True):
        save_feedback(
            st.session_state.get("last_query", ""),
            st.session_state.get("product", "segments"),
            st.session_state.get("mode", "rag"),
            st.session_state.get("last_answer", ""),
            False,
        )
        st.warning("Спасибо! Сохранили фидбэк на доработку.")

def render_admin(cases: List[Dict[str, Any]]) -> None:
    st.markdown("---")
    st.subheader("🛠 Админка кейсов")

    if is_admin_enabled() and not st.session_state["admin_ok"]:
        with st.form("admin_login"):
            password = st.text_input("Пароль администратора", type="password")
            submitted = st.form_submit_button("Войти")
            if submitted:
                if check_admin(password):
                    st.session_state["admin_ok"] = True
                    st.success("Доступ к админке открыт.")
                else:
                    st.error("Неверный пароль.")
        return

    if is_admin_enabled() and st.session_state["admin_ok"]:
        st.success("Режим администратора активен.")
    elif not is_admin_enabled():
        st.info("ADMIN_PASSWORD не задан. Админка доступна без пароля.")

    tab1, tab2, tab3 = st.tabs(["Список кейсов", "Добавить кейс", "Фидбэк"])

    with tab1:
        st.markdown("#### База кейсов")
        if not cases:
            st.warning("База кейсов пока пустая.")
        else:
            for idx, case in enumerate(cases):
                with st.expander(f'{case.get("title")} ({PRODUCTS.get(case.get("product", "segments"), {}).get("label", "segments")})'):
                    st.write(f"**ID:** {case.get('id')}")
                    st.write(f"**Метки:** {', '.join(case.get('labels', []))}")
                    st.write(f"**Tone:** {case.get('tone', 'neutral')}")
                    st.write("**Ответы:**")
                    for a in case.get("answers", []):
                        st.markdown(f"- {a}")
                    st.write("**Советы:**")
                    for t in case.get("tips", []):
                        st.markdown(f"- {t}")

                    if st.button("Удалить кейс", key=f"delete_{idx}"):
                        new_cases = [c for c in cases if c.get("id") != case.get("id")]
                        save_json(FAQ_PATH, new_cases)
                        st.success("Кейс удалён. Обновите страницу.")
                        st.stop()

    with tab2:
        st.markdown("#### Новый кейс")
        with st.form("new_case_form"):
            product = st.selectbox(
                "Продукт",
                options=list(PRODUCTS.keys()),
                format_func=lambda x: f'{PRODUCTS[x]["avatar"]} {PRODUCTS[x]["label"]}'
            )
            title = st.text_input("Заголовок кейса")
            labels = st.text_input("Метки через запятую")
            tone = st.selectbox("Тон", ["neutral", "calm", "formal"])
            answers = st.text_area("Ответы (каждый с новой строки)")
            tips = st.text_area("Советы (каждый с новой строки)")
            submitted = st.form_submit_button("Сохранить кейс")

            if submitted:
                if not title.strip():
                    st.error("Нужен заголовок.")
                else:
                    new_case = {
                        "id": str(uuid.uuid4())[:8],
                        "product": product,
                        "title": title.strip(),
                        "labels": [x.strip() for x in labels.split(",") if x.strip()],
                        "tone": tone,
                        "answers": normalize_multiline(answers),
                        "tips": normalize_multiline(tips),
                        "links": [],
                        "updatedAt": datetime.utcnow().isoformat(),
                    }
                    current = load_json(FAQ_PATH, [])
                    current.append(new_case)
                    save_json(FAQ_PATH, current)
                    st.success("Кейс добавлен. Обновите страницу или перезапустите приложение.")

    with tab3:
        feedback = load_json(FEEDBACK_PATH, [])
        st.markdown("#### Последний фидбэк")
        if not feedback:
            st.info("Фидбэк пока не собран.")
        else:
            for item in reversed(feedback[-20:]):
                helpful = "👍" if item.get("helpful") else "👎"
                st.write(
                    f"{helpful} [{item.get('timestamp')}] "
                    f"{PRODUCTS.get(item.get('product', 'segments'), {}).get('label', item.get('product'))} / "
                    f"{MODES.get(item.get('mode', 'rag'), item.get('mode'))}"
                )
                st.caption(item.get("query", ""))
                st.code(item.get("answer_preview", "")[:300])

# =========================
# MAIN ACTIONS
# =========================
def handle_demo_conflict(product_key: str) -> None:
    demo_text = DEMO_PROMPTS[product_key]
    st.session_state["demo_seed"] = demo_text

def handle_submit(
    client: Optional[OpenAI],
    system_prompt: str,
    cases: List[Dict[str, Any]],
    mode_key: str,
    product_key: str,
    user_query: str,
) -> None:
    if not client:
        st.error("Не найден OPENAI_API_KEY. Добавьте ключ в переменные окружения.")
        return

    if not user_query.strip():
        st.warning("Введите вопрос клиента.")
        return

    st.session_state["last_query"] = user_query

    rag_context = ""
    product_cases = filter_cases_by_product(cases, product_key)

    if mode_key == "rag":
        matches = semantic_search(user_query, product_cases, top_k=3)
        rag_context = build_rag_context(matches)

    history_for_gpt = None
    if mode_key == "chat":
        history_for_gpt = st.session_state["history"][:]

    user_prompt = build_user_prompt(
        query=user_query,
        product_key=product_key,
        mode_key=mode_key,
        rag_context=rag_context if mode_key == "rag" else None,
    )

    with st.spinner("NoDramaBot думает..."):
        answer = ask_gpt(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history_for_gpt,
        )

    st.session_state["history"].append({"role": "user", "content": user_query})
    st.session_state["history"].append({"role": "assistant", "content": answer})
    st.session_state["last_answer"] = answer

# =========================
# APP
# =========================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧯", layout="wide")
    init_state()

    ensure_file(FAQ_PATH, [])
    ensure_file(FEEDBACK_PATH, [])

    cases = load_json(FAQ_PATH, [])
    client = get_openai_client()

    mode_key, product_key, demo_mode = render_sidebar()
    render_header(product_key)

    if demo_mode:
        handle_demo_conflict(product_key)

    demo_seed = st.session_state.get("demo_seed", "")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 💬 Диалог")
        render_chat_history(product_key)

        default_value = demo_seed if demo_seed else ""
        user_query = st.chat_input("Напишите вопрос / жалобу клиента...")
        if demo_seed and not user_query:
            st.info(f"Демо-конфликт подготовлен:\n\n{demo_seed}")

        if user_query:
            handle_submit(
                client=client,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                cases=cases,
                mode_key=mode_key,
                product_key=product_key,
                user_query=user_query,
            )
            if "demo_seed" in st.session_state:
                del st.session_state["demo_seed"]
            st.rerun()

        if demo_seed:
            if st.button("▶️ Отправить демо-конфликт", use_container_width=True):
                handle_submit(
                    client=client,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    cases=cases,
                    mode_key=mode_key,
                    product_key=product_key,
                    user_query=demo_seed,
                )
                del st.session_state["demo_seed"]
                st.rerun()

        render_feedback_block()

    with col_right:
        st.markdown("### ℹ️ Как пользоваться")
        st.markdown(
            """
1. Выберите **режим**:
   - **GPT + База (RAG)** — ищет похожие кейсы и отвечает точнее
   - **Только GPT** — отвечает без опоры на базу
   - **Чат** — продолжает живой диалог с контекстом
2. Выберите **продукт**
3. Введите **вопрос клиента**
4. Получите:
   - ответ клиенту
   - советы менеджеру
"""
        )

        st.markdown("### 🧠 Что под капотом")
        st.markdown(
            """
- **GPT-4o mini** для генерации ответов
- **TF-IDF semantic search** по базе кейсов
- **Админка** для пополнения базы
- **Фидбэк** на качество ответов
"""
        )

        st.markdown("### ⚠️ Ограничения")
        st.caption(
            "Если база кейсов небольшая, режим GPT + База может работать слабее. "
            "Для сложных случаев используйте режим Только GPT или Чат."
        )

    render_admin(cases)


if __name__ == "__main__":
    main()
