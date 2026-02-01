import re
import pandas as pd
import streamlit as st
import time
import random

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "OHY_proj_sample.csv"
KEY_COL = "listing_id"

SHOW_COLS = [
    "event_name",
    "event_type",
    "event_date",
    "venue_name",
    "distance_km",
    "days_until_event",
    "current_price",
    "suggested_price",
]

RENAME_COLS = {
    "days_until_event": "days_until",
    "suggested_price": "suggested_price_for_event",
}

ID_RE = re.compile(r"^[A-Za-z0-9]+$")

st.set_page_config(page_title="EventBnb", layout="wide")

# ----------------------------
# Session state defaults
# ----------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "results"  # results | generated
if "last_lookup_id" not in st.session_state:
    st.session_state["last_lookup_id"] = ""
if "last_matches" not in st.session_state:
    st.session_state["last_matches"] = None
if "generated_text" not in st.session_state:
    st.session_state["generated_text"] = None
if "selected_row" not in st.session_state:
    st.session_state["selected_row"] = None
if "current_description" not in st.session_state:
    st.session_state["current_description"] = ""
if "gen_in_flight" not in st.session_state:
    st.session_state["gen_in_flight"] = False
if "last_gen_ts" not in st.session_state:
    st.session_state["last_gen_ts"] = 0.0
if "gen_count" not in st.session_state:
    st.session_state["gen_count"] = 0

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure event_date is readable
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date

    # Normalize key col to string
    if KEY_COL in df.columns:
        df[KEY_COL] = df[KEY_COL].astype(str)

    return df


def is_valid_id(s: str) -> bool:
    return bool(ID_RE.fullmatch((s or "").strip()))


def build_prompt(event_row: pd.Series, current_description: str) -> str:
    """
    Prompt tuned for Gemini:
    - enforce "no hallucinations"
    - short, punchy output
    - event-goer angle
    """

    def g(col, default=""):
        v = event_row.get(col, default)
        if pd.isna(v):
            return default
        return v

    event_name = g("event_name", "an upcoming event")
    event_type = g("event_type", "event")
    event_date = g("event_date", "")
    venue_name = g("venue_name", "")
    distance_km = g("distance_km", "")
    days_until = g("days_until_event", "")
    current_price = g("current_price", "")
    suggested_price = g("suggested_price", "")

    current_description = (current_description or "").strip()
    if len(current_description) > 1500:
        current_description = current_description[:1500].rstrip() + "..."

    prompt = f"""
Task: Rewrite an Airbnb listing description to attract guests attending an event.

Strict rules:
- Keep ONLY facts stated in the original description. Do NOT invent amenities, rules, views, parking, neighborhood claims, or anything else.
- 30–70 words, one paragraph.
- Friendly, natural tone (no cringe marketing).
- Mention the event and convenience to the venue using the provided event info.

Event info:
- Name: {event_name}
- Type: {event_type}
- Date: {event_date}
- Venue: {venue_name}
- Distance to venue (km): {distance_km}
- Days until event: {days_until}

Pricing context (do not claim discounts or guarantees):
- Current price: {current_price}
- Suggested price: {suggested_price}

Original description:
{current_description}

Output ONLY the improved description text:
""".strip()

    return prompt

def call_gemini_with_backoff(fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            # common patterns: "429", "too many requests", "resource_exhausted"
            if "429" in msg or "too many requests" in msg or "resource_exhausted" in msg:
                sleep_s = min((2 ** attempt) + random.random(), 10)
                time.sleep(sleep_s)
                continue
            raise
    raise RuntimeError("Gemini is rate-limiting right now. Try again in ~30–60 seconds.")


def generate_description_gemini(api_key: str, model_name: str, prompt: str) -> str:
    """
    Gemini API call using google-genai SDK.
    pip install -U google-genai
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return (response.text or "").strip()


# ----------------------------
# UI
# ----------------------------
st.title("EventBnb")

df = load_df(CSV_PATH)
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1) Lookup")

    listing_id = st.text_input(
        "Enter listing id (letters & numbers only)",
        value=st.session_state.get("last_lookup_id", ""),
        placeholder="e.g. 8fA12Bc9",
    )

    do_lookup = st.button("Find listing", type="primary")

    st.divider()

    st.subheader("2) Generate new description")

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Create a key in Google AI Studio and paste it here.",
    )

    model_name = st.selectbox(
        "Model",
        options=[
            "gemini-2.5-flash",
        ],
        index=0,
        help="For more options, add through the source code.",
    )

    generate_btn = st.button("Generate (based on selected event)")

    if st.session_state["page"] == "generated":
        st.divider()
        back_btn = st.button("⬅ Back to found events")
        if back_btn:
            st.session_state["page"] = "results"
            st.rerun()


# ----------------------------
# Lookup action
# ----------------------------
if do_lookup:
    st.session_state["generated_text"] = None
    st.session_state["page"] = "results"

    if not is_valid_id(listing_id):
        st.session_state["last_matches"] = None
        st.session_state["last_lookup_id"] = listing_id
    else:
        st.session_state["last_lookup_id"] = listing_id.strip()
        matches = df[df[KEY_COL] == listing_id.strip()]
        st.session_state["last_matches"] = matches.copy() if not matches.empty else pd.DataFrame()


with right:
    # Show validation error after lookup click
    if do_lookup and not is_valid_id(listing_id):
        st.error("Invalid listing_id. Use only A–Z, a–z, 0–9 (no spaces).")

    # Page: generated
    if st.session_state["page"] == "generated":
        st.subheader("Generated description")
        txt = st.session_state.get("generated_text")
        if not txt:
            st.warning("No generated text yet. Click Generate on the left.")
        else:
            st.write(txt)

    # Page: results
    else:
        matches = st.session_state.get("last_matches", None)

        if matches is None:
            st.info("Enter a listing_id and click **Find listing**. Examples: 27926486, 43546204, 45491410 ")
        elif matches.empty:
            st.warning("No events found for that listing_id.")
        else:
            view = matches.copy()

            missing = [c for c in SHOW_COLS if c not in view.columns]
            if missing:
                st.error(f"Missing columns in CSV: {missing}")
            else:
                table_df = view[SHOW_COLS].rename(columns=RENAME_COLS)

                st.subheader("Upcoming Events Near You")
                st.dataframe(table_df, use_container_width=True)

                st.subheader("Pick an event to target")
                row_idx = st.selectbox(
                    "Row index",
                    options=list(range(len(view))),
                    format_func=lambda i: (
                        f"{i}: {view.iloc[i].get('event_name','')} | "
                        f"{view.iloc[i].get('event_date','')} | "
                        f"{view.iloc[i].get('venue_name','')}"
                    ),
                )

                st.session_state["selected_row"] = view.iloc[row_idx]
                st.session_state["current_description"] = str(view.iloc[row_idx].get("description", ""))

                st.info("Now click **Generate (based on selected event)** on the left.")


# ----------------------------
# Generate action
# ----------------------------
if generate_btn:
    if st.session_state.get("selected_row") is None:
        st.session_state["page"] = "results"
        st.toast("First run a lookup and select an event row.", icon="⚠️")
        st.rerun()

    if not api_key or len(api_key.strip()) < 10:
        st.error("Please paste a valid Gemini API key to generate text.")
        st.stop()

    if st.session_state["gen_in_flight"]:
        st.warning("Generation already in progress. Please wait.")
        st.stop()


    # COOLDOWN CHECK
    now = time.time()
    cooldown_seconds = 8
    if now - st.session_state["last_gen_ts"] < cooldown_seconds:
        wait = int(cooldown_seconds - (now - st.session_state["last_gen_ts"]))
        st.warning(f"Please wait {wait}s before generating again.")
        st.stop()

    st.session_state["gen_in_flight"] = True

    selected_row = st.session_state["selected_row"]
    current_desc = st.session_state.get("current_description", "")
    prompt = build_prompt(selected_row, current_desc)

    st.session_state["gen_count"] += 1
    st.write(f"Gemini calls this session: {st.session_state['gen_count']}")

    # ACTUAL GEMINI CALL
    try:
        with st.spinner("Generating with Gemini..."):
            new_desc = generate_description_gemini(
                api_key=api_key.strip(),
                model_name=model_name,
                prompt=prompt,
            )
    except Exception as e:
        st.session_state["gen_in_flight"] = False
        st.error(str(e))
        st.stop()

    st.session_state["last_gen_ts"] = time.time()
    st.session_state["gen_in_flight"] = False
    st.session_state["generated_text"] = new_desc
    st.session_state["page"] = "generated"
    st.rerun()


