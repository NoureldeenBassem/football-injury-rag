"""PitchSide AI look for the Streamlit app, matching the pitch deck.

Night-pitch green, neon lime and orange accents, Oswald headings, Rubik text,
faint football-pitch lines in the background and a cartoon footballer on the welcome screen.
Colours and fonts for Streamlit's own widgets are set in .streamlit/config.toml.
"""
from urllib.parse import quote

import streamlit as st

DARK = "#0B1F17"
SURFACE = "#12301F"
LINE = "#24553A"
LIME = "#C8F031"
ORANGE = "#FF7A2F"
TEXT = "#F3F5EE"
MUTED = "#B9CBBF"

_PITCH_LINES = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1920 1080' preserveAspectRatio='xMidYMid slice'>"
    "<g fill='none' stroke='#F3F5EE' stroke-opacity='0.05' stroke-width='4'>"
    "<rect x='60' y='60' width='1800' height='960'/><line x1='960' y1='60' x2='960' y2='1020'/>"
    "<circle cx='960' cy='540' r='160'/>"
    "<rect x='60' y='300' width='280' height='480'/><rect x='60' y='430' width='100' height='220'/>"
    "<path d='M340 440 A130 130 0 0 1 340 640'/>"
    "<rect x='1580' y='300' width='280' height='480'/><rect x='1760' y='430' width='100' height='220'/>"
    "<path d='M1580 440 A130 130 0 0 0 1580 640'/></g></svg>"
)

# Original cartoon footballer (same drawing as the deck's cover).
PLAYER_SVG = """<svg class="ps-player" viewBox="0 0 640 640" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cartoon footballer kicking a ball"><circle cx="330" cy="330" r="290" fill="none" stroke="#C8F031" stroke-opacity="0.35" stroke-width="4" stroke-dasharray="14 14"/><circle cx="330" cy="330" r="235" fill="#12301F"/><ellipse cx="320" cy="600" rx="190" ry="18" fill="#000000" fill-opacity="0.3"/><g stroke-linecap="round" stroke-linejoin="round" fill="none"><path d="M228 245 L205 232" stroke="#C8F031" stroke-width="36"/><path d="M205 232 L178 212 L152 170" stroke="#C98B5E" stroke-width="28"/><path d="M255 400 L242 495" stroke="#C98B5E" stroke-width="44"/><path d="M243 478 L240 502" stroke="#F3F5EE" stroke-width="48"/><path d="M241 512 L238 572" stroke="#C8F031" stroke-width="40"/><path d="M300 400 L372 455" stroke="#C98B5E" stroke-width="44"/><path d="M372 455 L452 432" stroke="#C8F031" stroke-width="40"/></g><circle cx="152" cy="168" r="17" fill="#C98B5E"/><path d="M212 576 h52 q24 0 24 18 v6 h-76 z" fill="#FF7A2F"/><ellipse cx="474" cy="426" rx="34" ry="19" transform="rotate(-16 474 426)" fill="#FF7A2F"/><path d="M226 356 L334 356 L344 420 L296 426 L280 394 L266 426 L218 418 Z" fill="#F3F5EE"/><path d="M224 232 Q280 210 336 232 L346 366 L214 366 Z" fill="#C8F031"/><rect x="270" y="262" width="20" height="56" rx="3" fill="#0B1F17"/><rect x="252" y="280" width="56" height="20" rx="3" fill="#0B1F17"/><path d="M262 224 Q280 240 298 224" fill="none" stroke="#0B1F17" stroke-width="6" stroke-linecap="round"/><path d="M332 246 L356 276" stroke="#C8F031" stroke-width="36" stroke-linecap="round"/><path d="M356 276 L376 302 L410 320" fill="none" stroke="#C98B5E" stroke-width="28" stroke-linecap="round" stroke-linejoin="round"/><circle cx="412" cy="322" r="17" fill="#C98B5E"/><rect x="268" y="196" width="24" height="32" fill="#C98B5E"/><circle cx="280" cy="160" r="50" fill="#C98B5E"/><circle cx="250" cy="166" r="11" fill="#B07448"/><path d="M230 152 Q234 100 282 104 Q332 104 332 152 Q318 128 284 126 Q250 126 230 152 Z" fill="#2B1B12"/><circle cx="294" cy="158" r="5" fill="#0B1F17"/><circle cx="316" cy="158" r="5" fill="#0B1F17"/><path d="M296 182 Q308 192 320 180" fill="none" stroke="#0B1F17" stroke-width="4" stroke-linecap="round"/><g stroke="#C8F031" stroke-width="6" stroke-linecap="round" stroke-opacity="0.7"><line x1="572" y1="398" x2="612" y2="392"/><line x1="576" y1="422" x2="624" y2="420"/><line x1="572" y1="446" x2="612" y2="448"/></g><circle cx="524" cy="420" r="40" fill="#F3F5EE" stroke="#0B1F17" stroke-width="4"/><path d="M524 404 L539 415 L533 433 L515 433 L509 415 Z" fill="#0B1F17"/><path d="M524 404 L524 382 M539 415 L560 408 M533 433 L546 452 M515 433 L502 452 M509 415 L488 408" stroke="#0B1F17" stroke-width="3"/></svg>"""

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400..700&family=Rubik:wght@300..700&display=swap');

/* Pitch background */
.stApp {{
  background-color: {DARK};
  background-image: url("data:image/svg+xml,{quote(_PITCH_LINES)}"),
                    radial-gradient(circle at 80% 10%, #17402A 0%, {DARK} 55%);
  background-size: cover, auto;
  background-position: center, center;
  background-attachment: fixed, fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* Headings: condensed sports type */
h1, h2, h3, [data-testid="stSidebar"] h1 {{
  font-family: 'Oswald', sans-serif !important;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}}

/* Sidebar with a kit stripe on its edge */
[data-testid="stSidebar"] {{
  border-right: 6px solid {LIME};
  box-shadow: inset -10px 0 0 {DARK};
}}
[data-testid="stSidebar"] h1 span {{ color: {LIME}; }}

/* Chat messages */
[data-testid="stChatMessage"] {{
  background: {SURFACE};
  border: 1px solid {LINE};
  border-radius: 18px;
  padding: 1rem 1.1rem;
}}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{
  background: rgba(200, 240, 49, 0.08);
  border-color: rgba(200, 240, 49, 0.45);
}}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"] {{
  background: {LIME}; color: {DARK}; border-radius: 50%;
}}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageAvatarCustom"] {{
  background: {ORANGE};
}}
[data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] {{ color: {DARK}; }}

/* Chat input */
[data-testid="stChatInput"] > div {{
  border: 2px solid {LINE};
  border-radius: 999px;
  background: {SURFACE};
}}
[data-testid="stChatInput"] > div:focus-within {{ border-color: {LIME}; }}

/* Sources and status boxes */
[data-testid="stExpander"] details {{
  background: {DARK};
  border: 1px solid {LINE};
  border-radius: 14px;
}}
[data-testid="stExpander"] summary {{ font-family: 'Oswald', sans-serif; letter-spacing: 0.06em; text-transform: uppercase; }}

/* Suggestion pills */
[data-testid="stPills"] button, [data-testid="stBaseButton-pills"] {{
  border: 1px solid {LINE};
  background: {SURFACE};
  border-radius: 999px;
}}
[data-testid="stBaseButton-pills"]:hover {{ border-color: {LIME}; color: {LIME}; }}

/* Welcome hero */
.ps-hero {{
  display: flex; align-items: center; gap: 1.5rem;
  background: linear-gradient(135deg, {SURFACE} 0%, {DARK} 100%);
  border: 1px solid {LINE};
  border-top: 6px solid {LIME};
  border-radius: 22px;
  padding: 1.6rem 1.8rem;
  margin-bottom: 1.2rem;
}}
.ps-hero-text {{ flex: 1; }}
.ps-eyebrow {{
  font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 0.25em;
  font-size: 0.8rem; color: {LIME}; margin: 0 0 0.4rem;
}}
.ps-title {{
  font-family: 'Oswald', sans-serif; font-weight: 700; text-transform: uppercase;
  font-size: 3rem; line-height: 1; color: {TEXT}; margin: 0 0 0.5rem;
}}
.ps-title span {{ color: {LIME}; }}
.ps-tag {{
  font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 0.05em;
  font-size: 1.15rem; color: {TEXT}; margin: 0 0 0.4rem;
}}
.ps-sub {{ color: {MUTED}; margin: 0; }}
.ps-player {{ width: 190px; height: 190px; flex: none; }}
.ps-chips {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.9rem; }}
.ps-chip {{
  font-size: 0.8rem; padding: 0.2rem 0.75rem; border-radius: 999px;
  background: {DARK}; color: {LIME}; border: 1px solid {LINE};
}}
.ps-label {{
  font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 0.2em;
  font-size: 0.8rem; color: {MUTED}; margin: 0.2rem 0 0.5rem;
}}
@media (max-width: 640px) {{
  .ps-player {{ width: 120px; height: 120px; }}
  .ps-title {{ font-size: 2.2rem; }}
}}
</style>
"""


def inject_theme():
    """Add the PitchSide AI styles to the page. Call once, right after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero():
    """Welcome banner with the cartoon footballer."""
    st.markdown(
        f"""
<div class="ps-hero">
  <div class="ps-hero-text">
    <p class="ps-eyebrow">Football injury intelligence</p>
    <div class="ps-title">PitchSide <span>AI</span></div>
    <p class="ps-tag">Prevent it. Treat it. Return faster.</p>
    <p class="ps-sub">Evidence-based answers with sources, in English, Arabic or Franco-Arab.</p>
    <div class="ps-chips">
      <span class="ps-chip">Prevention</span>
      <span class="ps-chip">Rehabilitation</span>
      <span class="ps-chip">Recovery time</span>
    </div>
  </div>
  {PLAYER_SVG}
</div>
<p class="ps-label">Try asking</p>
""",
        unsafe_allow_html=True,
    )
