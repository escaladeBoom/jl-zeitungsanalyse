import streamlit as st
import pypdf
import google.generativeai as genai
import io
import json
import re
from datetime import datetime
import hashlib
from typing import Dict, List

# Konfiguration
st.set_page_config(
    page_title="JL Zeitungsanalyse", 
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Team-Zugangsdaten (in Produktion: externe Datenbank verwenden)
TEAM_CREDENTIALS = {
    "jl_team": "junge_liberale_2025"  # Passwort für alle Teammitglieder
}

# CSS für besseres Design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .category-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFD700;
        margin: 0.5rem 0;
    }
    .jl-highlight {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .login-box {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def authenticate_user(username: str, password: str) -> bool:
    """Nutzer-Authentifizierung"""
    return username in TEAM_CREDENTIALS and TEAM_CREDENTIALS[username] == password

def setup_gemini_api(api_key: str):
    """Gemini API konfigurieren"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"API-Fehler: {e}")
        return None

def extract_pdf_text(pdf_file) -> str:
    """PDF-Text extrahieren"""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF-Fehler: {e}")
        return ""

def analyze_articles_jl(text: str, model) -> Dict:
    """Artikel analysieren - speziell für Junge Liberale"""
    
    prompt = f"""
    Analysiere den folgenden Zeitungstext aus kommunalpolitischer Sicht der JUNGEN LIBERALEN.

    KATEGORIEN (Priorität für Junge Liberale):
    
    🔥 HOHE PRIORITÄT:
    - Junge Menschen (Jugendpolitik, Ausbildung, Studentenangelegenheiten)
    - Digitalisierung (E-Government, Breitband, Smart City)
    - Wirtschaft & Startups (Gründerförderung, Gewerbe, Innovation)
    - Bildung (Schulen, Unis, Digitale Bildung)
    - Verkehr & Mobilität (ÖPNV, Fahrrad, Digitale Lösungen)
    
    📊 MITTLERE PRIORITÄT:
    - Kommunalfinanzen (Haushalt, Verschuldung, Steuern)
    - Stadtentwicklung (Wohnen, Bauprojekte)
    - Umwelt & Klima (Nachhaltige Lösungen)
    - Kultur & Sport (Events, Jugendkultur)
    
    📝 SONSTIGE:
    - Soziales (klassische Sozialpolitik)
    - Sicherheit (Polizei, Ordnung)
    - Verwaltung (Bürokratie, Abläufe)
    - Sonstiges

    SPEZIAL-SUCHE nach Stichworten:
    - "junge Menschen", "Jugend", "Student", "Ausbildung"
    - "digital", "online", "App", "Internet"
    - "Startup", "Innovation", "Gründer"
    - "liberal", "FDP", "Freie Demokraten"

    AUFGABE:
    1. Erkenne alle relevanten Artikel
    2. Kategorisiere nach obigen Prioritäten
    3. Markiere SPEZIAL wenn JL-relevante Stichwörter vorkommen
    4. Bewerte Relevanz für Junge Liberale (1-5 Sterne)
    5. Kurze Zusammenfassung + Warum wichtig für JL

    AUSGABEFORMAT (JSON):
    {{
        "kategorie_name": [
            {{
                "titel": "Artikel-Titel",
                "relevanz_sterne": 4,
                "jl_spezial": true/false,
                "zusammenfassung": "Was passiert",
                "jl_relevanz": "Warum wichtig für Junge Liberale",
                "original_text": "Erste ~150 Wörter...",
                "keywords": ["gefundene", "JL-relevante", "begriffe"]
            }}
        ]
    }}

    TEXT:
    {text[:8000]}
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            st.error("Konnte JSON nicht aus API-Antwort extrahieren")
            return {}
    except Exception as e:
        st.error(f"Analyse-Fehler: {e}")
        return {}

def show_login():
    """Login-Bildschirm"""
    st.markdown('<div class="main-header"><h1>🗞️ JL Zeitungsanalyse Portal</h1><p>Kommunalpolitik-Dashboard für Junge Liberale</p></div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="login-box">', unsafe_allow_html=True)
            st.markdown("### 🔐 Team-Login")
            
            username = st.text_input("Benutzername:", placeholder="jl_team")
            password = st.text_input("Passwort:", type="password", placeholder="Vorstand-Passwort")
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("Einloggen", use_container_width=True):
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("❌ Falsche Zugangsdaten!")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.info("🔑 **Für Vorstandsmitglieder:** Benutzername und Passwort beim Vorstand erfragen.")

def show_dashboard():
    """Haupt-Dashboard nach erfolgreichem Login"""
    
    # Header
    st.markdown('<div class="main-header"><h1>🗞️ JL Zeitungsanalyse Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Logout Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("🚪 Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # API Setup
    with st.sidebar:
        st.header("⚙️ API-Konfiguration")
        api_key = st.text_input(
            "Google Gemini API-Key:", 
            type="password",
            help="Kostenlos bei https://makersuite.google.com/app/apikey"
        )
        
        if api_key:
            st.success("✅ API-Key gesetzt")
        else:
            st.warning("⚠️ API-Key benötigt")
            st.info("📖 https://makersuite.google.com/app/apikey")
    
    if not api_key:
        st.warning("🔑 Bitte zuerst API-Key in der Sidebar eingeben!")
        return
    
    # Gemini API setup
    model = setup_gemini_api(api_key)
    if not model:
        return
    
    # PDF Upload Section
    st.header("📄 Zeitung hochladen")
    
    uploaded_file = st.file_uploader(
        "PDF auswählen:",
        type=['pdf'],
        help="Lokale Zeitungen (z.B. Tagblatt, Rundschau, etc.)"
    )
    
    if uploaded_file:
        st.success(f"✅ PDF geladen: {uploaded_file.name}")
        
        # Analyse-Button
        if st.button("🚀 Analyse starten", type="primary", use_container_width=True):
            
            with st.spinner("🔍 PDF wird verarbeitet..."):
                text = extract_pdf_text(uploaded_file)
            
            if text:
                st.info(f"📄 Text extrahiert: {len(text):,} Zeichen")
                
                with st.spinner("🤖 KI analysiert für Junge Liberale..."):
                    analysis = analyze_articles_jl(text, model)
                
                if analysis:
                    st.success("✅ Analyse abgeschlossen!")
                    
                    # Statistiken
                    total_articles = sum(len(articles) for articles in analysis.values())
                    jl_special = sum(
                        sum(1 for article in articles if article.get('jl_spezial', False))
                        for articles in analysis.values()
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📰 Artikel gesamt", total_articles)
                    with col2:
                        st.metric("🔥 JL-relevant", jl_special)
                    with col3:
                        st.metric("📊 Kategorien", len(analysis))
                    with col4:
                        st.metric("📅 Analysiert", datetime.now().strftime("%H:%M"))
                    
                    st.markdown("---")
                    
                    # JL-Spezial Highlights zuerst
                    jl_highlights = []
                    for category, articles in analysis.items():
                        for article in articles:
                            if article.get('jl_spezial', False):
                                jl_highlights.append((category, article))
                    
                    if jl_highlights:
                        st.markdown('<div class="jl-highlight"><h3>🔥 JL-SPEZIAL: Besonders relevante Artikel</h3></div>', unsafe_allow_html=True)
                        
                        for category, article in jl_highlights:
                            with st.expander(f"⭐ {article.get('titel', 'Artikel')} ({category})"):
                                stars = "⭐" * article.get('relevanz_sterne', 1)
                                st.markdown(f"**Relevanz:** {stars}")
                                st.markdown(f"**Keywords:** {', '.join(article.get('keywords', []))}")
                                st.markdown(f"**Zusammenfassung:** {article.get('zusammenfassung', 'Keine Zusammenfassung')}")
                                st.markdown(f"**JL-Relevanz:** {article.get('jl_relevanz', 'Nicht spezifiziert')}")
                                
                                with st.expander("📖 Volltext anzeigen"):
                                    st.markdown(article.get('original_text', 'Kein Text'))
                    
                    st.markdown("---")
                    
                    # Kategorien-Tabs
                    st.header("📊 Alle Kategorien")
                    categories = list(analysis.keys())
                    
                    if categories:
                        # Prioritäts-Sortierung
                        priority_order = [
                            "Junge Menschen", "Digitalisierung", "Wirtschaft & Startups", 
                            "Bildung", "Verkehr & Mobilität", "Kommunalfinanzen",
                            "Stadtentwicklung", "Umwelt & Klima", "Kultur & Sport"
                        ]
                        
                        sorted_categories = []
                        for prio_cat in priority_order:
                            if prio_cat in categories:
                                sorted_categories.append(prio_cat)
                        
                        # Restliche Kategorien anhängen
                        for cat in categories:
                            if cat not in sorted_categories:
                                sorted_categories.append(cat)
                        
                        tabs = st.tabs([f"{cat} ({len(analysis[cat])})" for cat in sorted_categories])
                        
                        for i, category in enumerate(sorted_categories):
                            with tabs[i]:
                                articles = analysis[category]
                                
                                if articles:
                                    # Sortiere nach Relevanz
                                    articles_sorted = sorted(
                                        articles, 
                                        key=lambda x: (x.get('jl_spezial', False), x.get('relevanz_sterne', 0)), 
                                        reverse=True
                                    )
                                    
                                    for article in articles_sorted:
                                        special_icon = "🔥 " if article.get('jl_spezial', False) else ""
                                        stars = "⭐" * article.get('relevanz_sterne', 1)
                                        
                                        with st.expander(f"{special_icon}{article.get('titel', 'Artikel')} {stars}"):
                                            
                                            col_a, col_b = st.columns([2, 1])
                                            with col_a:
                                                st.markdown(f"**Zusammenfassung:**")
                                                st.markdown(article.get('zusammenfassung', 'Keine Zusammenfassung'))
                                            
                                            with col_b:
                                                if article.get('jl_spezial', False):
                                                    st.markdown("**🔥 JL-RELEVANT**")
                                                    st.markdown(f"**Keywords:** {', '.join(article.get('keywords', []))}")
                                            
                                            if article.get('jl_relevanz'):
                                                st.markdown(f"**💡 JL-Relevanz:** {article.get('jl_relevanz')}")
                                            
                                            with st.expander("📖 Originaltext lesen"):
                                                st.markdown(article.get('original_text', 'Kein Text'))
                                else:
                                    st.info("Keine Artikel in dieser Kategorie gefunden.")

def main():
    """Hauptfunktion"""
    
    # Session State initialisieren
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Login-Check
    if not st.session_state.authenticated:
        show_login()
    else:
        show_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **JL Zeitungsanalyse** - Entwickelt für effiziente Kommunalpolitik | Made with ❤️ für Junge Liberale")

if __name__ == "__main__":
    main()
