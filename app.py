import streamlit as st
import google.generativeai as genai
from datetime import datetime
import re

# Page Config
st.set_page_config(
    page_title="JuLi Zeitungsanalyse",
    page_icon="📰",
    layout="wide"
)

# Custom CSS für besseres Design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .priority-high {
        border-left: 5px solid #FF4B4B;
        padding-left: 1rem;
        background-color: #FFF5F5;
    }
    .priority-medium {
        border-left: 5px solid #FF8C00;
        padding-left: 1rem;
        background-color: #FFF8DC;
    }
    .priority-low {
        border-left: 5px solid #32CD32;
        padding-left: 1rem;
        background-color: #F0FFF0;
    }
    .article-card {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .badge-high { background-color: #FF4B4B; color: white; }
    .badge-medium { background-color: #FF8C00; color: white; }
    .badge-low { background-color: #32CD32; color: white; }
    .category-badge { background-color: #1f77b4; color: white; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📰 JuLi Zeitungsanalyse Dashboard</h1>
    <p>Intelligente Artikel-Analyse für die Jungen Liberalen</p>
</div>
""", unsafe_allow_html=True)

def configure_gemini():
    """Konfiguriert Gemini API"""
    api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    else:
        st.sidebar.warning("⚠️ Bitte Gemini API Key eingeben")
        return None

def extract_articles_structured(analysis_text: str) -> dict:
    """Extrahiert Artikel strukturiert aus der Analyse"""
    articles = {
        'höchste': [],
        'hohe': [], 
        'standard': []
    }
    
    # Teile Text in Artikel auf
    article_blocks = re.split(r'---+', analysis_text)
    
    for block in article_blocks:
        block = block.strip()
        if len(block) < 50:  # Überspringe sehr kurze Blöcke
            continue
            
        # Extrahiere Informationen mit Regex
        title_match = re.search(r'\*\*(.*?)\*\*', block)
        summary_match = re.search(r'📍\s*(.*?)(?=📄|🎯|$)', block, re.DOTALL)
        page_match = re.search(r'📄\s*Seite:\s*(.*?)(?=🎯|$)', block)
        relevance_match = re.search(r'🎯\s*JuLi-Relevanz:\s*(.*?)$', block, re.DOTALL)
        
        if title_match:
            article = {
                'title': title_match.group(1).strip(),
                'summary': summary_match.group(1).strip() if summary_match else "",
                'page': page_match.group(1).strip() if page_match else "Nicht erkennbar",
                'relevance': relevance_match.group(1).strip() if relevance_match else "",
                'full_text': block
            }
            
            # Kategorisiere nach Priorität
            title_lower = article['title'].lower()
            if any(cat in title_lower for cat in ['kommunalpolitik', 'wirtschaft', 'bildung', 'verkehr', 'infrastruktur']):
                articles['höchste'].append(article)
            elif any(cat in title_lower for cat in ['digitalisierung', 'umwelt', 'nachhaltigkeit', 'bürgerbeteiligung', 'demokratie', 'jugend']):
                articles['hohe'].append(article)
            else:
                articles['standard'].append(article)
    
    return articles

def display_article_card(article: dict, priority: str):
    """Zeigt einen Artikel als schöne Karte an"""
    
    # Prioritäts-Badge
    if priority == 'höchste':
        badge_class = 'badge-high'
        icon = '🔥'
    elif priority == 'hohe':
        badge_class = 'badge-medium' 
        icon = '⚡'
    else:
        badge_class = 'badge-low'
        icon = '📰'
    
    # Kategorie aus Titel extrahieren
    category = article['title'].split(' - ')[0] if ' - ' in article['title'] else "Allgemein"
    
    with st.expander(f"{icon} {article['title']}", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="article-card">
                <span class="badge {badge_class}">{priority.upper()}</span>
                <span class="badge category-badge">{category}</span>
                
                <h4>📍 Zusammenfassung</h4>
                <p>{article['summary']}</p>
                
                <h4>🎯 JuLi-Relevanz</h4>
                <p>{article['relevance']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("📄 Seite", article['page'])
            
            # Relevanz-Score (einfache Heuristik)
            relevance_words = ['wichtig', 'zentral', 'kritisch', 'dringend', 'priorität']
            score = sum(1 for word in relevance_words if word in article['relevance'].lower())
            st.metric("🎯 Relevanz-Score", f"{min(score, 5)}/5")

def analyze_text_chunked(text: str, model) -> str:
    """Analysiert Text in Chunks"""
    
    chunk_size = 15000
    chunks = []
    current_pos = 0
    
    # Text in Chunks aufteilen
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        
        # Bei Absatz trennen wenn möglich
        if end_pos < len(text):
            last_paragraph = text.rfind('\n\n', current_pos, end_pos)
            if last_paragraph > current_pos:
                end_pos = last_paragraph
        
        chunk = text[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)
        current_pos = end_pos
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_analyses = []
    
    for i, chunk in enumerate(chunks):
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"🔍 Analysiere Teil {i+1}/{len(chunks)}...")
        
        prompt = f"""
        AUFTRAG: Analysiere diesen Zeitungstext für die Jungen Liberalen (JuLi).
        
        KATEGORIEN & PRIORITÄTEN:
        🔥 HÖCHSTE PRIORITÄT: Kommunalpolitik, Wirtschaft & Gewerbe, Bildung, Verkehr & Infrastruktur
        ⚡ HOHE PRIORITÄT: Digitalisierung & Innovation, Umwelt & Nachhaltigkeit, Bürgerbeteiligung & Demokratie, Jugendthemen  
        📰 STANDARD: Kultur & Events, Sport, Soziales, Sonstiges

        FORMAT (für jeden gefundenen Artikel):
        **[Kategorie] - Überschrift**
        📍 Kurze prägnante Zusammenfassung (max. 2 Sätze)
        📄 Seite: [Seitennummer oder "nicht erkennbar"]
        🎯 JuLi-Relevanz: Konkrete Begründung warum dieser Artikel für die JuLi wichtig/unwichtig ist
        ---

        WICHTIG:
        - Verwende immer "JuLi" (nie "JL")
        - Nur vollständige Artikel analysieren
        - Fokus auf politische/gesellschaftliche Relevanz
        
        TEXT:
        {chunk}
        """
        
        try:
            response = model.generate_content(prompt)
            all_analyses.append(response.text)
        except Exception as e:
            all_analyses.append(f"❌ Fehler: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return '\n\n'.join(all_analyses)

# Hauptanwendung
def main():
    model = configure_gemini()
    
    if not model:
        st.stop()
    
    # Sidebar Konfiguration
    st.sidebar.header("⚙️ Konfiguration")
    max_chars = st.sidebar.slider("📝 Max. Zeichen", 1000, 100000, 50000)
    
    # Text Input
    st.header("📄 Text eingeben")
    
    # Tabs für Input-Methoden
    tab1, tab2 = st.tabs(["📝 Text eingeben", "📁 Datei hochladen"])
    
    with tab1:
        input_text = st.text_area(
            "Zeitungstext hier einfügen:",
            height=200,
            placeholder="Hier den kompletten Zeitungstext einfügen..."
        )
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Datei auswählen",
            type=['txt', 'pdf', 'docx']
        )
        input_text = ""
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                input_text = str(uploaded_file.read(), "utf-8")
            else:
                st.warning("⚠️ PDF/DOCX Upload noch nicht implementiert")
    
    # Analyse starten
    if st.button("🚀 Analyse starten", type="primary", use_container_width=True):
        if not input_text:
            st.error("❌ Bitte Text eingeben!")
            return
        
        if len(input_text) > max_chars:
            st.warning(f"⚠️ Text zu lang ({len(input_text)} Zeichen). Wird auf {max_chars} gekürzt.")
            input_text = input_text[:max_chars]
        
        # Analyse durchführen
        with st.spinner("🔍 Analyse läuft..."):
            analysis_result = analyze_text_chunked(input_text, model)
        
        # Artikel extrahieren und strukturieren
        articles = extract_articles_structured(analysis_result)
        
        # Ergebnisse anzeigen
        st.header("📊 Analyse-Ergebnisse")
        
        # Übersichts-Metriken
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 Höchste Priorität", len(articles['höchste']))
        with col2:
            st.metric("⚡ Hohe Priorität", len(articles['hohe']))
        with col3:
            st.metric("📰 Standard", len(articles['standard']))
        with col4:
            total = len(articles['höchste']) + len(articles['hohe']) + len(articles['standard'])
            st.metric("📝 Gesamt", total)
        
        # Tabs für Prioritätsstufen
        if total > 0:
            tab_high, tab_medium, tab_low = st.tabs([
                f"🔥 Höchste Priorität ({len(articles['höchste'])})",
                f"⚡ Hohe Priorität ({len(articles['hohe'])})", 
                f"📰 Standard ({len(articles['standard'])})"
            ])
            
            with tab_high:
                if articles['höchste']:
                    st.markdown('<div class="priority-high">', unsafe_allow_html=True)
                    st.write("**🚨 Sofortige Aufmerksamkeit erforderlich**")
                    for article in articles['höchste']:
                        display_article_card(article, 'höchste')
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Keine Artikel mit höchster Priorität gefunden.")
            
            with tab_medium:
                if articles['hohe']:
                    st.markdown('<div class="priority-medium">', unsafe_allow_html=True)
                    st.write("**⚡ Wichtig für JuLi-Agenda**")
                    for article in articles['hohe']:
                        display_article_card(article, 'hohe')
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Keine Artikel mit hoher Priorität gefunden.")
            
            with tab_low:
                if articles['standard']:
                    st.markdown('<div class="priority-low">', unsafe_allow_html=True)
                    st.write("**📰 Zur Kenntnisnahme**")
                    for article in articles['standard']:
                        display_article_card(article, 'standard')
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Keine Standard-Artikel gefunden.")
            
            # Download-Option
            st.header("💾 Export")
            if st.button("📥 Vollständige Analyse herunterladen"):
                st.download_button(
                    label="📄 Als Text herunterladen",
                    data=analysis_result,
                    file_name=f"juli_analyse_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("⚠️ Keine Artikel gefunden. Überprüfe den Input-Text.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>📰 <strong>JuLi Zeitungsanalyse Dashboard</strong> | Powered by Gemini AI</p>
    <p><small>Entwickelt für die Jungen Liberalen - Intelligente Medienanalyse</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
