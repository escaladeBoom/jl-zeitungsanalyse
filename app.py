import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
import io
import pandas as pd
from datetime import datetime
import hashlib
import re

# Konfiguration mit Fallback
def get_credentials():
    try:
        return {"jl_team": st.secrets["JL_PASSWORD"]}
    except:
        return {"jl_team": "junge_liberale_2025"}  # Fallback für Testing

TEAM_CREDENTIALS = get_credentials()

# Database-Funktionen
def save_analysis_to_db(pdf_name: str, analysis_text: str, full_text: str):
    """Analyseergebnis in CSV-Database speichern"""
    try:
        # Eindeutige ID für Artikel basierend auf Text-Hash
        article_hash = hashlib.md5(full_text.encode()).hexdigest()[:12]
        
        # Daten für CSV
        data = {
            'id': article_hash,
            'datum': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'pdf_name': pdf_name,
            'analyse': analysis_text,
            'volltext_kurz': full_text[:500] + "..." if len(full_text) > 500 else full_text
        }
        
        # CSV laden oder erstellen
        try:
            df = pd.read_csv('jl_artikel_database.csv')
        except:
            df = pd.DataFrame(columns=['id', 'datum', 'pdf_name', 'analyse', 'volltext_kurz'])
        
        # Duplikat-Check
        if article_hash not in df['id'].values:
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv('jl_artikel_database.csv', index=False)
            return True
        return False
        
    except Exception as e:
        st.error(f"Database-Fehler: {e}")
        return False

def load_article_database():
    """Artikel-Database laden"""
    try:
        return pd.read_csv('jl_artikel_database.csv')
    except:
        return pd.DataFrame(columns=['id', 'datum', 'pdf_name', 'analyse', 'volltext_kurz'])

def search_articles(query: str, df: pd.DataFrame):
    """Artikel durchsuchen"""
    if query:
        mask = df['analyse'].str.contains(query, case=False, na=False) | \
               df['pdf_name'].str.contains(query, case=False, na=False) | \
               df['volltext_kurz'].str.contains(query, case=False, na=False)
        return df[mask]
    return df

def extract_pdf_text(pdf_file) -> str:
    """PDF-Text extrahieren mit verbesserter Multi-Page Unterstützung"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
        
        # Debug-Info anzeigen
        total_pages = len(pdf_reader.pages)
        st.info(f"📄 PDF hat {total_pages} Seiten")
        
        text = ""
        page_texts = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            # Füge Seitenmarkierung hinzu
            page_texts.append(f"\n[SEITE {page_num}]\n{page_text}")
            text += f"\n[SEITE {page_num}]\n{page_text}\n"
        
        # Gesamt-Info
        st.success(f"✅ Extrahiert: {len(text)} Zeichen aus {total_pages} Seiten")
        
        return text
        
    except Exception as e:
        st.error(f"PDF-Fehler: {e}")
        return ""

def analyze_with_gemini(text: str, api_key: str) -> str:
    """Text mit Google Gemini analysieren - mit Chunking für lange Texte"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Text-Länge prüfen
        text_length = len(text)
        st.info(f"📝 Text-Länge: {text_length} Zeichen")
        
        # Gemini 1.5 Flash kann ~1M Tokens = ~4M Zeichen
        # Aber für Sicherheit chunken wir bei 50k Zeichen
        max_chunk_size = 50000
        
        if text_length <= max_chunk_size:
            # Kurzer Text - normale Analyse
            st.info("✅ Text passt in ein Stück - normale Analyse")
            return analyze_complete_text(text, model)
        else:
            # Langer Text - in Chunks aufteilen
            st.warning(f"⚠️ Text zu lang ({text_length} Zeichen) - wird in Teile aufgeteilt")
            return analyze_chunked_text(text, model, max_chunk_size)
        
    except Exception as e:
        return f"❌ **Analyse-Fehler:** {str(e)}"

def analyze_complete_text(text: str, model) -> str:
    """Gesamten Text analysieren und formatiert ausgeben"""
    prompt = f"""
    AUFTRAG: Analysiere diesen Zeitungstext und finde NUR die WICHTIGSTEN Artikel für die Jungen Liberalen.
    
    WICHTIG: 
    - Zeige NUR Artikel mit HÖCHSTER oder HOHER Priorität
    - Ignoriere Standard-Artikel komplett (Sport, Kultur, etc.)
    - Extrahiere IMMER die Seitenzahl aus [SEITE X] Markierungen
    
    HÖCHSTE PRIORITÄT (🔥):
    - Kommunalpolitik (Stadtrat, Bürgermeister, lokale Wahlen)
    - Wirtschaftsförderung & Gewerbeansiedlungen
    - Bildungspolitik (Schulen, Unis, Digitalisierung)
    - Verkehr & Infrastruktur
    
    HOHE PRIORITÄT (⚡):
    - Digitalisierung & Innovation
    - Umwelt & Nachhaltigkeit (pragmatische Ansätze)
    - Bürgerbeteiligung & Demokratie
    - Jugendthemen & -beteiligung
    
    FORMAT FÜR JEDEN ARTIKEL:
    ### [EMOJI] Überschrift des Artikels
    **Seite:** [Nummer]
    **Kernaussage:** [1-2 Sätze - Was ist die wichtigste Information?]
    **JuLi-Relevanz:** [1 Satz - Warum sollten JuLis hier aktiv werden?]
    
    ---
    
    GEBE NUR DIE WICHTIGSTEN ARTIKEL AUS!
    
    TEXT:
    {text}
    """
    
    response = model.generate_content(prompt)
    return format_final_output(response.text)

def analyze_chunked_text(text: str, model, chunk_size: int) -> str:
    """Langen Text in Chunks aufteilen und analysieren"""
    
    # Text in Chunks aufteilen
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        
        # Versuche bei Seitenmarkierung zu trennen
        if end_pos < len(text):
            # Suche letzte Seitenmarkierung
            last_page_marker = text.rfind('[SEITE', current_pos, end_pos)
            if last_page_marker > current_pos:
                end_pos = last_page_marker
        
        chunk = text[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)
        
        current_pos = end_pos
    
    st.info(f"📄 Text aufgeteilt in {len(chunks)} Teile")
    
    # Sammle alle Artikel aus allen Chunks
    all_articles = []
    
    for i, chunk in enumerate(chunks, 1):
        with st.spinner(f"🔍 Analysiere Teil {i}/{len(chunks)}..."):
            
            chunk_prompt = f"""
            AUFTRAG: Extrahiere NUR WICHTIGE Artikel aus diesem Zeitungstext-Teil für die Jungen Liberalen.
            
            WICHTIG: 
            - Nur HÖCHSTE und HOHE Priorität
            - Seitenzahlen aus [SEITE X] extrahieren
            - Kurz und präzise
            
            HÖCHSTE PRIORITÄT: Kommunalpolitik, Wirtschaft, Bildung, Verkehr
            HOHE PRIORITÄT: Digitalisierung, Umwelt, Bürgerbeteiligung, Jugend
            
            FORMAT PRO ARTIKEL:
            TITEL: [Überschrift]
            SEITE: [Nummer]
            KATEGORIE: [Höchste/Hohe Priorität]
            INHALT: [Kernaussage in 1-2 Sätzen]
            RELEVANZ: [JuLi-Relevanz in 1 Satz]
            ===
            
            TEXT TEIL {i}:
            {chunk}
            """
            
            try:
                response = model.generate_content(chunk_prompt)
                # Parse die Artikel aus der Antwort
                articles = parse_articles_from_response(response.text)
                all_articles.extend(articles)
            except Exception as e:
                st.error(f"Fehler bei Teil {i}: {e}")
    
    # Erstelle finale Ausgabe
    return create_final_summary(all_articles)

def parse_articles_from_response(response_text: str) -> list:
    """Extrahiere strukturierte Artikel aus der KI-Antwort"""
    articles = []
    
    # Teile nach === auf
    raw_articles = response_text.split('===')
    
    for raw_article in raw_articles:
        if not raw_article.strip():
            continue
            
        article = {}
        lines = raw_article.strip().split('\n')
        
        for line in lines:
            if line.startswith('TITEL:'):
                article['titel'] = line.replace('TITEL:', '').strip()
            elif line.startswith('SEITE:'):
                article['seite'] = line.replace('SEITE:', '').strip()
            elif line.startswith('KATEGORIE:'):
                article['kategorie'] = line.replace('KATEGORIE:', '').strip()
            elif line.startswith('INHALT:'):
                article['inhalt'] = line.replace('INHALT:', '').strip()
            elif line.startswith('RELEVANZ:'):
                article['relevanz'] = line.replace('RELEVANZ:', '').strip()
        
        if 'titel' in article:  # Nur hinzufügen wenn Titel vorhanden
            articles.append(article)
    
    return articles

def create_final_summary(articles: list) -> str:
    """Erstelle finale formatierte Zusammenfassung"""
    if not articles:
        return "❌ Keine relevanten Artikel gefunden."
    
    # Sortiere nach Priorität
    hoechste = [a for a in articles if 'Höchste' in a.get('kategorie', '')]
    hohe = [a for a in articles if 'Hohe' in a.get('kategorie', '')]
    
    output = "# 📰 ANALYSE-ERGEBNIS\n\n"
    
    # Zusammenfassung
    output += f"**Gefunden:** {len(articles)} relevante Artikel\n"
    output += f"- 🔥 Höchste Priorität: {len(hoechste)}\n"
    output += f"- ⚡ Hohe Priorität: {len(hohe)}\n\n"
    output += "---\n\n"
    
    # Höchste Priorität
    if hoechste:
        output += "## 🔥 HÖCHSTE PRIORITÄT - Sofort handeln!\n\n"
        for article in hoechste:
            output += format_article(article, "🔥")
    
    # Hohe Priorität
    if hohe:
        output += "## ⚡ HOHE PRIORITÄT - Wichtig für JuLis\n\n"
        for article in hohe:
            output += format_article(article, "⚡")
    
    return output

def format_article(article: dict, emoji: str) -> str:
    """Formatiere einzelnen Artikel"""
    output = f"### {emoji} {article.get('titel', 'Unbekannter Titel')}\n"
    output += f"**📄 Seite:** {article.get('seite', 'k.A.')}\n"
    output += f"**📍 Kernaussage:** {article.get('inhalt', 'Keine Zusammenfassung verfügbar')}\n"
    output += f"**🎯 JuLi-Relevanz:** {article.get('relevanz', 'Relevant für liberale Kommunalpolitik')}\n"
    output += "\n---\n\n"
    return output

def format_final_output(raw_output: str) -> str:
    """Formatiere die finale Ausgabe schöner"""
    # Zähle Artikel
    hoechste_count = raw_output.count("🔥")
    hohe_count = raw_output.count("⚡")
    
    # Füge Zusammenfassung hinzu
    summary = f"""# 📰 ANALYSE-ERGEBNIS

**Gefunden:** {hoechste_count + hohe_count} relevante Artikel
- 🔥 Höchste Priorität: {hoechste_count}
- ⚡ Hohe Priorität: {hohe_count}

---

"""
    
    return summary + raw_output

def show_login():
    """Login-Seite anzeigen"""
    st.title("🔐 JL Zeitungsanalyse - Login")
    
    with st.form("login_form"):
        username = st.text_input("👤 Benutzername:")
        password = st.text_input("🔒 Passwort:", type="password")
        submit = st.form_submit_button("🚀 Einloggen")
        
        if submit:
            if username in TEAM_CREDENTIALS and TEAM_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Falsche Anmeldedaten!")
    
    st.info("💡 **Demo-Zugang:** jl_team / junge_liberale_2025")

def analyze_tab():
    """Tab für neue Artikel-Analyse"""
    st.header("📤 PDF-Upload & Analyse")
    
    # API Setup mit besserer Behandlung
    api_key = None
    
    # Versuche API-Key aus Secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            st.sidebar.success("✅ API-Key aus Konfiguration geladen")
    except:
        pass
    
    # Falls nicht in Secrets, dann Sidebar-Eingabe
    if not api_key:
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
    
    # PDF Upload
    pdf_file = st.file_uploader(
        "📄 Zeitungs-PDF hochladen:",
        type=['pdf'],
        help="Lokale Zeitungen, Gemeindeblätter, etc."
    )
    
    if pdf_file:
        st.success(f"✅ **{pdf_file.name}** hochgeladen ({pdf_file.size:,} Bytes)")
        
        # PDF analysieren
        if st.button("🔍 Zeitung analysieren", type="primary", disabled=not api_key):
            if api_key:
                with st.spinner("📖 PDF wird gelesen..."):
                    text = extract_pdf_text(pdf_file)
                
                if text.strip():
                    with st.spinner("🤖 KI analysiert relevante Artikel..."):
                        analysis = analyze_with_gemini(text, api_key)
                    
                    # Ergebnis anzeigen
                    st.success("✅ Analyse abgeschlossen!")
                    st.markdown("---")
                    
                    # Analyse in schönem Format
                    st.markdown(analysis)
                    
                    # In Database speichern
                    if save_analysis_to_db(pdf_file.name, analysis, text):
                        st.success("💾 Artikel in Database gespeichert!")
                    
                    # Download-Option
                    st.download_button(
                        label="📥 Analyse als Markdown herunterladen",
                        data=f"# JL Zeitungsanalyse - {pdf_file.name}\n\nDatum: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n{analysis}",
                        file_name=f"JL_Analyse_{pdf_file.name.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("❌ Kein Text im PDF gefunden!")
            else:
                st.warning("⚠️ Bitte API-Key in der Sidebar eingeben!")

def search_tab():
    """Tab für Artikel-Suche in der Database"""
    st.header("🔍 Artikel-Database durchsuchen")
    
    # Database laden
    df = load_article_database()
    
    if df.empty:
        st.info("📭 Noch keine Artikel in der Database. Analysiere zuerst ein paar PDFs!")
        return
    
    # Suchfunktionen
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔎 Suche nach Themen, Stichwörtern oder PDF-Namen:",
            placeholder="z.B. Stadtrat, Digitalisierung, Verkehr..."
        )
    
    with col2:
        st.metric("📊 Gesamt-Artikel", len(df))
    
    # Zeitfilter
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("📅 Von:", value=None)
    with col2:
        date_to = st.date_input("📅 Bis:", value=None)
    
    # Suche ausführen
    filtered_df = df.copy()
    
    if search_query:
        filtered_df = search_articles(search_query, filtered_df)
    
    if date_from:
        filtered_df = filtered_df[pd.to_datetime(filtered_df['datum']).dt.date >= date_from]
    if date_to:
        filtered_df = filtered_df[pd.to_datetime(filtered_df['datum']).dt.date <= date_to]
    
    # Ergebnisse anzeigen
    st.markdown(f"### 📋 Gefunden: {len(filtered_df)} Artikel")
    
    if not filtered_df.empty:
        # Sortierung
        sort_by = st.selectbox("Sortieren nach:", ["Datum (neu→alt)", "PDF-Name", "Datum (alt→neu)"])
        
        if sort_by == "Datum (neu→alt)":
            filtered_df = filtered_df.sort_values('datum', ascending=False)
        elif sort_by == "PDF-Name":
            filtered_df = filtered_df.sort_values('pdf_name')
        else:
            filtered_df = filtered_df.sort_values('datum', ascending=True)
        
        # Artikel anzeigen
        for idx, row in filtered_df.iterrows():
            with st.container():
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 📰 {row['pdf_name']}")
                with col2:
                    st.markdown(f"📅 {pd.to_datetime(row['datum']).strftime('%d.%m.%Y')}")
                
                # Analyse in Container
                with st.container():
                    st.markdown(row['analyse'])
        
        # Export-Option
        st.markdown("---")
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Suchergebnisse als CSV exportieren",
            data=csv_data,
            file_name=f"JL_Artikel_Export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def stats_tab():
    """Tab für Statistiken"""
    st.header("📊 Artikel-Statistiken")
    
    df = load_article_database()
    
    if df.empty:
        st.info("📭 Noch keine Daten für Statistiken verfügbar.")
        return
    
    # Basis-Statistiken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📰 Gesamt-Artikel", len(df))
    
    with col2:
        if not df.empty:
            latest_date = pd.to_datetime(df['datum']).max().strftime('%d.%m.%Y')
            st.metric("📅 Letzte Analyse", latest_date)
    
    with col3:
        unique_pdfs = df['pdf_name'].nunique()
        st.metric("📄 Verschiedene PDFs", unique_pdfs)
    
    with col4:
        # Artikel mit hoher/höchster Priorität zählen
        high_prio_count = df['analyse'].str.contains('🔥|⚡', na=False).sum()
        st.metric("🎯 Relevante Artikel", high_prio_count)
    
    # Top Themen
    st.markdown("### 🏆 Top Themen in den Analysen")
    
    # Themen-Keywords definieren
    themes = {
        'Kommunalpolitik': ['Stadtrat', 'Bürgermeister', 'Gemeinderat', 'Kommune'],
        'Verkehr': ['Verkehr', 'ÖPNV', 'Mobilität', 'Straße', 'Radweg'],
        'Digitalisierung': ['Digital', 'Internet', 'Online', 'IT'],
        'Bildung': ['Schule', 'Bildung', 'Universität', 'Studium'],
        'Wirtschaft': ['Wirtschaft', 'Unternehmen', 'Gewerbe', 'Arbeitsplätze'],
        'Umwelt': ['Umwelt', 'Klima', 'Nachhaltigkeit', 'Energie']
    }
    
    theme_counts = {}
    all_text = ' '.join(df['analyse'].astype(str))
    
    for theme, keywords in themes.items():
        count = sum(all_text.lower().count(kw.lower()) for kw in keywords)
        if count > 0:
            theme_counts[theme] = count
    
    if theme_counts:
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Thema', 'Erwähnungen'])
        theme_df = theme_df.sort_values('Erwähnungen', ascending=False)
        st.bar_chart(theme_df.set_index('Thema'))
    
    # Zeitverlauf
    st.markdown("### 📈 Analysen im Zeitverlauf")
    df['datum_date'] = pd.to_datetime(df['datum']).dt.date
    timeline = df.groupby('datum_date').size().reset_index(name='Anzahl')
    if len(timeline) > 1:
        st.line_chart(timeline.set_index('datum_date'))

def main_app():
    """Hauptanwendung nach Login"""
    st.set_page_config(
        page_title="JL Zeitungsanalyse",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 JL Zeitungsanalyse für Kommunalpolitik")
    st.markdown("*Finde relevante Artikel für liberale Politik auf einen Blick*")
    
    # User Info & Logout
    col1, col2 = st.columns([6, 1])
    with col2:
        st.markdown(f"👤 {st.session_state.get('username', 'User')}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    # Tab-Navigation
    tab1, tab2, tab3 = st.tabs(["📤 Neue Analyse", "🔍 Artikel-Suche", "📊 Statistiken"])
    
    with tab1:
        analyze_tab()
    
    with tab2:
        search_tab()
        
    with tab3:
        stats_tab()

def main():
    """Hauptfunktion mit Session State Management"""
    
    # Page Config
    st.set_page_config(
        page_title="JL Zeitungsanalyse",
        page_icon="📰",
        layout="wide"
    )
    
    # Session State initialisieren
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # App-Routing
    if st.session_state.logged_in:
        main_app()
    else:
        show_login()

# App starten
if __name__ == "__main__":
    main()
