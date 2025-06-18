import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
import io
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import re
import requests
import time
import base64

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
    AUFTRAG: Analysiere diesen Zeitungstext und finde NUR LOKALE/REGIONALE Artikel für die Jungen Liberalen.
    
    WICHTIG: 
    - NUR Artikel mit Bezug zu DESSAU-ROßLAU oder SACHSEN-ANHALT
    - IGNORIERE Bundespolitik, internationale Themen, andere Bundesländer
    - Zeige NUR Artikel mit HÖCHSTER oder HOHER Priorität
    - Extrahiere IMMER die Seitenzahl aus [SEITE X] Markierungen
    
    HÖCHSTE PRIORITÄT (🔥) - NUR LOKAL/REGIONAL:
    - Dessau-Roßlauer Stadtrat & Kommunalpolitik
    - Lokale Wirtschaft & Gewerbeansiedlungen in Dessau-Roßlau
    - Schulen & Bildung in Dessau-Roßlau und Sachsen-Anhalt
    - Lokaler Verkehr & Infrastruktur (Straßen, ÖPNV in Dessau)
    - Landespolitik Sachsen-Anhalt
    
    HOHE PRIORITÄT (⚡) - NUR LOKAL/REGIONAL:
    - Digitalisierung in Dessau-Roßlau
    - Lokale Umwelt- & Nachhaltigkeitsprojekte
    - Bürgerbeteiligung in Dessau-Roßlau
    - Jugendthemen in der Region
    
    IGNORIERE KOMPLETT:
    - Bundespolitik (Bundestag, Bundesregierung, etc.)
    - Internationale Themen
    - Andere Städte/Bundesländer (außer Sachsen-Anhalt)
    - Sport, Kultur (außer mit politischer Relevanz)
    
    FORMAT FÜR JEDEN ARTIKEL:
    ### [EMOJI] Überschrift des Artikels
    **Seite:** [Nummer]
    **Kernaussage:** [1-2 Sätze - Was ist die wichtigste Information?]
    **JuLi-Relevanz:** [1 Satz - Warum sollten JuLis hier aktiv werden?]
    
    ---
    
    GEBE NUR LOKALE/REGIONALE ARTIKEL AUS!
    
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
            AUFTRAG: Extrahiere NUR LOKALE/REGIONALE Artikel aus diesem Zeitungstext-Teil für die Jungen Liberalen.
            
            WICHTIG: 
            - NUR Artikel über DESSAU-ROßLAU oder SACHSEN-ANHALT
            - KEINE Bundespolitik oder internationale Themen
            - Nur HÖCHSTE und HOHE Priorität
            - Seitenzahlen aus [SEITE X] extrahieren
            
            HÖCHSTE PRIORITÄT: Dessau-Roßlauer Stadtrat, lokale Wirtschaft, Schulen in Dessau, Verkehr in Dessau, Landespolitik Sachsen-Anhalt
            HOHE PRIORITÄT: Digitalisierung in Dessau, lokale Umweltprojekte, Bürgerbeteiligung Dessau, regionale Jugendthemen
            
            IGNORIERE: Bundespolitik, andere Städte/Länder, Sport, Kultur
            
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
        return "❌ Keine relevanten lokalen/regionalen Artikel gefunden."
    
    # Sortiere nach Priorität
    hoechste = [a for a in articles if 'Höchste' in a.get('kategorie', '')]
    hohe = [a for a in articles if 'Hohe' in a.get('kategorie', '')]
    
    output = "# 📰 ANALYSE-ERGEBNIS - DESSAU-ROßLAU & SACHSEN-ANHALT\n\n"
    
    # Zusammenfassung
    output += f"**Gefunden:** {len(articles)} relevante lokale/regionale Artikel\n"
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
    summary = f"""# 📰 ANALYSE-ERGEBNIS - DESSAU-ROßLAU & SACHSEN-ANHALT

**Gefunden:** {hoechste_count + hohe_count} relevante lokale/regionale Artikel
- 🔥 Höchste Priorität: {hoechste_count}
- ⚡ Hohe Priorität: {hohe_count}

🎯 **Fokus:** Nur Dessau-Roßlau und Sachsen-Anhalt
❌ **Ignoriert:** Bundespolitik & internationale Themen

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

def automated_analysis_tab():
    """Tab für automatisierte Google Drive Analyse"""
    st.header("🤖 Automatisierte Zeitungsanalyse")
    
    # Eingebettete URL
    DEFAULT_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbye1Pj2dOmadeRvFcZqLV-mIEGTcgwvxunVOncoRBBDQQUWfOngRfluEwYJew-cCiQ/exec"
    
    # Einfache Version - nur neueste PDF
    st.markdown("### 📄 Neueste Zeitung automatisch analysieren")
    
    st.info("""
    ✅ **Google Drive Integration aktiv!**
    
    Die App ist bereits mit deinem Zeitungsordner verbunden und kann:
    - Die neueste PDF automatisch erkennen
    - Sie herunterladen und analysieren
    - Duplikate vermeiden (bereits analysierte PDFs überspringen)
    """)
    
    # URL ist voreingestellt aber änderbar
    web_app_url = st.text_input(
        "Web App URL (bereits konfiguriert):",
        value=st.session_state.get('web_app_url', DEFAULT_WEB_APP_URL),
        help="Die Standard-URL ist bereits eingestellt. Nur ändern wenn nötig!"
    )
    
    if web_app_url:
        st.session_state.web_app_url = web_app_url
        
        # Hauptaktionen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Neueste PDF anzeigen", type="primary", use_container_width=True):
                check_newest_pdf(web_app_url)
        
        with col2:
            if st.button("🚀 Jetzt analysieren", type="secondary", use_container_width=True):
                analyze_newest_pdf(web_app_url)
                
        with col3:
            if st.button("🔄 Auto-Check (stündlich)", type="secondary", use_container_width=True):
                st.session_state.auto_check = True
                auto_check_loop(web_app_url)
        
        # Erweiterte Optionen
        with st.expander("⚙️ Erweiterte Optionen"):
            st.markdown("### 📅 Zeitraum-Analyse")
            
            days_back = st.slider(
                "Analysiere PDFs der letzten X Tage:",
                min_value=1,
                max_value=30,
                value=7,
                help="Analysiert alle PDFs aus diesem Zeitraum"
            )
            
            if st.button("📚 Mehrere PDFs analysieren"):
                analyze_recent_pdfs(web_app_url, days_back)
            
            st.markdown("---")
            
            st.markdown("### 🔧 Script-Verwaltung")
            
            if st.button("🧪 Verbindung testen"):
                test_connection(web_app_url)
            
            st.markdown(f"""
            **Script-Editor öffnen:**
            [Google Apps Script Editor →](https://script.google.com)
            
            **Aktuelle Web App URL:**
            ```
            {web_app_url}
            ```
            """)

def check_newest_pdf(web_app_url):
    """Zeige Info über die neueste PDF"""
    try:
        with st.spinner("Prüfe neueste PDF..."):
            response = requests.get(web_app_url)
            data = response.json()
            
            if data.get('success'):
                file_info = data['file']
                modified = datetime.fromisoformat(file_info['modified'].replace('Z', '+00:00'))
                
                # Info-Box mit schönem Format
                info_col1, info_col2 = st.columns([2, 1])
                
                with info_col1:
                    st.success(f"""
                    ### 📄 Neueste Zeitung gefunden:
                    **{file_info['name']}**
                    
                    📅 Hochgeladen: {modified.strftime('%d.%m.%Y um %H:%M Uhr')}
                    """)
                
                with info_col2:
                    # Prüfe ob bereits analysiert
                    df = load_article_database()
                    if not df.empty and file_info['name'] in df['pdf_name'].values:
                        st.info("✅ Bereits analysiert")
                    else:
                        st.warning("⚠️ Noch nicht analysiert")
                    
            else:
                st.error(f"Fehler: {data.get('error')}")
                
    except Exception as e:
        st.error(f"Verbindungsfehler: {e}")

def analyze_newest_pdf(web_app_url):
    """Analysiere nur die neueste PDF"""
    try:
        # API Key prüfen
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("❌ Gemini API Key fehlt! Bitte in Streamlit Secrets hinzufügen.")
            with st.expander("🔑 So fügst du den API Key hinzu:"):
                st.markdown("""
                1. Klicke auf **Manage app** (unten rechts)
                2. Gehe zu **Settings** → **Secrets**
                3. Füge hinzu:
                ```toml
                GEMINI_API_KEY = "dein-api-key-hier"
                ```
                4. Speichern und App neu starten
                """)
            return
        
        # Progress Container
        progress_container = st.container()
        
        with progress_container:
            # Hole neueste PDF Info
            with st.spinner("🔍 Suche neueste PDF..."):
                response = requests.get(web_app_url)
                data = response.json()
                
                if not data.get('success'):
                    st.error(f"Fehler: {data.get('error')}")
                    return
                
                file_info = data['file']
                
                # Prüfe ob bereits analysiert
                df = load_article_database()
                if not df.empty and file_info['name'] in df['pdf_name'].values:
                    st.warning(f"⚠️ '{file_info['name']}' wurde bereits analysiert!")
                    if not st.checkbox("Trotzdem erneut analysieren?"):
                        return
            
            # Status-Updates
            status = st.empty()
            progress_bar = st.progress(0)
            
            # Download PDF
            status.text("📥 Lade PDF herunter...")
            progress_bar.progress(25)
            
            download_response = requests.post(
                web_app_url,
                data={'fileId': file_info['id']},
                timeout=60
            )
            
            if download_response.status_code != 200:
                st.error(f"Download-Fehler: HTTP {download_response.status_code}")
                return
            
            pdf_content = base64.b64decode(download_response.text)
            
            # PDF Buffer erstellen
            pdf_buffer = io.BytesIO(pdf_content)
            pdf_buffer.name = file_info['name']
            
            # Text extrahieren
            status.text("📖 Extrahiere Text aus PDF...")
            progress_bar.progress(50)
            
            text = extract_pdf_text(pdf_buffer)
            
            if not text.strip():
                st.error("❌ Kein Text im PDF gefunden!")
                return
            
            # Analysieren
            status.text("🤖 KI analysiert relevante Artikel...")
            progress_bar.progress(75)
            
            analysis = analyze_with_gemini(text, api_key)
            
            # Speichern
            save_analysis_to_db(file_info['name'], analysis, text)
            
            progress_bar.progress(100)
            status.text("✅ Analyse abgeschlossen!")
        
        # Ergebnis anzeigen
        st.success(f"✅ **{file_info['name']}** erfolgreich analysiert!")
        st.markdown("---")
        
        # Analyse in schönem Format
        with st.container():
            st.markdown(analysis)
        
        # Download-Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Als Markdown speichern",
                data=f"# JL Zeitungsanalyse\n\n**Datei:** {file_info['name']}\n**Datum:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n{analysis}",
                file_name=f"JL_Analyse_{file_info['name'].replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Als PDF speichern (später implementieren)
            st.button("📄 Als PDF speichern", disabled=True, use_container_width=True, help="Kommt bald!")
        
    except Exception as e:
        st.error(f"Fehler: {e}")
        with st.expander("🐛 Fehlerdetails"):
            import traceback
            st.code(traceback.format_exc())

def analyze_recent_pdfs(web_app_url, days):
    """Analysiere alle PDFs der letzten X Tage"""
    st.info(f"📅 Suche PDFs der letzten {days} Tage...")
    
    # Hier würde die Logik für mehrere PDFs kommen
    # Für jetzt zeigen wir nur eine Info
    st.warning("Diese Funktion ist noch in Entwicklung!")
    
def test_connection(web_app_url):
    """Teste die Verbindung zum Google Apps Script"""
    try:
        with st.spinner("Teste Verbindung..."):
            response = requests.get(web_app_url)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if response.status_code == 200:
                    st.success("✅ Verbindung erfolgreich!")
                else:
                    st.error(f"❌ HTTP Fehler: {response.status_code}")
            
            with col2:
                try:
                    data = response.json()
                    if data.get('success'):
                        st.success("✅ Script funktioniert!")
                    else:
                        st.warning("⚠️ Script-Fehler")
                except:
                    st.error("❌ Keine gültige Antwort")
            
            with st.expander("🔍 Technische Details"):
                st.json(response.json() if response.status_code == 200 else {"error": response.text})
                
    except Exception as e:
        st.error(f"Verbindungsfehler: {e}")

def auto_check_loop(web_app_url):
    """Automatische stündliche Überprüfung"""
    st.info("🔄 Automatische Überprüfung aktiviert - prüft stündlich auf neue PDFs")
    
    # Placeholder für zukünftige Implementation
    st.warning("Diese Funktion läuft nur solange die App geöffnet ist. Für echte Automatisierung nutze einen Cron-Job oder GitHub Actions.")

def check_newest_pdf(web_app_url):
    """Zeige Info über die neueste PDF"""
    try:
        with st.spinner("Prüfe neueste PDF..."):
            response = requests.get(web_app_url)
            data = response.json()
            
            if data.get('success'):
                file_info = data['file']
                modified = datetime.fromisoformat(file_info['modified'].replace('Z', '+00:00'))
                
                st.success(f"""
                ### 📄 Neueste PDF gefunden:
                - **Name:** {file_info['name']}
                - **Geändert:** {modified.strftime('%d.%m.%Y %H:%M')}
                - **ID:** {file_info['id']}
                """)
                
                # Prüfe ob bereits analysiert
                df = load_article_database()
                if not df.empty and file_info['name'] in df['pdf_name'].values:
                    st.info("ℹ️ Diese PDF wurde bereits analysiert!")
                else:
                    st.warning("⚠️ Diese PDF wurde noch nicht analysiert!")
                    
            else:
                st.error(f"Fehler: {data.get('error')}")
                
    except Exception as e:
        st.error(f"Fehler: {e}")

def analyze_newest_pdf(web_app_url):
    """Analysiere nur die neueste PDF"""
    try:
        # API Key prüfen
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("❌ Gemini API Key fehlt!")
            return
        
        # Hole neueste PDF Info
        with st.spinner("Hole neueste PDF..."):
            response = requests.get(web_app_url)
            data = response.json()
            
            if not data.get('success'):
                st.error(f"Fehler: {data.get('error')}")
                return
            
            file_info = data['file']
            st.info(f"📄 Verarbeite: {file_info['name']}")
        
        # Download PDF
        with st.spinner("Lade PDF herunter..."):
            download_response = requests.post(
                web_app_url,
                data={'fileId': file_info['id']},
                timeout=60
            )
            
            if download_response.status_code != 200:
                st.error(f"Download-Fehler: HTTP {download_response.status_code}")
                return
            
            # Prüfe ob Fehler-JSON
            try:
                error_check = download_response.json()
                if 'error' in error_check:
                    st.error(f"Script-Fehler: {error_check['error']}")
                    return
            except:
                # Kein JSON = gut (Base64 Content)
                pass
            
            pdf_content = base64.b64decode(download_response.text)
            st.success(f"✅ Download erfolgreich ({len(pdf_content):,} Bytes)")
        
        # PDF Buffer erstellen
        pdf_buffer = io.BytesIO(pdf_content)
        pdf_buffer.name = file_info['name']
        
        # Text extrahieren
        with st.spinner("Extrahiere Text..."):
            text = extract_pdf_text(pdf_buffer)
            
            if not text.strip():
                st.error("❌ Kein Text im PDF gefunden!")
                return
        
        # Analysieren
        with st.spinner("🤖 KI analysiert Artikel..."):
            analysis = analyze_with_gemini(text, api_key)
        
        # Speichern
        if save_analysis_to_db(file_info['name'], analysis, text):
            st.success("💾 In Database gespeichert!")
        
        # Ergebnis anzeigen
        st.success("✅ Analyse abgeschlossen!")
        st.markdown("---")
        st.markdown(analysis)
        
        # Download
        st.download_button(
            label="📥 Analyse herunterladen",
            data=f"# JL Zeitungsanalyse\n\n**Datei:** {file_info['name']}\n**Datum:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n{analysis}",
            file_name=f"JL_Analyse_{file_info['name'].replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        
    except Exception as e:
        st.error(f"Fehler: {e}")
        import traceback
        st.code(traceback.format_exc())

def apps_script_integration():
    """Integration über Google Apps Script für private Ordner"""
    st.markdown("### 🔐 Private Ordner-Integration mit Google Apps Script")
    
    with st.expander("📖 Einrichtungsanleitung", expanded=True):
        st.markdown("""
        **So richtest du Google Apps Script ein:**
        
        1. **Öffne Google Drive** und gehe zu deinem Zeitungsordner
        2. **Erstelle ein neues Apps Script**: Rechtsklick → Mehr → Google Apps Script
        3. **Lösche allen vorhandenen Code und kopiere diesen Code:**
        
        ```javascript
        function doGet(e) {
          try {
            // WICHTIG: Ersetze diese ID mit deiner Ordner-ID!
            const folderId = 'DEINE_ORDNER_ID_HIER';
            
            const folder = DriveApp.getFolderById(folderId);
            const files = folder.getFilesByType(MimeType.PDF);
            
            const fileList = [];
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - 7); // Letzte 7 Tage
            
            while (files.hasNext()) {
              const file = files.next();
              if (file.getLastUpdated() > cutoffDate) {
                fileList.push({
                  id: file.getId(),
                  name: file.getName(),
                  modified: file.getLastUpdated().toISOString()
                });
              }
            }
            
            // Wichtig: Als JSON zurückgeben
            return ContentService
              .createTextOutput(JSON.stringify(fileList))
              .setMimeType(ContentService.MimeType.JSON);
              
          } catch (error) {
            // Fehler als JSON zurückgeben
            return ContentService
              .createTextOutput(JSON.stringify({
                error: error.toString(),
                message: "Fehler beim Abrufen der Dateien"
              }))
              .setMimeType(ContentService.MimeType.JSON);
          }
        }
        
        function doPost(e) {
          try {
            const fileId = e.parameter.fileId;
            const file = DriveApp.getFileById(fileId);
            const content = file.getBlob().getBytes();
            
            return ContentService
              .createTextOutput(Utilities.base64Encode(content))
              .setMimeType(ContentService.MimeType.TEXT);
              
          } catch (error) {
            return ContentService
              .createTextOutput(JSON.stringify({
                error: error.toString()
              }))
              .setMimeType(ContentService.MimeType.JSON);
          }
        }
        ```
        
        4. **Finde deine Ordner-ID:**
           - Öffne deinen Zeitungsordner
           - URL: `https://drive.google.com/drive/folders/XXXXX`
           - Kopiere `XXXXX` (das ist deine Ordner-ID)
           - Ersetze `DEINE_ORDNER_ID_HIER` im Script
        
        5. **Speichern und Deploy:**
           - Strg+S zum Speichern
           - Deploy → New Deployment
           - Type: **Web app**
           - Execute as: **Me**
           - Who has access: **Anyone**
           - Deploy klicken
           
        6. **WICHTIG: Autorisierung**
           - Beim ersten Deploy: "Review Permissions" klicken
           - Wähle dein Google-Konto
           - "Advanced" → "Go to [Scriptname] (unsafe)"
           - "Allow" klicken
           
        7. **Kopiere die Web App URL** (beginnt mit https://script.google.com/macros/s/...)
        
        **Teste die URL:**
        - Öffne die URL in einem neuen Browser-Tab
        - Du solltest JSON-Daten sehen (Array mit Dateien)
        - Wenn Fehler: Überprüfe die Ordner-ID
        """)
    
    # Test-Button für direkte URL
    st.markdown("### 🧪 Script testen")
    
    test_url = st.text_input(
        "Teste deine Web App URL direkt:",
        placeholder="https://script.google.com/macros/s/.../exec"
    )
    
    if test_url and st.button("🔍 URL testen"):
        try:
            with st.spinner("Teste URL..."):
                response = requests.get(test_url)
                
                st.info(f"HTTP Status: {response.status_code}")
                
                # Zeige Raw Response
                with st.expander("Raw Response anzeigen"):
                    st.code(response.text[:1000])
                
                # Versuche als JSON zu parsen
                try:
                    data = response.json()
                    st.success("✅ Gültiges JSON erkannt!")
                    
                    if isinstance(data, list):
                        st.write(f"Gefundene Dateien: {len(data)}")
                        if data:
                            st.write("Erste Datei:", data[0])
                    elif 'error' in data:
                        st.error(f"Script-Fehler: {data['error']}")
                    else:
                        st.json(data)
                        
                except:
                    st.error("❌ Kein gültiges JSON. Überprüfe das Script!")
                    
        except Exception as e:
            st.error(f"Verbindungsfehler: {e}")
    
    # Web App URL eingeben
    web_app_url = st.text_input(
        "Google Apps Script Web App URL:",
        placeholder="https://script.google.com/macros/s/.../exec",
        help="Die URL aus dem Deployment"
    )
    
    if web_app_url:
        if st.button("📂 Dateien abrufen", type="primary"):
            fetch_and_analyze_apps_script(web_app_url)

def fetch_and_analyze_apps_script(web_app_url):
    """Hole und analysiere Dateien über Apps Script"""
    try:
        # API Key prüfen
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("❌ Gemini API Key fehlt in den Secrets!")
            return
        
        # Dateien abrufen
        with st.spinner("📂 Hole Dateiliste..."):
            response = requests.get(web_app_url)
            
            # Debug: Zeige Raw Response
            if response.status_code != 200:
                st.error(f"❌ HTTP Fehler: {response.status_code}")
                st.code(response.text[:500])
                return
            
            # Versuche JSON zu parsen
            try:
                files = response.json()
            except:
                st.error("❌ Antwort ist kein gültiges JSON")
                st.code(response.text[:500])
                
                # Versuche HTML-Fehler zu erkennen
                if "<!DOCTYPE" in response.text or "<html" in response.text:
                    st.error("""
                    ⚠️ Das Script gibt HTML zurück statt JSON.
                    
                    **Mögliche Ursachen:**
                    1. Die Web App URL ist falsch
                    2. Das Script wurde nicht deployed
                    3. Zugriffsfehler
                    
                    **Lösung:** 
                    - Überprüfe die URL
                    - Stelle sicher, dass das Script deployed ist
                    - Teste die URL direkt im Browser
                    """)
                return
        
        if not files:
            st.warning("Keine Dateien gefunden")
            return
            
        st.success(f"✅ {len(files)} PDF(s) gefunden")
        
        # Zeige Dateien
        with st.expander("📋 Gefundene Dateien"):
            for file in files:
                st.write(f"📄 {file.get('name', 'Unbekannt')} ({file.get('size', 0):,} Bytes)")
        
        # Container für Live-Updates
        status_container = st.container()
        
        # Analyse-Button direkt sichtbar
        if st.button("🚀 Alle Dateien analysieren", type="primary", key="analyze_all"):
            
            # Progress-Elemente
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()
            
            successful = 0
            failed = 0
            all_analyses = []
            
            st.write("📊 Starte Analyse...")
            
            for idx, file in enumerate(files):
                progress = (idx + 1) / len(files)
                progress_bar.progress(progress)
                
                file_name = file.get('name', 'Unbekannt')
                status_text.text(f"Verarbeite {idx + 1}/{len(files)}: {file_name}")
                
                with log_container:
                    with st.expander(f"📄 {file_name}", expanded=True):
                        try:
                            # Schritt 1: Download
                            st.write("⏳ Lade PDF herunter...")
                            
                            download_response = requests.post(
                                web_app_url,
                                data={'fileId': file['id']},
                                timeout=30
                            )
                            
                            if download_response.status_code != 200:
                                st.error(f"❌ Download-Fehler: HTTP {download_response.status_code}")
                                failed += 1
                                continue
                            
                            # Check if response is JSON (error)
                            if download_response.headers.get('content-type', '').startswith('application/json'):
                                error_data = download_response.json()
                                st.error(f"❌ Script-Fehler: {error_data.get('error', 'Unbekannter Fehler')}")
                                failed += 1
                                continue
                            
                            st.write(f"✅ Download erfolgreich ({len(download_response.content):,} Bytes)")
                            
                            # Schritt 2: Base64 Decode
                            try:
                                pdf_content = base64.b64decode(download_response.text)
                                st.write(f"✅ Dekodierung erfolgreich ({len(pdf_content):,} Bytes)")
                            except Exception as e:
                                st.error(f"❌ Base64-Dekodierung fehlgeschlagen: {e}")
                                failed += 1
                                continue
                            
                            # Schritt 3: PDF Buffer erstellen
                            pdf_buffer = io.BytesIO(pdf_content)
                            pdf_buffer.name = file_name
                            
                            # Schritt 4: Text extrahieren
                            st.write("⏳ Extrahiere Text aus PDF...")
                            text = extract_pdf_text(pdf_buffer)
                            
                            if not text.strip():
                                st.warning("⚠️ Kein Text im PDF gefunden")
                                failed += 1
                                continue
                            
                            st.write(f"✅ Text extrahiert ({len(text):,} Zeichen)")
                            
                            # Schritt 5: Analysieren
                            st.write("🤖 Analysiere mit KI...")
                            analysis = analyze_with_gemini(text, api_key)
                            st.write("✅ Analyse abgeschlossen")
                            
                            # Schritt 6: Speichern
                            if save_analysis_to_db(file_name, analysis, text):
                                st.write("💾 In Datenbank gespeichert")
                            
                            all_analyses.append({
                                'filename': file_name,
                                'date': datetime.now().strftime('%d.%m.%Y %H:%M'),
                                'analysis': analysis
                            })
                            
                            successful += 1
                            st.success(f"✅ Erfolgreich analysiert!")
                            
                            # Kurze Pause für API Rate Limits
                            time.sleep(1)
                            
                        except requests.Timeout:
                            st.error("❌ Zeitüberschreitung beim Download")
                            failed += 1
                        except Exception as e:
                            st.error(f"❌ Unerwarteter Fehler: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            failed += 1
            
            # Finale Zusammenfassung
            progress_bar.progress(1.0)
            status_text.text("✅ Batch-Analyse abgeschlossen!")
            
            st.markdown("---")
            st.success(f"""
            ### 📊 Zusammenfassung:
            - ✅ **Erfolgreich:** {successful} PDFs
            - ❌ **Fehlgeschlagen:** {failed} PDFs
            - 📁 **Gesamt:** {len(files)} PDFs
            """)
            
            # Bericht erstellen wenn Analysen vorhanden
            if all_analyses:
                report = create_batch_report(all_analyses)
                
                st.markdown("---")
                with st.expander("📄 Gesamtbericht anzeigen", expanded=True):
                    st.markdown(report)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Bericht als Markdown",
                        data=report,
                        file_name=f"JL_Batch_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # CSV Export
                    df = pd.DataFrame(all_analyses)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Ergebnisse als CSV",
                        data=csv,
                        file_name=f"JL_Analysen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Keine erfolgreichen Analysen zum Exportieren")
                
    except Exception as e:
        st.error(f"Kritischer Fehler: {e}")
        import traceback
        st.code(traceback.format_exc())

def public_folder_integration():
    """Integration für öffentliche Google Drive Ordner"""
    st.markdown("### 📂 Öffentlicher Ordner (Vereinfachte Methode)")
    
    st.info("""
    ⚠️ Diese Methode funktioniert nur mit öffentlich freigegebenen Ordnern!
    
    Für private Ordner nutze bitte die Google Apps Script Methode oben.
    """)
    
    gdrive_url = st.text_input(
        "Öffentlicher Google Drive Ordner URL:",
        placeholder="https://drive.google.com/drive/folders/..."
    )
    
    if gdrive_url:
        st.warning("Diese Funktion ist noch nicht implementiert für private Ordner.")
        st.info("Bitte nutze die Google Apps Script Methode oben!")

def manual_batch_upload():
    """Manueller Batch-Upload von mehreren PDFs"""
    st.markdown("### 📤 Manueller Batch-Upload")
    
    # API Key prüfen
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("❌ Gemini API Key fehlt in den Secrets!")
        return
    
    # Mehrere PDFs hochladen
    pdf_files = st.file_uploader(
        "Wähle mehrere PDFs aus:",
        type=['pdf'],
        accept_multiple_files=True,
        help="Du kannst mehrere Dateien gleichzeitig auswählen"
    )
    
    if pdf_files:
        st.success(f"✅ {len(pdf_files)} PDFs ausgewählt")
        
        # Liste anzeigen
        with st.expander("📋 Ausgewählte Dateien"):
            for pdf in pdf_files:
                st.write(f"📄 {pdf.name} ({pdf.size:,} Bytes)")
        
        # Analyse starten
        if st.button("🚀 Batch-Analyse starten", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            successful = 0
            all_analyses = []
            
            for idx, pdf_file in enumerate(pdf_files):
                progress = (idx + 1) / len(pdf_files)
                progress_bar.progress(progress)
                status_text.text(f"Analysiere: {pdf_file.name}")
                
                try:
                    # PDF lesen
                    pdf_file.seek(0)
                    text = extract_pdf_text(pdf_file)
                    
                    if text.strip():
                        # Analysieren
                        analysis = analyze_with_gemini(text, api_key)
                        
                        # Speichern
                        save_analysis_to_db(pdf_file.name, analysis, text)
                        
                        all_analyses.append({
                            'filename': pdf_file.name,
                            'date': datetime.now().strftime('%d.%m.%Y %H:%M'),
                            'analysis': analysis
                        })
                        
                        successful += 1
                        
                except Exception as e:
                    st.error(f"Fehler bei {pdf_file.name}: {e}")
            
            # Ergebnis
            progress_bar.progress(1.0)
            status_text.text("✅ Batch-Analyse abgeschlossen!")
            
            st.success(f"""
            ### 📊 Ergebnis:
            - ✅ Erfolgreich: {successful} PDFs
            - ❌ Fehlgeschlagen: {len(pdf_files) - successful} PDFs
            """)
            
            # Bericht
            if all_analyses:
                report = create_batch_report(all_analyses)
                
                with st.expander("📄 Gesamtbericht", expanded=True):
                    st.markdown(report)
                
                st.download_button(
                    label="📥 Bericht herunterladen",
                    data=report,
                    file_name=f"JL_Batch_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )

def create_batch_report(analyses):
    """Erstelle Batch-Analyse Bericht"""
    report = f"""# 🤖 JL BATCH-ANALYSE BERICHT

**Datum:** {datetime.now().strftime('%d.%m.%Y %H:%M')}
**Anzahl Zeitungen:** {len(analyses)}

---

## 📊 ZUSAMMENFASSUNG

"""
    
    # Zähle Prioritäten
    total_highest = 0
    total_high = 0
    
    for analysis_data in analyses:
        total_highest += analysis_data['analysis'].count('🔥')
        total_high += analysis_data['analysis'].count('⚡')
    
    report += f"""
- 🔥 **Höchste Priorität gesamt:** {total_highest} Artikel
- ⚡ **Hohe Priorität gesamt:** {total_high} Artikel
- 📰 **Analysierte Zeitungen:** {len(analyses)}

---

## 📰 EINZELANALYSEN
"""
    
    for analysis_data in analyses:
        report += f"\n### 📄 {analysis_data['filename']}\n"
        report += f"*Analysiert: {analysis_data['date']}*\n\n"
        report += analysis_data['analysis']
        report += "\n\n---\n"
    
    return report

def main_app():
    """Hauptanwendung nach Login"""
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
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Neue Analyse", "🔍 Artikel-Suche", "📊 Statistiken", "🤖 Automatisierung"])
    
    with tab1:
        analyze_tab()
    
    with tab2:
        search_tab()
        
    with tab3:
        stats_tab()
    
    with tab4:
        # Importiere die Automatisierungs-Funktionen
        automated_analysis_tab()

def main():
    """Hauptfunktion mit Session State Management"""
    
    # Page Config MUSS als allererstes kommen
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
