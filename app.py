import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
import io
import pandas as pd
from datetime import datetime
import hashlib

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
            page_texts.append(f"=== SEITE {page_num} ===\n{page_text}\n")
            text += page_text + "\n\n"
            
            # Debug: Zeige Text-Länge pro Seite
            st.write(f"Seite {page_num}: {len(page_text)} Zeichen")
        
        # Gesamt-Info
        st.success(f"✅ Extrahiert: {len(text)} Zeichen aus {total_pages} Seiten")
        
        # Zeige ersten Teil zur Kontrolle
        with st.expander("🔍 Extrahierter Text (erste 1000 Zeichen)"):
            st.text(text[:1000] + "..." if len(text) > 1000 else text)
        
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
            return analyze_single_chunk(text, model)
        else:
            # Langer Text - in Chunks aufteilen
            st.warning(f"⚠️ Text zu lang ({text_length} Zeichen) - wird in Teile aufgeteilt")
            return analyze_chunked_text(text, model, max_chunk_size)
        
    except Exception as e:
        return f"❌ **Analyse-Fehler:** {str(e)}"

def analyze_single_chunk(text: str, model) -> str:
    """Einzelnen Text-Chunk analysieren"""
    prompt = f"""
    AUFTRAG: Analysiere diesen Zeitungstext und kategorisiere alle gefundenen Artikel für die Jungen Liberalen.

    KATEGORIEN:
    🔥 HÖCHSTE PRIORITÄT (Sofort handeln):
    - Kommunalpolitik (Stadtrat, Bürgermeister, lokale Wahlen)
    - Wirtschaft & Gewerbe (Ansiedlungen, Arbeitsplätze, Startups)
    - Bildung (Schulen, Unis, Digitalisierung)
    - Verkehr & Infrastruktur (ÖPNV, Radwege, Straßen)

    ⚡ HOHE PRIORITÄT (Wichtig für JL):
    - Digitalisierung & Innovation
    - Umwelt & Nachhaltigkeit (pragmatische Lösungen)
    - Bürgerbeteiligung & Demokratie
    - Jugendthemen

    📰 STANDARD (Beobachten):
    - Kultur & Events
    - Sport
    - Soziales
    - Sonstiges

    FORMAT:
    Für jeden Artikel:
    **[KATEGORIE]** - Überschrift
    📍 Kurze Zusammenfassung (1-2 Sätze)
    🎯 JL-Relevanz: Warum wichtig für Junge Liberale
    ---

    TEXT:
    {text}
    """
    
    response = model.generate_content(prompt)
    return response.text

def analyze_chunked_text(text: str, model, chunk_size: int) -> str:
    """Langen Text in Chunks aufteilen und analysieren"""
    
    # Text in Chunks aufteilen (versuche bei Absätzen zu trennen)
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        
        # Versuche bei Absatz oder Satz zu trennen
        if end_pos < len(text):
            # Suche letzten Absatz in diesem Chunk
            last_paragraph = text.rfind('\n\n', current_pos, end_pos)
            if last_paragraph > current_pos:
                end_pos = last_paragraph
            else:
                # Falls kein Absatz, suche letzten Satz
                last_sentence = text.rfind('.', current_pos, end_pos)
                if last_sentence > current_pos:
                    end_pos = last_sentence + 1
        
        chunk = text[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)
        
        current_pos = end_pos
    
    st.info(f"📄 Text aufgeteilt in {len(chunks)} Teile")
    
    # Jeden Chunk einzeln analysieren
    all_analyses = []
    
    for i, chunk in enumerate(chunks, 1):
        st.write(f"🔍 Analysiere Teil {i}/{len(chunks)}...")
        
        chunk_prompt = f"""
        AUFTRAG: Analysiere diesen Zeitungstext-Teil und kategorisiere alle gefundenen Artikel für die Jungen Liberalen.
        WICHTIG: Dies ist Teil {i} von {len(chunks)} - analysiere nur die vollständigen Artikel in diesem Teil.

        KATEGORIEN:
        🔥 HÖCHSTE PRIORITÄT: Kommunalpolitik, Wirtschaft & Gewerbe, Bildung, Verkehr & Infrastruktur
        ⚡ HOHE PRIORITÄT: Digitalisierung & Innovation, Umwelt & Nachhaltigkeit, Bürgerbeteiligung & Demokratie, Jugendthemen
        📰 STANDARD: Kultur & Events, Sport, Soziales, Sonstiges

        FORMAT:
        **[KATEGORIE]** - Überschrift
        📍 Zusammenfassung
        🎯 JL-Relevanz
        ---

        TEXT TEIL {i}:
        {chunk}
        """
        
        try:
            response = model.generate_content(chunk_prompt)
            chunk_analysis = response.text
            all_analyses.append(f"## 📄 TEIL {i}/{len(chunks)}\n\n{chunk_analysis}")
        except Exception as e:
            all_analyses.append(f"## 📄 TEIL {i}/{len(chunks)}\n\n❌ Fehler bei Teil {i}: {e}")
    
    # Alle Analysen zusammenfassen
    final_analysis = f"""
# 📰 VOLLSTÄNDIGE ZEITUNGSANALYSE
*Analysiert in {len(chunks)} Teilen wegen Textlänge*

{chr(10).join(all_analyses)}

---
**📊 ZUSAMMENFASSUNG:** {len(chunks)} Teile analysiert, {len(text)} Zeichen Gesamttext
"""
    
    return final_analysis

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
        help="Lokale Zeitungen, Gemeindeblatts, etc."
    )
    
    if pdf_file:
        st.success(f"✅ **{pdf_file.name}** hochgeladen ({pdf_file.size} Bytes)")
        
        # PDF analysieren
        if st.button("🔍 Zeitung analysieren", type="primary"):
            if api_key:
                with st.spinner("📖 PDF wird gelesen..."):
                    text = extract_pdf_text(pdf_file)
                
                if text.strip():
                    with st.spinner("🤖 KI analysiert Artikel..."):
                        analysis = analyze_with_gemini(text, api_key)
                    
                    # Ergebnis anzeigen
                    st.success("✅ Analyse abgeschlossen!")
                    st.markdown("---")
                    st.markdown("## 📋 Analyseergebnis:")
                    st.markdown(analysis)
                    
                    # In Database speichern
                    if save_analysis_to_db(pdf_file.name, analysis, text):
                        st.success("💾 Artikel in Database gespeichert!")
                    else:
                        st.info("ℹ️ Artikel bereits in Database vorhanden")
                    
                    # Download-Option
                    st.download_button(
                        label="📥 Analyse herunterladen",
                        data=f"# JL Zeitungsanalyse - {pdf_file.name}\n\n{analysis}",
                        file_name=f"JL_Analyse_{pdf_file.name}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("❌ Kein Text im PDF gefunden!")
            else:
                st.warning("⚠️ API-Key benötigt!")

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
        sort_by = st.selectbox("Sortieren nach:", ["Datum (neu-alt)", "PDF-Name", "Datum (alt-neu)"])
        
        if sort_by == "Datum (neu-alt)":
            filtered_df = filtered_df.sort_values('datum', ascending=False)
        elif sort_by == "PDF-Name":
            filtered_df = filtered_df.sort_values('pdf_name')
        else:
            filtered_df = filtered_df.sort_values('datum', ascending=True)
        
        # Artikel anzeigen - OHNE NESTED EXPANDERS
        for idx, row in filtered_df.iterrows():
            st.markdown("---")
            st.markdown(f"### 📰 {row['pdf_name']}")
            st.markdown(f"**📅 Datum:** {row['datum']}")
            st.markdown("**🔍 Analyse:**")
            st.markdown(row['analyse'])
            
            # Volltext als Button + Text Area (NICHT als Expander)
            if st.button(f"📖 Volltext anzeigen/verstecken", key=f"toggle_{idx}"):
                if f"show_text_{idx}" not in st.session_state:
                    st.session_state[f"show_text_{idx}"] = True
                else:
                    st.session_state[f"show_text_{idx}"] = not st.session_state[f"show_text_{idx}"]
            
            if st.session_state.get(f"show_text_{idx}", False):
                st.markdown("**📄 Volltext-Vorschau:**")
                st.text_area("", value=row['volltext_kurz'], height=150, key=f"text_{idx}", disabled=True)
        
        # Export-Option
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Suchergebnisse als CSV exportieren",
            data=csv_data,
            file_name=f"JL_Artikel_Export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("🔍 Keine Artikel gefunden. Versuche andere Suchbegriffe.")

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
        avg_per_day = len(df) / max(1, (pd.to_datetime(df['datum']).max() - pd.to_datetime(df['datum']).min()).days + 1) if len(df) > 1 else len(df)
        st.metric("📈 Ø Artikel/Tag", f"{avg_per_day:.1f}")
    
    # Top PDFs
    st.markdown("### 🏆 Meistanalysierte Zeitungen")
    pdf_counts = df['pdf_name'].value_counts().head(10)
    if not pdf_counts.empty:
        st.bar_chart(pdf_counts)
    
    # Keyword-Analyse (einfach)
    st.markdown("### 🔤 Häufige Begriffe in Analysen")
    all_text = ' '.join(df['analyse'].astype(str))
    keywords = ['Stadtrat', 'Bürgermeister', 'Verkehr', 'Digitalisierung', 'Bildung', 'Wirtschaft', 'Umwelt', 'Kultur']
    keyword_counts = {kw: all_text.lower().count(kw.lower()) for kw in keywords}
    keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Häufigkeit'])
    keyword_df = keyword_df[keyword_df['Häufigkeit'] > 0].sort_values('Häufigkeit', ascending=False)
    
    if not keyword_df.empty:
        st.bar_chart(keyword_df.set_index('Keyword'))

def main_app():
    """Hauptanwendung nach Login"""
    st.title("📰 JL Zeitungsanalyse")
    st.markdown("*Automatische Kategorisierung für Kommunalpolitik*")
    
    # Logout Button
    if st.button("🚪 Logout"):
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
