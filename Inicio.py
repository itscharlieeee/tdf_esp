import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("💻 Demo TF-IDF en Español: Tecnología y Trabajo Remoto")

# Documentos de ejemplo
default_docs = """El trabajo remoto permite a los empleados laborar desde casa con flexibilidad.
Las reuniones virtuales se realizan a través de plataformas como Zoom o Teams.
El equipo de desarrollo colabora en proyectos mediante GitHub.
La ciberseguridad es esencial para proteger la información de la empresa.
Muchos trabajadores usan escritorios ergonómicos para mejorar su postura.
Las startups tecnológicas están contratando diseñadores y programadores en todo el mundo.
La inteligencia artificial está transformando los procesos empresariales."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=170)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Qué plataformas se usan para reuniones virtuales?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    
    if st.button("¿Qué plataformas se usan para reuniones virtuales?", use_container_width=True):
        st.session_state.question = "¿Qué plataformas se usan para reuniones virtuales?"
        st.rerun()
    
    if st.button("¿Qué herramientas usan los desarrolladores para colaborar?", use_container_width=True):
        st.session_state.question = "¿Qué herramientas usan los desarrolladores para colaborar?"
        st.rerun()
        
    if st.button("¿Cómo mejora la postura un trabajador remoto?", use_container_width=True):
        st.session_state.question = "¿Cómo mejora la postura un trabajador remoto?"
        st.rerun()
        
    if st.button("¿Qué papel cumple la ciberseguridad?", use_container_width=True):
        st.session_state.question = "¿Qué papel cumple la ciberseguridad?"
        st.rerun()
        
    if st.button("¿Qué área está siendo transformada por la inteligencia artificial?", use_container_width=True):
        st.session_state.question = "¿Qué área está siendo transformada por la inteligencia artificial?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )
        
        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)
        
        # Mostrar matriz TF-IDF
        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Calcular similitud con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        # Encontrar mejor respuesta
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # Mostrar respuesta
        st.markdown("### 🎯 Respuesta")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")
