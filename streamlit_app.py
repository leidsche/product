import streamlit as st
import joblib

@st.cache_resource
def load():
    return joblib.load('tfidf.pkl'), joblib.load('model.pkl')
tfidf, model = load()

text = st.text_area("Вставьте текст:", height=200)
s = len(text.strip())
st.caption(f"Символов: {s} (допустимо: 300-5000)")
if st.button("Проверить текст", type="primary"):
    if not text.strip():
        st.warning("Текст не введён")
    elif s < 300:
        st.warning(f"Текст слишком короткий, минимум 300 символов")
    elif s > 5000:
        st.warning(f"Текст слишком длинный, максимум 5000 символов")
    else:
        textv = tfidf.transform([text])
        ai = model.predict_proba(textv)[0][1]
        st.subheader("Результат:")
        st.text(f"Вероятность написания человеком: {(1-ai)*100:.1f}%")
        st.text(f"Вероятность генерации ИИ: {ai*100:.1f}%")
