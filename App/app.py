import streamlit as st
import streamlit.components.v1 as components
import requests
from moduls.auto_scraper import AutoScraper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
from moduls.predictor_bias import BIAS
from moduls.predictor_fake import FAKE
from moduls.mistral_sm import EXPLAIN,SUMMARY
from moduls.figuras import FigBarras,FigTarta
import time
import html
import base64

st.set_page_config(page_title="NewsReaderAI", layout="wide")

logo_path = "./images/logo.png"
with open(logo_path, "rb") as f:
    img_bytes = f.read()
    encoded_logo = base64.b64encode(img_bytes).decode()

st.markdown(f"""
<div style='display: flex; align-items: center; width: 100%; padding: 45px 0 50px 0; border-bottom: 2px solid #3a3a3a;'>
    <div style='flex: 0 0 auto;'>
        <img src='data:image/png;base64,{encoded_logo}' style='height: 180px;' />
    </div>
    <div style='flex: 1; padding-left: 60px;'>
        <h1 style='font-size: 5.2rem; margin: 0; color: white; letter-spacing: 1px;'>NewsReader AI</h1>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)



with st.container():
    col1, col2 = st.columns([7, 1])
    with col1:
        url = st.text_input(
            label="Introduce una URL para analizar:",
            placeholder="Type the URL to analyze...",
            label_visibility="collapsed",
            key="url_input"
        )
    with col2:
        analizar = st.button("ANALYZE", use_container_width=True)

if analizar and url:

    can_embed = True
    try:
        response = requests.head(url, timeout=8, allow_redirects=True)
        xfo = response.headers.get('X-Frame-Options', '').lower()
        if 'deny' in xfo or 'sameorigin' in xfo:
            can_embed = False
    except Exception:
        can_embed = False


    col1, col2 = st.columns([1.5, 1])  

    with col1:
        st.markdown(
            """
            <h3 style="
                text-align: center;
                color: #ffffff;
                font-size: 1.8rem;
                margin-top: 0;
                margin-bottom: 1.2rem;
                font-weight: bold;
                letter-spacing: 0.5px;
            ">PAGE PREVIEW</h3>
            """,
            unsafe_allow_html=True
        )

        if can_embed:
            components.iframe(url, height=500, scrolling=True)
        else:
            st.warning("⚠️ The content preview is limited by the website's settings.")


    with col2:
        with st.spinner("Retrieving and summarizing the page content..."):
            try:
                texto_completo = AutoScraper(url)
                resumen = SUMMARY(texto_completo)
            except Exception as e:
                resumen = None
                

        if texto_completo and not resumen:
            st.warning("⚠️ The text was retrieved, but the summary could not be generated.")

        if resumen:
            st.markdown(
                """
                <h3 style="
                    text-align: center;
                    color: #ffffff;
                    font-size: 1.8rem;
                    margin-top: 0;
                    margin-bottom: 1.2rem;
                    font-weight: bold;
                    letter-spacing: 0.5px;
                ">SUMMARY</h3>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                    height: 460px;
                    overflow-y: auto;
                    padding: 1.5rem 2rem;
                    font-size: 15.5px;
                    line-height: 1.7;
                    background-color: #0f1015;  /* fondo igual al de la imagen */
                    border: 1px solid #2a2d3a;  /* azul oscuro-grisáceo para borde */
                    border-radius: 14px;
                    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.4);
                    color: #e0e0e0;
                    margin-top: 10px;
                    margin-bottom: 40px;
                ">
                    {html.escape(resumen).replace("\n", "<br>")}
                </div>
                """,
                unsafe_allow_html=True
            )

        

        else:
            st.warning("⚠️ Error retrieving the content — the source may require a subscription.")

    with st.spinner("Loading models..."):
        
        # MODELO BIAS

        dir1 = "./saved_models/Bias"
        model_bias = AutoModelForSequenceClassification.from_pretrained(dir1)
        tokenizer_bias = AutoTokenizer.from_pretrained(dir1)
        le = joblib.load(f"{dir1}/label_encoder.pkl")

        # MODELO FAKE

        dir2 = './saved_models/FakeNews'

        tokenizer_fake = AutoTokenizer.from_pretrained(dir2)
        model_fake = AutoModelForSequenceClassification.from_pretrained(
                    dir2,
                    num_labels=2,
                    use_safetensors=True
                )
    
    with st.spinner("Making predictions..."):

        # Predicciones bias

        start1 = time.time()
        try:
            results1 = BIAS(texto_completo, tokenizer_bias, model_bias, le)
        except Exception as e:
            results1 = None

        if results1:
            pred1 = results1['prediction']
            bias_fig1 = FigBarras(results1)
            bias_fig2 = FigTarta(results1)
        duration1 = time.time() - start1

        # Predicciones fake

        start2 = time.time()
        try:
            results2 = FAKE(texto_completo, tokenizer_fake, model_fake)
        except Exception as e:
            results2 = None

        if results2:
            pred2 = results2['prediction']
            fake_fig1 = FigBarras(results2)
            fake_fig2 = FigTarta(results2)

        duration2 = time.time() - start2

        # Interpretación con Mistral

        error_llm = False
        if results1 and results2:

            try:
                start3 = time.time()
                texto_llm = EXPLAIN(texto_completo, results1, results2)
                duration3 = time.time() - start3
            except:
                texto_llm = None
                error_llm = True
        
        else:
            texto_llm = None


    tab1, tab2, tab3 = st.tabs(["BiasDetector", "FakeNewsDetector", "LLM-Interpretation"])

    with tab1:

        if results1:
            st.subheader("Political Bias Predictor")

            st.markdown(
                f"""
                <div style="
                    background-color: #e0ecf1;
                    padding: 1.5rem;
                    border-radius: 15px;
                    border: 1px solid #a3b5c5;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                    margin-bottom: 25px;
                ">
                    <div style="font-size: 22px; font-weight: bold; color: #2b3e50;">
                        Prediction: <span style="color: #007acc;">{pred1}</span>
                    </div>
                    <div style="font-size: 16px; margin-top: 0.5rem; color: #4a4a4a;">
                        ⏱ Execution time: {duration1:.2f} seconds
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            chart11, chart12 = st.tabs(["Bar Chart", "Pie Chart"])

            with chart11:
                st.plotly_chart(bias_fig1, use_container_width=True)
            
            with chart12:
                st.plotly_chart(bias_fig2, use_container_width=True)


            with st.expander("View raw results (JSON)"):
                st.json(results1)
        
        else:
            st.warning("⚠️ Political bias prediction could not be performed. Please check the URL or the page content.")

    with tab2:

        if results2:
            st.subheader("Fake News Detector")

            st.markdown(
                f"""
                <div style="
                    background-color: #e0ecf1;
                    padding: 1.5rem;
                    border-radius: 15px;
                    border: 1px solid #a3b5c5;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                    margin-bottom: 25px;
                ">
                    <div style="font-size: 22px; font-weight: bold; color: #2b3e50;">
                        Prediction: <span style="color: #007acc;">{pred2}</span>
                    </div>
                    <div style="font-size: 16px; margin-top: 0.5rem; color: #4a4a4a;">
                        ⏱ Execution time: {duration2:.2f} seconds
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            chart21, chart22 = st.tabs(["Bar Chart", "Pie Chart"])

            with chart21:
                st.plotly_chart(fake_fig1, use_container_width=True)
            
            with chart22:
                st.plotly_chart(fake_fig2, use_container_width=True)


            with st.expander("View raw results (JSON)"):
                st.json(results2)

        else:
            st.warning("⚠️ Fake News detection could not be performed. Please check the URL or the page content.")


    def separar_parrafos_explicacion(texto: str) -> dict:
        partes = {'interpretation': '', 'justification': '', 'risk_analysis': ''}
        try:
            secciones = texto.split('**Interpretation paragraph:**')[1].split('**Justification paragraph:**')
            partes['interpretation'] = secciones[0].strip()
            resto = secciones[1].split('**Risk analysis paragraph:**')
            partes['justification'] = resto[0].strip()
            partes['risk_analysis'] = resto[1].strip()
        except (IndexError, AttributeError):
            partes = None

        return partes

    with tab3:

        if texto_llm:
            st.subheader("Interpretation")

            st.markdown(
                f"""
                <div style="
                    background-color: #dde9f2;
                    padding: 1rem 1.5rem;
                    border-radius: 15px;
                    border-left: 6px solid #3399cc;
                    margin-bottom: 25px;
                    font-size: 16px;
                    color: #2a3d4c;
                    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
                ">
                    ⏱ Execution time: <b>{duration3:.2f} seconds</b>
                </div>
                """,
                unsafe_allow_html=True
            )

            partes = separar_parrafos_explicacion(texto_llm)

            if partes is None:
                st.warning("⚠️ A problem occurred while formatting the text.")
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1f1f1f;
                        padding: 2rem;
                        border-radius: 18px;
                        border: 1px solid #3a3a3a;
                        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
                        font-size: 15.5px;
                        line-height: 1.7;
                        color: #e6e6e6;
                        margin: 30px 0;
                    ">
                        <h5 style="
                            margin-top: 0;
                            margin-bottom: 1rem;
                            color: #ffffff;
                            text-align: center;
                            font-size: 18px;
                            letter-spacing: 1px;
                        ">RAW INTERPRETATION</h5>
                            {html.escape(texto_llm).replace("\n", "<br>")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            else:
                for titulo, contenido in [
                    ("EXPLANATION", partes['interpretation']),
                    ("JUSTIFICATION", partes['justification']),
                    ("RISK WARNING", partes['risk_analysis'])
                ]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #1f1f1f;
                            padding: 2rem;
                            border-radius: 18px;
                            border: 1px solid #3a3a3a;
                            box-shadow: 0 6px 18px rgba(0,0,0,0.4);
                            font-size: 15.5px;
                            line-height: 1.7;
                            color: #e6e6e6;
                            margin: 30px 0;
                        ">
                            <h5 style="
                                margin-top: 0;
                                margin-bottom: 1rem;
                                color: #ffffff;
                                text-align: center;
                                font-size: 18px;
                                letter-spacing: 1px;
                            ">{titulo}</h5>
                            {html.escape(contenido).replace("\n", "<br>")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:
            if error_llm:
                st.error("⚠️ An error occurred connecting to the LLM model. Please try again later.")
            else:
                st.warning("⚠️ Analysis could not be generated using the LLM model.")








