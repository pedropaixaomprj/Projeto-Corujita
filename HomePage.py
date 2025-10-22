import streamlit as st
# from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(
    page_title="Corujita - Início",
    page_icon="🦉",
    layout="centered"
)

# --- Add extra margin at the top of the page ---
st.markdown(
    """
    <div style="margin-top: 220px;">
    </div>
    """,
    unsafe_allow_html=True
)

# --- Logo and Introductory Text ---
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.image("imgs/corujita_branco.png", width=400)
with col_text:
    st.markdown(
        '''
        <div style="margin-top: 20px;">
            <p style="font-size: 1.5rem; margin: 0;">Olá, eu sou a Corujita, a assistente virtual do GATE!</p>
            <p style="font-size: 1.5rem; margin: 0;"><b>Em que posso ajudar?</b></p>
        </div>
        ''',
        unsafe_allow_html=True
    )

# --- Stylable container for the text input ---
with stylable_container(
    "input_container",
    css_styles="""
    /* The container around the Text Input */
    .stTextInput > div {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 30px;
        margin-bottom: 40px;  /* Added extra vertical space below the input */
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    /* Force the typed text color to black */
    .stTextInput input {
        color: #000000 !important;
        background-color: #FFFFFF;
    }
    /* Change the placeholder color */
    .stTextInput input::placeholder {
        color: #000000 !important;
        background-color: #FFFFFF;
    }
    """
):
    pergunta = st.text_input("pergunta_chat", placeholder="Escreva sua dúvida aqui:", label_visibility="collapsed")

# --- Stylable container for the "Iniciar Conversa" button ---
# Use columns to center the "Iniciar Conversa" button
col_left, col_center, col_right = st.columns([3, 2, 3])
with col_center:
    with stylable_container(
        "button_container",
        css_styles="""
        button {
            background-color: #ADB4C2 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 10px 20px !important;
            font-weight: bold !important;
            font-size: 1rem !important;
            cursor: pointer !important;
        }
        button:disabled {
            cursor: not-allowed !important;
            opacity: 0.6 !important;
        }
        """
    ):
        iniciar = st.button("Iniciar Conversa")

if iniciar:
    st.session_state["pending_user_message"] = pergunta
    st.switch_page("pages/Converse com a Corujita.py")
