"""
Módulo de processamento de comandos de voz.
Interpreta o texto reconhecido e executa ações automatizadas.
"""

import webbrowser
import threading
from datetime import datetime

# Importações opcionais (com fallback gracioso)
try:
    import wikipedia
    wikipedia.set_lang("pt")
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    import pyjokes
    PYJOKES_AVAILABLE = True
except ImportError:
    PYJOKES_AVAILABLE = False

try:
    import winshell
    WINSHELL_AVAILABLE = True
except ImportError:
    WINSHELL_AVAILABLE = False


# ─────────────────────────────────────────────
#  Utilitários internos
# ─────────────────────────────────────────────

def _saudacao() -> str:
    hora = datetime.now().hour
    if 5 <= hora < 12:
        return "Bom dia"
    elif 12 <= hora < 18:
        return "Boa tarde"
    return "Boa noite"


# ─────────────────────────────────────────────
#  Processador principal de comandos
# ─────────────────────────────────────────────

def process_command(text: str, speak_func, on_waiting_input=None) -> str:
    """
    Analisa 'text' (em minúsculas) e executa a ação correspondente.

    Retorna:
        str  – mensagem para exibir no chat.
        "EXIT" – sinaliza que o assistente deve encerrar.
        "WAIT:<context>" – aguarda entrada do usuário para 'context'.
    """
    text = text.lower().strip()

    # ── YouTube ──────────────────────────────────────────────
    if any(k in text for k in ("youtube", "youtu")):
        speak_func("O que você quer pesquisar no YouTube?")
        if on_waiting_input:
            on_waiting_input("youtube")
        return "🎥 O que deseja buscar no YouTube?"

    # ── Wikipedia / Pesquisa ─────────────────────────────────
    if any(k in text for k in ("wikipedia", "pesquisar", "pesquisa", "buscar", "busca")):
        speak_func("O que você quer pesquisar?")
        if on_waiting_input:
            on_waiting_input("wikipedia")
        return "📚 O que deseja pesquisar na Wikipedia?"

    # ── Farmácia próxima ─────────────────────────────────────
    if any(k in text for k in ("farmácia", "farmacia", "remédio", "remedio", "medicamento")):
        url = "https://www.google.com/maps/search/farmácia+próxima/"
        webbrowser.open(url)
        msg = "Abrindo farmácias próximas no Google Maps."
        speak_func(msg)
        return f"🗺️ {msg}"

    # ── Hora ─────────────────────────────────────────────────
    if any(k in text for k in ("que horas", "horas são", "hora é", "que hora")):
        hora = datetime.now().strftime("%H:%M")
        msg = f"São {hora}."
        speak_func(msg)
        return f"🕐 {msg}"

    # ── Data ─────────────────────────────────────────────────
    if any(k in text for k in ("que dia", "data de hoje", "dia é hoje", "qual a data")):
        _DIAS = {
            "Monday": "segunda-feira", "Tuesday": "terça-feira",
            "Wednesday": "quarta-feira", "Thursday": "quinta-feira",
            "Friday": "sexta-feira", "Saturday": "sábado", "Sunday": "domingo",
        }
        now = datetime.now()
        dia_semana = _DIAS.get(now.strftime("%A"), now.strftime("%A"))
        data = now.strftime("%d/%m/%Y")
        msg = f"Hoje é {dia_semana}, {data}."
        speak_func(msg)
        return f"📅 {msg}"

    # ── Piada ─────────────────────────────────────────────────
    if any(k in text for k in ("piada", "joke", "me divirta", "engraçado")):
        if PYJOKES_AVAILABLE:
            joke = pyjokes.get_joke(language="en")
        else:
            joke = "Não consegui carregar as piadas. Instale a biblioteca pyjokes!"
        speak_func(joke)
        return f"🤣 {joke}"

    # ── Esvaziar Lixeira ─────────────────────────────────────
    if any(k in text for k in ("esvaziar lixeira", "limpar lixeira", "lixeira")):
        if WINSHELL_AVAILABLE:
            try:
                winshell.recycle_bin().empty(confirm=False, show_progress=False, sound=True)
                msg = "Lixeira esvaziada com sucesso!"
                speak_func(msg)
                return f"🗑️ {msg}"
            except Exception as e:
                return f"❌ Erro ao esvaziar lixeira: {e}"
        else:
            return "❌ winshell não disponível. Execute: pip install winshell pywin32"

    # ── Saudações ─────────────────────────────────────────────
    if any(k in text for k in ("olá", "oi", "hello", "bom dia", "boa tarde", "boa noite", "tudo bem")):
        msg = f"{_saudacao()}! Como posso ajudar você?"
        speak_func(msg)
        return f"👋 {msg}"

    # ── Sair ─────────────────────────────────────────────────
    if any(k in text for k in ("sair", "tchau", "fechar", "exit", "encerrar", "até logo")):
        speak_func("Até logo! Foi um prazer ajudar.")
        return "EXIT"

    # ── Ajuda ─────────────────────────────────────────────────
    if any(k in text for k in ("ajuda", "help", "o que você faz", "comandos")):
        msg = (
            "Posso: pesquisar no YouTube, buscar na Wikipedia, "
            "mostrar farmácias próximas, informar hora e data, "
            "contar piadas, esvaziar lixeira e muito mais!"
        )
        speak_func(msg)
        return f"ℹ️ {msg}"

    # ── Comando não reconhecido ───────────────────────────────
    msg = (
        "Não entendi. Tente: "
        "hora, data, piada, YouTube, pesquisar, farmácia, esvaziar lixeira ou ajuda."
    )
    speak_func(msg)
    return f"❓ {msg}"


# ─────────────────────────────────────────────
#  Executores de busca (chamados após input extra)
# ─────────────────────────────────────────────

def execute_youtube_search(keyword: str, speak_func) -> str:
    """Abre o YouTube com a pesquisa fornecida."""
    url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}"
    webbrowser.open(url)
    msg = f"Buscando '{keyword}' no YouTube."
    speak_func(msg)
    return f"🎥 {msg}"


def execute_wikipedia_search(query: str, speak_func) -> str:
    """Busca um resumo na Wikipedia e retorna o texto."""
    if not WIKIPEDIA_AVAILABLE:
        return "❌ Biblioteca wikipedia não instalada. Execute: pip install wikipedia"
    try:
        result = wikipedia.summary(query, sentences=3, auto_suggest=True)
        intro = f"Segundo a Wikipedia sobre '{query}': "
        speak_func(intro + result)
        return f"📚 {result}"
    except wikipedia.exceptions.DisambiguationError as e:
        sugestoes = ", ".join(e.options[:4])
        msg = f"Há múltiplos resultados. Seja mais específico. Sugestões: {sugestoes}."
        speak_func(msg)
        return f"📚 {msg}"
    except wikipedia.exceptions.PageError:
        msg = f"Não encontrei nada sobre '{query}' na Wikipedia."
        speak_func(msg)
        return f"📚 {msg}"
    except Exception as e:
        msg = f"Erro na busca: {e}"
        speak_func(msg)
        return f"📚 {msg}"
