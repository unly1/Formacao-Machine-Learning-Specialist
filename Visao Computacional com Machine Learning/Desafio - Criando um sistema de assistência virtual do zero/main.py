"""
Assistente Virtual - Interface Gráfica Principal
Desafio DIO: Criando um sistema de assistência virtual do zero

Módulos:
  - Text-to-Speech (gTTS + pygame)
  - Speech-to-Text (SpeechRecognition + Google API)
  - Comandos automatizados por voz (YouTube, Wikipedia, Farmácia, etc.)
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import sys
import os

# Garante que o diretório do projeto está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.tts import TextToSpeech
from modules.stt import SpeechToText
from modules import commands


# ═══════════════════════════════════════════════════════════════════
#  Paleta de cores e constantes de estilo
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark":       "#0d0d1a",
    "bg_medium":     "#151528",
    "bg_card":       "#1c1c35",
    "bg_input":      "#12122b",
    "accent_blue":   "#2563eb",
    "accent_red":    "#e94560",
    "accent_purple": "#7c3aed",
    "accent_green":  "#10b981",
    "accent_yellow": "#f59e0b",
    "text_primary":  "#f1f5f9",
    "text_secondary":"#94a3b8",
    "text_user":     "#93c5fd",
    "text_assist":   "#f0abfc",
    "text_system":   "#64748b",
    "text_error":    "#f87171",
    "border":        "#2d2d4e",
}

FONTS = {}  # inicializado em _setup_fonts()


# ═══════════════════════════════════════════════════════════════════
#  Aplicação Principal
# ═══════════════════════════════════════════════════════════════════

class AssistenteVirtualApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self._configure_window()
        self._setup_fonts()
        self._init_modules()
        self._init_state()
        self._build_ui()
        self._pulse_job = None  # animação do botão mic

        # Mensagem de boas-vindas após a janela aparecer
        self.root.after(400, self._welcome)

    # ─────────────────────────────────────────────────────────────
    #  Inicialização
    # ─────────────────────────────────────────────────────────────

    def _configure_window(self):
        self.root.title("🤖 Assistente Virtual · DIO")
        self.root.geometry("760x620")
        self.root.minsize(560, 460)
        self.root.configure(bg=COLORS["bg_dark"])
        # Tenta centralizar na tela
        self.root.update_idletasks()
        w, h = 760, 620
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _setup_fonts(self):
        FONTS["title"]   = tkfont.Font(family="Segoe UI", size=15, weight="bold")
        FONTS["label"]   = tkfont.Font(family="Segoe UI", size=9,  weight="bold")
        FONTS["body"]    = tkfont.Font(family="Segoe UI", size=10)
        FONTS["small"]   = tkfont.Font(family="Segoe UI", size=9)
        FONTS["mono"]    = tkfont.Font(family="Consolas", size=10)
        FONTS["mic_btn"] = tkfont.Font(family="Segoe UI Emoji", size=22)
        FONTS["icon"]    = tkfont.Font(size=16)

    def _init_modules(self):
        self.tts = TextToSpeech()
        self.stt = SpeechToText()

    def _init_state(self):
        self.waiting_for: str | None = None   # "youtube" | "wikipedia" | None
        self._mic_pulse_colors = [
            COLORS["accent_red"], "#c73050", "#a02040", "#c73050"
        ]
        self._mic_pulse_idx = 0

    # ─────────────────────────────────────────────────────────────
    #  Construção da UI
    # ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_chat_area()
        self._build_quick_buttons()
        self._build_input_bar()

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLORS["bg_card"], height=58)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Linha decorativa inferior
        tk.Frame(header, bg=COLORS["accent_blue"], height=2).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        tk.Label(
            header, text="🤖  Assistente Virtual",
            font=FONTS["title"],
            bg=COLORS["bg_card"], fg=COLORS["text_primary"],
        ).pack(side=tk.LEFT, padx=18, pady=12)

        # Status dot + texto
        self._status_var = tk.StringVar(value="● Aguardando")
        self._status_label = tk.Label(
            header,
            textvariable=self._status_var,
            font=FONTS["small"],
            bg=COLORS["bg_card"], fg=COLORS["accent_green"],
        )
        self._status_label.pack(side=tk.RIGHT, padx=18)

    def _build_chat_area(self):
        frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 4))

        # Scrollbar customizada
        scrollbar = tk.Scrollbar(frame, bg=COLORS["bg_card"],
                                  troughcolor=COLORS["bg_dark"],
                                  activebackground=COLORS["accent_blue"],
                                  width=8, relief=tk.FLAT)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat = tk.Text(
            frame,
            bg=COLORS["bg_medium"],
            fg=COLORS["text_primary"],
            font=FONTS["body"],
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=16, pady=10,
            cursor="arrow",
            selectbackground=COLORS["accent_blue"],
            yscrollcommand=scrollbar.set,
            spacing1=2, spacing3=2,
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.chat.yview)

        # Tags de formatação das mensagens
        self.chat.tag_config(
            "user_label",
            foreground=COLORS["text_user"],
            font=FONTS["label"],
        )
        self.chat.tag_config(
            "user_msg",
            foreground=COLORS["text_user"],
            font=FONTS["body"],
            lmargin1=16, lmargin2=16,
        )
        self.chat.tag_config(
            "assist_label",
            foreground=COLORS["text_assist"],
            font=FONTS["label"],
        )
        self.chat.tag_config(
            "assist_msg",
            foreground=COLORS["text_primary"],
            font=FONTS["body"],
            lmargin1=16, lmargin2=16,
        )
        self.chat.tag_config(
            "system_msg",
            foreground=COLORS["text_system"],
            font=FONTS["small"],
            lmargin1=8, lmargin2=8,
        )
        self.chat.tag_config(
            "error_msg",
            foreground=COLORS["text_error"],
            font=FONTS["small"],
            lmargin1=8, lmargin2=8,
        )
        self.chat.tag_config(
            "divider",
            foreground=COLORS["border"],
            font=FONTS["small"],
        )

    def _build_quick_buttons(self):
        """Linha de botões de ação rápida."""
        frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        frame.pack(fill=tk.X, padx=12, pady=(2, 4))

        quick = [
            ("🕐 Hora",      "que horas são"),
            ("📅 Data",      "que dia é hoje"),
            ("🤣 Piada",     "piada"),
            ("🎥 YouTube",   "youtube"),
            ("📚 Wikipedia", "pesquisar"),
            ("🗺️ Farmácia",  "farmácia"),
            ("❓ Ajuda",     "ajuda"),
        ]

        for label, cmd in quick:
            btn = tk.Button(
                frame,
                text=label,
                font=FONTS["small"],
                bg=COLORS["bg_card"],
                fg=COLORS["text_secondary"],
                activebackground=COLORS["accent_blue"],
                activeforeground=COLORS["text_primary"],
                relief=tk.FLAT,
                padx=8, pady=4,
                cursor="hand2",
                command=lambda c=cmd: self._dispatch_command(c),
                bd=0,
            )
            btn.pack(side=tk.LEFT, padx=2)
            # Efeito hover
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg=COLORS["text_primary"]))
            btn.bind("<Leave>", lambda e, b=btn: b.config(fg=COLORS["text_secondary"]))

    def _build_input_bar(self):
        """Barra inferior: botão mic + campo de texto + botão enviar."""
        bar = tk.Frame(self.root, bg=COLORS["bg_card"], height=68)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        # Linha decorativa superior
        tk.Frame(bar, bg=COLORS["border"], height=1).pack(fill=tk.X, side=tk.TOP)

        inner = tk.Frame(bar, bg=COLORS["bg_card"])
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # ── Botão Microfone ──────────────────────────
        self.mic_btn = tk.Button(
            inner,
            text="🎤",
            font=FONTS["mic_btn"],
            bg=COLORS["accent_red"],
            fg="white",
            activebackground="#c73050",
            activeforeground="white",
            relief=tk.FLAT,
            width=2,
            cursor="hand2",
            command=self._on_mic_click,
            bd=0,
        )
        self.mic_btn.pack(side=tk.LEFT, padx=(0, 10))

        # ── Campo de texto ───────────────────────────
        entry_frame = tk.Frame(inner, bg=COLORS["bg_input"],
                               highlightbackground=COLORS["border"],
                               highlightthickness=1)
        entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.text_entry = tk.Entry(
            entry_frame,
            font=FONTS["body"],
            bg=COLORS["bg_input"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            relief=tk.FLAT,
            bd=6,
        )
        self.text_entry.pack(fill=tk.X)
        self.text_entry.bind("<Return>", lambda _e: self._on_send())
        self.text_entry.insert(0, "Digite um comando ou use o microfone...")
        self.text_entry.config(fg=COLORS["text_system"])
        self.text_entry.bind("<FocusIn>",  self._on_entry_focus_in)
        self.text_entry.bind("<FocusOut>", self._on_entry_focus_out)

        # ── Botão Enviar ─────────────────────────────
        send_btn = tk.Button(
            inner,
            text="➤",
            font=FONTS["icon"],
            bg=COLORS["accent_blue"],
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief=tk.FLAT,
            width=2,
            cursor="hand2",
            command=self._on_send,
            bd=0,
        )
        send_btn.pack(side=tk.RIGHT, padx=(10, 0))

    # ─────────────────────────────────────────────────────────────
    #  Helpers de UI (thread-safe)
    # ─────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = None):
        color = color or COLORS["accent_green"]
        self._status_var.set(f"● {text}")
        self._status_label.config(fg=color)

    def _add_message(self, role: str, text: str, kind: str = "assist"):
        """Adiciona uma mensagem ao chat. Deve ser chamado na main thread."""
        self.chat.config(state=tk.NORMAL)

        if kind == "user":
            self.chat.insert(tk.END, "\n👤 Você\n", "user_label")
            self.chat.insert(tk.END, f"   {text}\n", "user_msg")
        elif kind == "assist":
            self.chat.insert(tk.END, "\n🤖 Assistente\n", "assist_label")
            self.chat.insert(tk.END, f"   {text}\n", "assist_msg")
        elif kind == "system":
            self.chat.insert(tk.END, f"ℹ  {text}\n", "system_msg")
        elif kind == "error":
            self.chat.insert(tk.END, f"✖  {text}\n", "error_msg")

        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    # thread-safe wrapper
    def _ui_add_message(self, role, text, kind="assist"):
        self.root.after(0, lambda: self._add_message(role, text, kind))

    def _ui_set_status(self, text, color=None):
        self.root.after(0, lambda: self._set_status(text, color))

    # ─────────────────────────────────────────────────────────────
    #  Placeholder do campo de texto
    # ─────────────────────────────────────────────────────────────

    _PLACEHOLDER = "Digite um comando ou use o microfone..."

    def _on_entry_focus_in(self, _e):
        if self.text_entry.get() == self._PLACEHOLDER:
            self.text_entry.delete(0, tk.END)
            self.text_entry.config(fg=COLORS["text_primary"])

    def _on_entry_focus_out(self, _e):
        if not self.text_entry.get().strip():
            self.text_entry.insert(0, self._PLACEHOLDER)
            self.text_entry.config(fg=COLORS["text_system"])

    # ─────────────────────────────────────────────────────────────
    #  Animação do botão microfone
    # ─────────────────────────────────────────────────────────────

    def _start_mic_pulse(self):
        self._mic_pulse_idx = 0
        self._pulse_step()

    def _pulse_step(self):
        if not self.stt.is_listening:
            self.mic_btn.config(bg=COLORS["accent_red"], text="🎤")
            return
        colors = self._mic_pulse_colors
        self.mic_btn.config(bg=colors[self._mic_pulse_idx % len(colors)])
        self._mic_pulse_idx += 1
        self._pulse_job = self.root.after(350, self._pulse_step)

    # ─────────────────────────────────────────────────────────────
    #  Ações de entrada
    # ─────────────────────────────────────────────────────────────

    def _on_mic_click(self):
        if self.stt.is_listening or self.tts.is_speaking:
            return

        self._set_status("Ouvindo…", COLORS["accent_yellow"])
        self.mic_btn.config(text="⏸")
        self._start_mic_pulse()

        self.stt.listen(
            on_result=self._on_speech_result,
            on_error=self._on_speech_error,
        )

    def _on_speech_result(self, text: str):
        self.root.after(0, lambda: self._handle_voice_input(text))

    def _on_speech_error(self, message: str):
        self.root.after(0, lambda: self._handle_voice_error(message))

    def _handle_voice_input(self, text: str):
        self.mic_btn.config(bg=COLORS["accent_red"], text="🎤")
        self._set_status("Processando…", COLORS["accent_blue"])
        self._add_message("Você", text, "user")
        self._dispatch_command(text)

    def _handle_voice_error(self, message: str):
        self.mic_btn.config(bg=COLORS["accent_red"], text="🎤")
        self._set_status("Aguardando", COLORS["accent_green"])
        self._add_message("", message, "error")

    def _on_send(self):
        text = self.text_entry.get().strip()
        if not text or text == self._PLACEHOLDER:
            return
        self.text_entry.delete(0, tk.END)
        self._add_message("Você", text, "user")
        self._dispatch_command(text)

    # ─────────────────────────────────────────────────────────────
    #  Dispatcher de comandos
    # ─────────────────────────────────────────────────────────────

    def _dispatch_command(self, text: str):
        """Roteia o texto para o handler correto (busca pendente ou novo cmd)."""
        text_lower = text.lower().strip()

        # ── Contexto pendente: aguardando termos de busca ────────
        if self.waiting_for == "youtube":
            self.waiting_for = None
            self._set_status("Abrindo YouTube…", COLORS["accent_blue"])
            result = commands.execute_youtube_search(text, self._speak)
            self._ui_add_message("Assistente", result, "assist")
            self._ui_set_status("Aguardando", COLORS["accent_green"])
            return

        if self.waiting_for == "wikipedia":
            self.waiting_for = None
            self._set_status("Buscando na Wikipedia…", COLORS["accent_blue"])

            def do_search():
                result = commands.execute_wikipedia_search(text, self._speak)
                self._ui_add_message("Assistente", result, "assist")
                self._ui_set_status("Aguardando", COLORS["accent_green"])

            threading.Thread(target=do_search, daemon=True).start()
            return

        # ── Novo comando ─────────────────────────────────────────
        def _set_waiting(ctx: str):
            self.waiting_for = ctx
            self._ui_set_status(f"Aguardando: {ctx}…", COLORS["accent_yellow"])

        result = commands.process_command(
            text_lower,
            speak_func=self._speak,
            on_waiting_input=_set_waiting,
        )

        if result == "EXIT":
            self._add_message("Assistente", "Até logo! 👋", "assist")
            self.root.after(2200, self.root.destroy)
        elif result:
            self._add_message("Assistente", result, "assist")

        if self.waiting_for is None:
            self._set_status("Aguardando", COLORS["accent_green"])

    # ─────────────────────────────────────────────────────────────
    #  TTS thread-safe
    # ─────────────────────────────────────────────────────────────

    def _speak(self, text: str, lang: str = "pt"):
        self._ui_set_status("Falando…", COLORS["accent_purple"])
        self.tts.speak(
            text, lang=lang,
            on_done=lambda: self._ui_set_status("Aguardando", COLORS["accent_green"]),
        )

    # ─────────────────────────────────────────────────────────────
    #  Boas-vindas
    # ─────────────────────────────────────────────────────────────

    def _welcome(self):
        self._add_message(
            "Sistema",
            "Pronto! Clique em 🎤 para falar ou use os botões de ação rápida.",
            "system",
        )
        self._add_message(
            "Sistema",
            "Dica: diga 'ajuda' para ver todos os comandos disponíveis.",
            "system",
        )
        self._speak(
            "Olá! Sou seu assistente virtual. Como posso ajudar você hoje?"
        )


# ═══════════════════════════════════════════════════════════════════
#  Entry-point
# ═══════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = AssistenteVirtualApp(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
