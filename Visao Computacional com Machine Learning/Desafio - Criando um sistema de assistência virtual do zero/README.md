# 🤖 Assistente Virtual - Desafio DIO

Sistema de assistência virtual construído com Python, combinando **PLN (Processamento de Linguagem Natural)**, reconhecimento de fala e síntese de voz em uma interface gráfica moderna.

---

## 📋 Requisitos do Desafio

| Requisito | Status |
|-----------|--------|
| Módulo Text-to-Speech (texto → áudio) | ✅ `modules/tts.py` |
| Módulo Speech-to-Text (fala → texto)  | ✅ `modules/stt.py` |
| Abrir pesquisa no YouTube por voz     | ✅ `modules/commands.py` |
| Busca na Wikipedia por voz            | ✅ `modules/commands.py` |
| Localização da farmácia mais próxima  | ✅ `modules/commands.py` |
| Interface gráfica                     | ✅ `main.py` (Tkinter) |

---

## 🗂️ Estrutura do Projeto

```
assistente-virtual/
├── main.py              ← Interface gráfica principal (Tkinter)
├── modules/
│   ├── __init__.py
│   ├── tts.py           ← Text-to-Speech  (gTTS + pygame)
│   ├── stt.py           ← Speech-to-Text  (SpeechRecognition)
│   └── commands.py      ← Processador de comandos de voz
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação

### 1. Pré-requisito: Python 3.9+

Verifique com:
```bash
python --version
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> **Nota:** `pyaudio` pode exigir instalação manual no Windows:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```
> Ou baixe o `.whl` em https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

---

## ▶️ Executando

```bash
python main.py
```

---

## 🎤 Comandos de Voz Disponíveis

| Fale... | Ação |
|---------|------|
| "Que horas são?" | Informa a hora atual |
| "Que dia é hoje?" | Informa a data atual |
| "Conta uma piada" | Conta uma piada aleatória |
| "YouTube" | Abre busca no YouTube (pede o termo) |
| "Pesquisar" / "Wikipedia" | Busca na Wikipedia (pede o termo) |
| "Farmácia" | Abre farmácias próximas no Google Maps |
| "Esvaziar lixeira" | Esvazia a Lixeira do Windows |
| "Ajuda" | Lista todos os comandos |
| "Sair" / "Tchau" | Encerra o assistente |

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Finalidade |
|------------|-----------|
| `tkinter` | Interface gráfica (built-in) |
| `SpeechRecognition` | Reconhecimento de fala (STT) |
| `gTTS` | Síntese de voz do Google (TTS) |
| `pygame` | Reprodução do áudio gerado |
| `wikipedia` | Busca de informações |
| `pyjokes` | Piadas aleatórias |
| `winshell` | Manipulação do Windows Shell |
| `webbrowser` | Abertura de URLs no navegador |

---

## 📚 Referências

- [Text to Speech - DIO](https://github.com/diegobrunoDIO/Text-to-Speech-DIO)
- [Speech to Text - DIO](https://github.com/diegobrunoDIO/Speech-to-text-ML-DIO)
- [SpeechRecognition docs](https://pypi.org/project/SpeechRecognition/)
- [gTTS docs](https://gtts.readthedocs.io/)
