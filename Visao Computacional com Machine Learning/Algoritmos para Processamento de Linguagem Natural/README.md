# Algoritmos para Processamento de Linguagem Natural

> Módulo da formação **Machine Learning Specialist** focado em técnicas e algoritmos para Processamento de Linguagem Natural (PLN/NLP), abrangendo desde conceitos introdutórios até aplicações práticas com Deep Learning, síntese e reconhecimento de voz.

---

## 📚 Sumário

- [Aula 1 – Introdução ao Processamento de Linguagem Natural](#aula-1--introdução-ao-processamento-de-linguagem-natural)
- [Aula 2 – Deep Learning para Processamento de Linguagem Natural](#aula-2--deep-learning-para-processamento-de-linguagem-natural)
- [Aula 3 – Text to Speech com Processamento de Linguagem Natural](#aula-3--text-to-speech-com-processamento-de-linguagem-natural)
- [Aula 4 – Speech to Text com Processamento de Linguagem Natural](#aula-4--speech-to-text-com-processamento-de-linguagem-natural)

---

## Aula 1 – Introdução ao Processamento de Linguagem Natural

### O que é PLN?

**Processamento de Linguagem Natural** (PLN), do inglês *Natural Language Processing* (NLP), é uma subárea da Inteligência Artificial que se dedica à interação entre computadores e a linguagem humana. Seu objetivo é permitir que máquinas compreendam, interpretem, manipulem e gerem texto ou fala de forma que seja significativa e útil.

O PLN situa-se na interseção de três grandes campos:

- **Linguística** – estudo da estrutura e do significado da linguagem
- **Ciência da Computação** – algoritmos e estruturas de dados para processar informação
- **Inteligência Artificial / Machine Learning** – modelos que aprendem padrões a partir de dados

### Por que PLN é desafiador?

A linguagem humana é intrinsecamente ambígua e dependente de contexto. Alguns desafios clássicos:

| Desafio | Descrição | Exemplo |
|---|---|---|
| **Ambiguidade léxica** | Uma palavra com múltiplos significados | "banco" (assento / instituição financeira) |
| **Ambiguidade sintática** | Estrutura gramatical com múltiplas interpretações | "Vi o homem com o telescópio" |
| **Ironia e sarcasmo** | Sentido oposto ao literal | "Que dia maravilhoso" (dia ruim) |
| **Referência e correferência** | Identificar a quem/o quê um pronome se refere | "João disse que **ele** chegaria cedo" |
| **Linguagem informal** | Gírias, abreviações, erros ortográficos | "vc viu isso aí kkkk" |

### Pipeline clássico de PLN

Um projeto de PLN geralmente segue estas etapas:

```
Texto bruto
    │
    ▼
1. Pré-processamento
   ├── Tokenização
   ├── Remoção de stopwords
   ├── Normalização (lowercase, pontuação)
   └── Stemming / Lematização
    │
    ▼
2. Representação / Extração de Características
   ├── Bag of Words (BoW)
   ├── TF-IDF
   └── Word Embeddings (Word2Vec, GloVe, FastText)
    │
    ▼
3. Modelagem
   ├── Modelos estatísticos (Naive Bayes, SVM)
   └── Modelos de Deep Learning (RNN, LSTM, Transformers)
    │
    ▼
4. Avaliação e Interpretação
```

### Principais Tarefas de PLN

- **Análise de Sentimentos** – classificar o sentimento de um texto (positivo, negativo, neutro)
- **Classificação de Texto** – categorizar documentos em classes predefinidas
- **Reconhecimento de Entidades Nomeadas (NER)** – identificar nomes de pessoas, locais, organizações
- **Tradução Automática** – traduzir texto entre idiomas
- **Resumo Automático** – gerar resumos de textos longos
- **Resposta a Perguntas (QA)** – sistemas capazes de responder perguntas em linguagem natural
- **Modelagem de Tópicos** – descobrir temas latentes em um corpus
- **Síntese de Texto** – gerar texto novo e coerente
- **Síntese de Voz (TTS)** – converter texto em fala
- **Reconhecimento de Voz (STT/ASR)** – converter fala em texto

### Técnicas Fundamentais

#### Tokenização
Processo de dividir o texto em unidades menores (tokens), que podem ser palavras, subpalavras ou caracteres.

```
"O gato subiu no telhado" → ["O", "gato", "subiu", "no", "telhado"]
```

#### Stopwords
Palavras de alta frequência com pouco valor semântico que geralmente são removidas (artigos, preposições, conjunções). Ex.: "de", "a", "o", "que", "em".

#### Stemming vs. Lematização

| Técnica | Descrição | Exemplo |
|---|---|---|
| **Stemming** | Reduz a palavra ao seu radical, podendo gerar formas não existentes | "correndo" → "corr" |
| **Lematização** | Reduz a palavra à sua forma canônica (lema) do dicionário | "correndo" → "correr" |

#### TF-IDF (*Term Frequency – Inverse Document Frequency*)

Métrica que avalia a importância de uma palavra em um documento relativo a um corpus:

$$\text{TF-IDF}(t, d) = TF(t, d) \times \log\left(\frac{N}{df(t)}\right)$$

Onde:
- `TF(t, d)` = frequência do termo `t` no documento `d`
- `N` = número total de documentos
- `df(t)` = número de documentos que contêm o termo `t`

#### Word Embeddings
Representações vetoriais densas de palavras em um espaço de alta dimensão, onde palavras com significados semelhantes ficam próximas. Modelos populares: **Word2Vec**, **GloVe**, **FastText**.

---

## Aula 2 – Deep Learning para Processamento de Linguagem Natural

### Por que Deep Learning em PLN?

Os métodos tradicionais de PLN (SVM, Naive Bayes, regressão logística com TF-IDF) atingem bons resultados, mas dependem fortemente de engenharia manual de características (*feature engineering*). O **Deep Learning** permite que o modelo aprenda representações hierárquicas diretamente dos dados brutos, capturando padrões complexos e contextuais.

### Arquiteturas Fundamentais

#### 1. Redes Neurais Recorrentes (RNN)

As RNNs foram a primeira grande revolução do Deep Learning em PLN. Elas processam sequências token a token, mantendo um **estado oculto** que carrega informação dos passos anteriores.

```
Entrada:  x₁ → x₂ → x₃ → ... → xₙ
              ↓      ↓      ↓          ↓
Estado:   h₁ → h₂ → h₃ → ... → hₙ
```

**Limitação:** sofrem com o problema de *vanishing gradient*, dificultando o aprendizado de dependências de longo prazo.

#### 2. LSTM (*Long Short-Term Memory*)

Arquitetura que resolve o problema do *vanishing gradient* por meio de um mecanismo de **portas (gates)**:

- **Forget Gate** – decide o que descartar da memória de longo prazo
- **Input Gate** – decide o que adicionar à memória
- **Output Gate** – decide o que transmitir como saída

As LSTMs são capazes de capturar dependências em longas sequências, sendo amplamente usadas em tradução, geração de texto e análise de sentimento.

#### 3. GRU (*Gated Recurrent Unit*)

Variante mais simples da LSTM com apenas dois portões (*reset gate* e *update gate*), geralmente com desempenho comparável e menor custo computacional.

#### 4. Redes Convolucionais para Texto (TextCNN)

Aplicam filtros convolucionais sobre sequências de *embeddings* de palavras para capturar n-gramas locais. São muito eficazes em tarefas de classificação de texto pela sua velocidade e eficiência.

#### 5. Mecanismo de Atenção (*Attention Mechanism*)

Introduzido para superar a limitação do vetor de contexto fixo dos encoders seq2seq. O mecanismo de atenção permite que o decoder "olhe" para todos os estados do encoder e pese aqueles mais relevantes para cada posição de saída.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 6. Transformers

Arquitetura introduzida no artigo *"Attention is All You Need"* (Vaswani et al., 2017). Elimina recorrência e usa exclusivamente **atenção multi-cabeça** (*multi-head self-attention*), permitindo paralelização total durante o treinamento.

**Componentes principais:**

```
┌──────────────────────────────────┐
│         TRANSFORMER              │
│  ┌────────────┐  ┌────────────┐  │
│  │  ENCODER   │  │  DECODER   │  │
│  │            │  │            │  │
│  │ Self-Attn  │  │ Masked     │  │
│  │ Feed-Fwd   │  │ Self-Attn  │  │
│  │ Layer Norm │  │ Cross-Attn │  │
│  └────────────┘  │ Feed-Fwd   │  │
│                  └────────────┘  │
└──────────────────────────────────┘
```

### Modelos Pré-Treinados (Transfer Learning em PLN)

O conceito de **Transfer Learning** transformou o campo de PLN. Modelos treinados em grandes corpora podem ser *fine-tuned* para tarefas específicas com muito menos dados.

| Modelo | Ano | Arquitetura | Destaques |
|---|---|---|---|
| **ELMo** | 2018 | LSTM bidirecional | Embeddings contextuais |
| **GPT** | 2018 | Transformer Decoder | Geração de texto |
| **BERT** | 2018 | Transformer Encoder | Bidirecional, mascaramento |
| **GPT-2/3/4** | 2019-2023 | Transformer Decoder | Modelos generativos de grande escala |
| **RoBERTa** | 2019 | BERT otimizado | Treinamento mais robusto |
| **T5** | 2019 | Encoder-Decoder | Text-to-text unificado |
| **GPT-3.5 / ChatGPT** | 2022 | Transformer Decoder | RLHF, assistente conversacional |
| **LLaMA / Mistral** | 2023 | Transformer Decoder | Modelos open-source |

### BERT – *Bidirectional Encoder Representations from Transformers*

O BERT é pré-treinado com dois objetivos:

1. **Masked Language Model (MLM)** – 15% dos tokens são mascarados e o modelo deve prevê-los
2. **Next Sentence Prediction (NSP)** – o modelo aprende se uma sentença segue a outra

Após o pré-treinamento, pode ser *fine-tuned* para:
- Classificação de texto
- NER
- Resposta a perguntas
- Análise de sentimento

### Fine-Tuning vs. Feature Extraction

| Abordagem | Descrição | Quando usar |
|---|---|---|
| **Feature Extraction** | Congela os pesos do modelo pré-treinado e usa as representações como entrada para outro modelo | Poucos dados, recursos limitados |
| **Fine-Tuning** | Ajusta todos (ou parte dos) parâmetros do modelo pré-treinado na tarefa alvo | Dados suficientes, melhor desempenho |
| **Prompt Engineering** | Instrui o modelo via texto, sem alterar pesos | LLMs de grande porte |

### Métricas de Avaliação em PLN

| Tarefa | Métricas Comuns |
|---|---|
| Classificação | Accuracy, Precision, Recall, F1-Score |
| Tradução | BLEU Score |
| Resumo | ROUGE Score |
| Geração de texto | Perplexidade |
| NER | F1 por entidade |

---

## Aula 3 – Text to Speech com Processamento de Linguagem Natural

### O que é Text-to-Speech (TTS)?

**Text-to-Speech** (TTS), também chamado de **síntese de voz**, é a tecnologia que converte texto escrito em fala sintetizada. É uma das aplicações mais importantes de PLN, com uso em:

- Assistentes virtuais (Alexa, Siri, Google Assistant)
- Acessibilidade (leitores de tela para deficientes visuais)
- Audiobooks e narração automatizada
- Sistemas de atendimento por voz (URA)
- Aprendizado de idiomas
- GPS e navegação

### Histórico e Evolução do TTS

| Era | Abordagem | Característica |
|---|---|---|
| **1950–1980** | Síntese articulatória | Modelagem do trato vocal humano, qualidade limitada |
| **1980–2000** | Síntese por concatenação | Concatenação de segmentos de voz gravada, mais natural |
| **2000–2015** | Síntese paramétrica (HMM) | Modelos estatísticos, mais flexível porém robótico |
| **2016–hoje** | Deep Learning (WaveNet, Tacotron) | Qualidade próxima à voz humana |

### Arquiteturas Modernas de TTS

#### WaveNet (DeepMind, 2016)
Rede neural convolucional causal que gera amostras de áudio diretamente, amostra por amostra. Produz áudio de altíssima qualidade, porém com alto custo computacional.

#### Tacotron / Tacotron 2 (Google, 2017-2018)
Sistema seq2seq com atenção que converte texto diretamente em espectrogramas Mel, que são depois convertidos em áudio por um vocoder (ex.: WaveNet).

#### FastSpeech / FastSpeech 2
Modelo não-autorregressivo de geração de espectrograma, muito mais rápido que o Tacotron, mantendo qualidade comparável.

#### VITS (*Variational Inference with adversarial learning for end-to-end Text-to-Speech*)
Modelo end-to-end que combina VAE e GAN para gerar voz diretamente do texto com qualidade estado da arte.

### Pipeline de TTS

```
Texto de entrada
      │
      ▼
1. Pré-processamento de Texto
   ├── Normalização (números, abreviações, siglas)
   ├── Análise gramatical (G2P: grapheme-to-phoneme)
   └── Prosódia (ritmo, entonação, ênfase)
      │
      ▼
2. Modelo Acústico
   └── Gera espectrograma Mel a partir de fonemas
      │
      ▼
3. Vocoder
   └── Converte espectrograma em forma de onda de áudio
      │
      ▼
Áudio sintetizado (.wav / .mp3)
```

### gTTS – *Google Text-to-Speech*

A biblioteca **gTTS** é uma interface Python para a API de síntese de voz do Google. É simples de usar, suporta múltiplos idiomas e dialetos, e gera arquivos de áudio no formato MP3/WAV.

**Instalação:**
```bash
pip install gTTS
```

**Idiomas suportados (exemplos):**

| Código | Idioma |
|---|---|
| `en` | Inglês |
| `pt` | Português |
| `fr` | Francês |
| `es` | Espanhol |
| `de` | Alemão |
| `ja` | Japonês |
| `zh` | Chinês |

---

### 📄 Script: [Text-to-Speech-DIO.py](Text-to-Speech-DIO.py)

> Script disponibilizado pelo professor demonstrando o uso da biblioteca gTTS para síntese de voz em inglês e francês.

```python
# instalar a biblioteca gTTS !pip install gTTS

from IPython.display import Audio
from gtts import gTTS

# --- Exemplo 1: Síntese em Inglês ---
text_to_say = "How are you doing?."
language = "en"

gtts_object = gTTS(text=text_to_say,
                   lang=language,
                   slow=False)          # slow=False: velocidade normal de fala

gtts_object.save("/content/gtts.wav")
Audio("/content/gtts.wav")             # Reproduz o áudio no Jupyter/Colab


# --- Exemplo 2: Síntese em Francês (velocidade lenta) ---
french_text = "Je vais au supermarché"
french_language = "fr"

french_gtts_object = gTTS(text=french_text,
                           lang=french_language,
                           slow=True)   # slow=True: velocidade reduzida (útil para aprendizado)

french_gtts_object.save("/content/french.wav")
Audio("/content/french.wav")
```

**Conceitos demonstrados:**

| Parâmetro | Descrição |
|---|---|
| `text` | Texto a ser convertido em fala |
| `lang` | Código do idioma (ISO 639-1) |
| `slow` | `True` para fala mais devagar, `False` para velocidade normal |
| `.save()` | Salva o áudio em disco como arquivo WAV/MP3 |
| `Audio()` | Reproduz o áudio diretamente no Jupyter Notebook / Google Colab |

---

## Aula 4 – Speech to Text com Processamento de Linguagem Natural

### O que é Speech-to-Text (STT)?

**Speech-to-Text** (STT), também chamado de **Reconhecimento Automático de Fala** (ASR – *Automatic Speech Recognition*), é a tecnologia que converte sinais de fala (áudio) em texto escrito. É a operação inversa ao TTS.

**Aplicações:**

- Assistentes virtuais e chatbots por voz
- Transcrição automática de reuniões e podcasts
- Legendagem automática de vídeos
- Controle por voz de dispositivos e interfaces
- Sistemas de acessibilidade para deficientes auditivos
- Ditado por voz em editores de texto

### Pipeline de Reconhecimento de Fala

```
Áudio capturado pelo microfone
         │
         ▼
1. Pré-processamento de Áudio
   ├── Remoção de ruído (Noise Reduction)
   ├── Normalização de volume
   └── Segmentação (VAD – Voice Activity Detection)
         │
         ▼
2. Extração de Características Acústicas
   └── MFCC (Mel-Frequency Cepstral Coefficients)
         │
         ▼
3. Modelo Acústico
   └── Mapeia características de áudio para fonemas
         │
         ▼
4. Modelo de Linguagem
   └── Refina probabilidades com conhecimento linguístico
         │
         ▼
5. Decodificação
   └── Gera a sequência de texto mais provável
         │
         ▼
Texto transcrito
```

### Técnicas e Modelos de ASR

#### MFCC (*Mel-Frequency Cepstral Coefficients*)
Representação compacta do espectro de potência de um sinal de áudio que imita a percepção auditiva humana. É a principal feature usada como entrada para modelos acústicos.

#### Modelos Clássicos: HMM + GMM
Por décadas, o estado da arte em ASR combinava *Hidden Markov Models* (HMM) com *Gaussian Mixture Models* (GMM). Modelavam fonemas como estados ocultos numa sequência de Markov.

#### Modelos de Deep Learning para ASR

| Modelo | Organização | Destaques |
|---|---|---|
| **Deep Speech** | Baidu | RNN end-to-end com CTC Loss |
| **Listen, Attend and Spell (LAS)** | Google | Encoder-Decoder com atenção |
| **Conformer** | Google | Combina Transformer + CNN |
| **Whisper** | OpenAI | Transformer treinado em 680k horas de áudio multilíngue |
| **wav2vec 2.0** | Meta AI | Self-supervised, estado da arte em poucos dados |

#### CTC (*Connectionist Temporal Classification*)
Função de perda usada para treinar modelos end-to-end sem necessidade de alinhamento fonema-áudio. Permite que a rede aprenda o alinhamento diretamente dos dados.

### Bibliotecas Python para STT

| Biblioteca | Motor | Observação |
|---|---|---|
| `SpeechRecognition` | Google, Sphinx, Wit.ai, Azure... | Interface unificada para vários engines |
| `whisper` (OpenAI) | Modelo Whisper | Local, multilíngue, muito preciso |
| `vosk` | Modelos offline | Reconhecimento offline, leve |
| `deepspeech` | Mozilla | Open-source, baseado no Deep Speech |

### Assistente de Voz Inteligente

Um assistente de voz completo combina STT, PLN e TTS num único pipeline conversacional:

```
Microfone → [STT] → Texto → [NLU] → Intenção → [Lógica] → Resposta → [TTS] → Alto-falante
```

- **NLU** (*Natural Language Understanding*): compreende a intenção do usuário
- **Dialogue Management**: gerencia o contexto da conversa
- **NLG** (*Natural Language Generation*): gera a resposta em linguagem natural

---

### 📄 Script: [Speech-to-text.py](Speech-to-text.py)

> Script disponibilizado pelo professor implementando um **assistente de voz** completo que reconhece comandos por voz e responde com síntese de fala, integrando STT e TTS.

**Bibliotecas utilizadas:**

| Biblioteca | Função |
|---|---|
| `speech_recognition` | Captura e transcrição de áudio do microfone via Google Speech API |
| `gTTS` | Síntese de voz para as respostas do assistente |
| `playsound` | Reprodução de arquivos de áudio |
| `pyjokes` | Geração de piadas aleatórias |
| `wikipedia` | Consulta de informações na Wikipédia |
| `pyaudio` | Interface com o hardware de áudio |
| `webbrowser` | Abertura de URLs no navegador |
| `winshell` | Interação com o shell do Windows (ex.: Lixeira) |
| `pygame.mixer` | Reprodução de músicas |

**Instalação das dependências:**
```bash
pip install SpeechRecognition gTTS playsound pyjokes wikipedia pyaudio winshell pygame
```

**Comandos suportados pelo assistente:**

| Comando de voz | Ação executada |
|---|---|
| `"youtube"` | Abre o YouTube com uma busca por voz |
| `"search"` | Pesquisa na Wikipédia e lê o resultado |
| `"joke"` | Conta uma piada aleatória |
| `"empty recycle bin"` | Esvazia a lixeira do Windows |
| `"what time"` | Informa a hora atual |
| `"play music"` / `"play song"` | Reproduz música de um diretório local |
| `"stop music"` | Para a reprodução de música |
| `"exit"` | Encerra o assistente |

**Fluxo do assistente:**

```python
while True:
    print("I am listening...")   # Loop infinito de escuta
    text = get_audio()           # 1. Captura e transcreve o áudio (STT)
    respond(text)                # 2. Interpreta e responde ao comando
                                 #    (speak() usa gTTS para TTS)
```

**Funções principais:**

```python
# Captura áudio do microfone e converte para texto via Google Speech API
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)  # Redução de ruído ambiente
        audio = r.listen(source)
        said = r.recognize_google(audio)  # STT via Google
    return said.lower()

# Converte texto em fala e reproduz (TTS via gTTS)
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("voice.mp3")
    playsound.playsound("voice.mp3")

# Interpreta comandos e executa ações correspondentes
def respond(text):
    if 'youtube' in text: ...
    elif 'search' in text: ...
    elif 'joke' in text: ...
    # ...
```

---

## 🔧 Tecnologias e Ferramentas

| Tecnologia | Uso neste módulo |
|---|---|
| **Python 3.x** | Linguagem principal dos scripts |
| **gTTS** | Síntese de voz (Aulas 3 e 4) |
| **SpeechRecognition** | Reconhecimento de voz (Aula 4) |
| **Google Speech API** | Engine de STT (Aula 4) |
| **TensorFlow / PyTorch** | Frameworks para modelos de Deep Learning (Aula 2) |
| **Hugging Face Transformers** | Modelos pré-treinados (BERT, GPT, etc.) (Aula 2) |
| **NLTK / spaCy** | Ferramentas clássicas de PLN (Aula 1) |
| **Google Colab** | Ambiente de execução dos notebooks |

---

## 📖 Referências

- JURAFSKY, D.; MARTIN, J. H. **Speech and Language Processing**. 3rd ed. (draft). 2023. Disponível em: https://web.stanford.edu/~jurafsky/slp3/
- VASWANI, A. et al. **Attention Is All You Need**. NeurIPS, 2017.
- DEVLIN, J. et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL, 2019.
- OORD, A. et al. **WaveNet: A Generative Model for Raw Audio**. DeepMind, 2016.
- RADFORD, A. et al. **Robust Speech Recognition via Large-Scale Weak Supervision** (Whisper). OpenAI, 2022.
- Documentação gTTS: https://gtts.readthedocs.io/
- Documentação SpeechRecognition: https://pypi.org/project/SpeechRecognition/

---

*Formação Machine Learning Specialist – Visão Computacional com Machine Learning*
