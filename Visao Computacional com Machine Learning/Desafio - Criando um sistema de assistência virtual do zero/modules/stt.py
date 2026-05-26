"""
Módulo Speech-to-Text (STT)
Captura áudio do microfone e converte em texto usando Google Speech Recognition.
"""

import threading
import speech_recognition as sr


class SpeechToText:
    """
    Módulo de Speech-to-Text usando SpeechRecognition + Google API.
    A escuta ocorre em thread separada para não bloquear a UI.
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.recognizer.energy_threshold = 300
        self._listening = False

    @property
    def is_listening(self) -> bool:
        return self._listening

    def listen(
        self,
        on_result,
        on_error=None,
        lang: str = "pt-BR",
        timeout: int = 10,
        phrase_limit: int = 12,
    ):
        """
        Ouve o microfone e retorna o texto reconhecido via callback.

        Args:
            on_result:    Callback(text: str) chamado com o texto reconhecido.
            on_error:     Callback(message: str) chamado em caso de erro.
            lang:         Idioma para reconhecimento ('pt-BR', 'en-US', etc.).
            timeout:      Segundos máximos aguardando o início da fala.
            phrase_limit: Segundos máximos de uma frase.
        """
        if self._listening:
            return

        def _run():
            self._listening = True
            try:
                with sr.Microphone() as source:
                    # Calibra para o ruído ambiente
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_limit,
                    )

                text = self.recognizer.recognize_google(audio, language=lang)
                on_result(text)

            except sr.WaitTimeoutError:
                if on_error:
                    on_error("Nenhuma fala detectada. Tente novamente.")
            except sr.UnknownValueError:
                if on_error:
                    on_error("Não consegui entender. Fale mais claramente.")
            except sr.RequestError as e:
                if on_error:
                    on_error(f"Serviço de reconhecimento indisponível: {e}")
            except OSError:
                if on_error:
                    on_error("Microfone não encontrado. Verifique sua entrada de áudio.")
            except Exception as e:
                if on_error:
                    on_error(f"Erro inesperado no STT: {e}")
            finally:
                self._listening = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
