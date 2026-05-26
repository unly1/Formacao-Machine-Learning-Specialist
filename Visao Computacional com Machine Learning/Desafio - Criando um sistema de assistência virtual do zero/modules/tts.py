"""
Módulo Text-to-Speech (TTS)
Converte texto em fala usando gTTS e reproduz com pygame.
"""

import os
import threading
from gtts import gTTS
from pygame import mixer


class TextToSpeech:
    """
    Módulo de Text-to-Speech usando gTTS + pygame.mixer.
    A reprodução ocorre em thread separada para não bloquear a UI.
    """

    TEMP_FILE = "temp_voice.mp3"

    def __init__(self):
        mixer.init()
        self._lock = threading.Lock()
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def speak(self, text: str, lang: str = "pt", on_start=None, on_done=None):
        """
        Converte 'text' em áudio e reproduz.

        Args:
            text:     Texto a ser falado.
            lang:     Código do idioma ('pt' para português, 'en' para inglês).
            on_start: Callback chamado quando a reprodução começa.
            on_done:  Callback chamado quando a reprodução termina.
        """
        def _run():
            with self._lock:
                self._speaking = True
                if on_start:
                    on_start()
                try:
                    tts = gTTS(text=text, lang=lang, slow=False)

                    # Remove arquivo anterior se existir
                    if os.path.exists(self.TEMP_FILE):
                        try:
                            mixer.music.stop()
                            mixer.music.unload()
                            os.remove(self.TEMP_FILE)
                        except OSError:
                            pass

                    tts.save(self.TEMP_FILE)
                    mixer.music.load(self.TEMP_FILE)
                    mixer.music.play()

                    # Aguarda o fim da reprodução
                    while mixer.music.get_busy():
                        pass

                except Exception as e:
                    print(f"[TTS] Erro: {e}")
                finally:
                    self._speaking = False
                    if on_done:
                        on_done()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def stop(self):
        """Para a reprodução em andamento."""
        try:
            mixer.music.stop()
        except Exception:
            pass
        self._speaking = False
