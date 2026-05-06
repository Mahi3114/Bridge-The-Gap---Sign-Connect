import time
from grammar_corrector import polish_sentence

class GramifyEngine:
    def __init__(self, pause_threshold=1.5):
        self.buffer = []
        self.last_update_time = time.time()
        self.pause_threshold = pause_threshold

    # -----------------------------
    def add_word(self, word):
        word = word.strip().lower()
        self.buffer.append(word)
        self.last_update_time = time.time()

    # -----------------------------
    def is_pause(self):
        return (time.time() - self.last_update_time) > self.pause_threshold

    # -----------------------------
    def reset(self):
        self.buffer = []

    # -----------------------------
    def gramify(self):
        if not self.buffer:
            return ""

        words = self.buffer.copy()
        self.reset()

        sentence = self.apply_rules(words)

        # Final polish
        final = polish_sentence(sentence)
        return final

    # -----------------------------
    def apply_rules(self, words):
        w = words

        # ---------- DIRECT PHRASES ----------
        if "hello" in w:
            return "Hello"

        if "namaste" in w:
            return "Namaste"

        if "thank_you" in w:
            return "Thank you"

        if "sorry" in w:
            return "I am sorry"

        if "how_are_you" in w:
            return "How are you?"

        if "i_am_fine" in w:
            return "I am fine"

        # ---------- NAME ----------
        if "my" in w and "name_is" in w:
            return "My name is"

        if "name_is" in w:
            return "My name is"

        # ---------- EMOTIONS ----------
        if "happy" in w:
            return "I am happy"

        if "sad" in w:
            return "I am sad"

        if "sick" in w:
            return "I am sick"

        if "healthy" in w:
            return "I am healthy"

        # ---------- CONDITIONS ----------
        if "hot" in w:
            return "It is hot"

        if "cold" in w:
            return "It is cold"

        if "good" in w:
            return "It is good"

        if "bad" in w:
            return "It is bad"

        # ---------- HELP ----------
        if "help" in w:
            return "I need help"

        # ---------- TIME ----------
        if "today" in w:
            return "Today"

        if "tomorrow" in w:
            return "Tomorrow"

        if "yesterday" in w:
            return "Yesterday"

        if "morning" in w:
            return "In the morning"

        if "evening" in w:
            return "In the evening"

        if "night" in w:
            return "At night"

        # ---------- DEFAULT ----------
        return " ".join(w).capitalize()