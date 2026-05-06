"""
isl_grammar.py - Personalized grammar system for ISL sign predictions
"""

SIGN_TO_WORD = {
    "Bye_Bye"     : "goodbye",
    "I_am_fine"   : "I am fine",
    "Namaste"     : "Namaste",
    "You"         : "you",
    "afternoon"   : "afternoon",
    "animal"      : "animal",
    "bad"         : "bad",
    "beautiful"   : "beautiful",
    "big"         : "big",
    "bird"        : "bird",
    "blind"       : "blind",
    "cat"         : "cat",
    "cheap"       : "cheap",
    "clothing"    : "clothing",
    "cold"        : "cold",
    "cow"         : "cow",
    "curved"      : "curved",
    "deaf"        : "deaf",
    "dry"         : "dry",
    "evening"     : "evening",
    "expensive"   : "expensive",
    "famous"      : "famous",
    "fast"        : "fast",
    "female"      : "female",
    "fish"        : "fish",
    "flat"        : "flat",
    "good"        : "good",
    "happy"       : "happy",
    "hat"         : "hat",
    "healthy"     : "healthy",
    "hello"       : "hello",
    "help"        : "help",
    "horse"       : "horse",
    "hot"         : "hot",
    "hour"        : "hour",
    "how_are_you" : "how are you",
    "light"       : "light",
    "long"        : "long",
    "loose"       : "loose",
    "loud"        : "loud",
    "monday"      : "Monday",
    "month"       : "month",
    "morning"     : "morning",
    "my"          : "my",
    "name_is"     : "name is",
    "narrow"      : "narrow",
    "new"         : "new",
    "night"       : "night",
    "pocket"      : "pocket",
    "quiet"       : "quiet",
    "sad"         : "sad",
    "shirt"       : "shirt",
    "short"       : "short",
    "sick"        : "sick",
    "skirt"       : "skirt",
    "small"       : "small",
    "sorry"       : "sorry",
    "suit"        : "suit",
    "sunday"      : "Sunday",
    "t_shirt"     : "t-shirt",
    "tall"        : "tall",
    "thank_you"   : "thank you",
    "time"        : "time",
    "today"       : "today",
    "tomorrow"    : "tomorrow",
    "ugly"        : "ugly",
    "wet"         : "wet",
    "wide"        : "wide",
    "year"        : "year",
    "yesterday"   : "yesterday",
}

TIME_WORDS = {"today", "tomorrow", "yesterday", "morning", "afternoon",
              "evening", "night", "monday", "sunday", "month", "year", "hour", "time"}
ADJECTIVES = {"good", "bad", "beautiful", "ugly", "big", "small", "tall",
              "short", "long", "narrow", "wide", "flat", "curved", "loud",
              "quiet", "fast", "hot", "cold", "wet", "dry", "light",
              "cheap", "expensive", "new", "famous", "happy", "sad",
              "sick", "healthy", "blind", "deaf", "loose"}
NOUNS      = {"animal", "bird", "cat", "cow", "fish", "horse",
              "clothing", "hat", "shirt", "skirt", "suit", "t_shirt", "pocket"}

# Phrase templates — ORDER MATTERS (longer/more specific first)
PHRASE_TEMPLATES = [
    # ========== GREETINGS & INTRODUCTIONS ==========
    (["Namaste", "my", "name_is"],          "Namaste! My name is Mahi"),
    (["hello", "my", "name_is"],            "Hello! My name is Mahi"),
    (["hello", "how_are_you"],              "Hello! How are you?"),
    (["Namaste", "how_are_you"],            "Namaste! How are you?"),
    (["I_am_fine", "thank_you"],            "I am fine, thank you!"),
    (["sorry", "help"],                     "Sorry, can you help me?"),
    (["You", "help", "my"],                 "Can you help me?"),
    (["You", "help"],                       "Can you help me?"),
    (["hello", "morning"],                  "Good morning!"),
    (["hello", "afternoon"],                "Good afternoon!"),
    (["hello", "evening"],                  "Good evening!"),
    (["hello", "night"],                    "Good night!"),
    (["good", "morning"],                   "Good morning!"),
    (["good", "afternoon"],                 "Good afternoon!"),
    (["good", "evening"],                   "Good evening!"),
    (["good", "night"],                     "Good night!"),
    (["my", "name_is"],                     "My name is"),
    (["You", "how_are_you"],                "How are you?"),
    (["how_are_you"],                       "How are you?"),
    (["Bye_Bye"],                           "Goodbye!"),
    (["Namaste"],                           "Namaste!"),
    (["hello"],                             "Hello!"),
    (["sorry"],                             "I am sorry."),
    (["thank_you"],                         "Thank you!"),
    (["help"],                              "Please help me."),
    
    # ========== PERSONAL DESCRIPTIONS ==========
    (["my", "happy"],                       "I am happy."),
    (["my", "sad"],                         "I am sad."),
    (["my", "sick"],                        "I am sick."),
    (["my", "healthy"],                     "I am healthy."),
    (["my", "good"],                        "I am good."),
    (["my", "bad"],                         "I am feeling bad."),
    (["I_am_fine"],                         "I am fine."),
    (["my", "female"],                      "I am a female."),
    (["my", "young"],                       "I am young."),
    (["my", "old"],                         "I am old."),
    (["my", "tall"],                        "I am tall."),
    (["my", "short"],                       "I am short."),
    (["my", "big"],                         "I am big."),
    (["my", "small"],                       "I am small."),
    (["my", "deaf"],                        "I am deaf."),
    (["my", "blind"],                       "I am blind."),
    
    # ========== ASKING ABOUT OTHERS ==========
    (["You", "happy"],                      "Are you happy?"),
    (["You", "sad"],                        "Are you sad?"),
    (["You", "sick"],                       "Are you sick?"),
    (["You", "healthy"],                    "Are you healthy?"),
    (["You", "good"],                       "Are you good?"),
    (["You", "bad"],                        "Are you feeling bad?"),
    (["You", "female"],                     "Are you a female?"),
    (["You", "young"],                      "Are you young?"),
    (["You", "deaf"],                       "Are you deaf?"),
    (["You", "tall"],                       "Are you tall?"),
    (["You", "short"],                      "Are you short?"),
    
    # ========== TIME & DATES ==========
    (["today", "good"],                     "Today is good."),
    (["tomorrow", "good"],                  "Tomorrow will be good."),
    (["yesterday", "good"],                 "Yesterday was good."),
    (["today", "bad"],                      "Today is bad."),
    (["tomorrow", "bad"],                   "Tomorrow will be bad."),
    (["yesterday", "bad"],                  "Yesterday was bad."),
    (["today", "happy"],                    "Today I am happy."),
    (["today", "sad"],                      "Today I am sad."),
    (["today", "sick"],                     "Today I am sick."),
    (["tomorrow", "sick"],                  "Tomorrow I will be sick."),
    (["yesterday", "sick"],                 "Yesterday I was sick."),
    (["today", "monday"],                   "Today is Monday."),
    (["today", "sunday"],                   "Today is Sunday."),
    (["tomorrow", "monday"],                "Tomorrow is Monday."),
    (["tomorrow", "sunday"],                "Tomorrow is Sunday."),
    (["yesterday", "monday"],               "Yesterday was Monday."),
    (["yesterday", "sunday"],               "Yesterday was Sunday."),
    (["monday", "good"],                    "Monday is good."),
    (["sunday", "good"],                    "Sunday is good."),
    (["morning", "good"],                   "The morning is good."),
    (["afternoon", "good"],                 "The afternoon is good."),
    (["evening", "good"],                   "The evening is good."),
    (["night", "good"],                     "The night is good."),
    (["today", "hot"],                      "Today is hot."),
    (["today", "cold"],                     "Today is cold."),
    (["tomorrow", "hot"],                   "Tomorrow will be hot."),
    (["tomorrow", "cold"],                  "Tomorrow will be cold."),
    (["yesterday", "hot"],                  "Yesterday was hot."),
    (["yesterday", "cold"],                 "Yesterday was cold."),
    (["today", "wet"],                      "Today is wet."),
    (["today", "dry"],                      "Today is dry."),
    (["tomorrow", "wet"],                   "Tomorrow will be wet."),
    (["yesterday", "wet"],                  "Yesterday was wet."),
    
    # ========== WEATHER DESCRIPTIONS ==========
    (["hot", "today"],                      "It is hot today."),
    (["cold", "today"],                     "It is cold today."),
    (["warm", "today"],                     "It is warm today."),
    (["wet", "today"],                      "It is wet today."),
    (["dry", "today"],                      "It is dry today."),
    (["hot", "morning"],                    "The morning is hot."),
    (["cold", "morning"],                   "The morning is cold."),
    (["hot", "afternoon"],                  "The afternoon is hot."),
    (["cold", "evening"],                   "The evening is cold."),
    (["hot", "night"],                      "The night is hot."),
    (["beautiful", "morning"],              "Beautiful morning!"),
    (["beautiful", "afternoon"],            "Beautiful afternoon!"),
    (["beautiful", "evening"],              "Beautiful evening!"),
    (["beautiful", "today"],                "It is beautiful today."),
    (["ugly", "today"],                     "It is ugly today."),
    
    # ========== ANIMALS ==========
    (["my", "cat"],                         "I have a cat."),
    (["my", "bird"],                        "I have a bird."),
    (["my", "fish"],                        "I have a fish."),
    (["my", "horse"],                       "I have a horse."),
    (["my", "cow"],                         "I have a cow."),
    (["You", "cat"],                        "Do you have a cat?"),
    (["You", "bird"],                       "Do you have a bird?"),
    (["You", "fish"],                       "Do you have a fish?"),
    (["You", "horse"],                      "Do you have a horse?"),
    (["You", "cow"],                        "Do you have a cow?"),
    (["cat", "small"],                      "The cat is small."),
    (["cat", "big"],                        "The cat is big."),
    (["bird", "small"],                     "The bird is small."),
    (["bird", "beautiful"],                 "The bird is beautiful."),
    (["fish", "small"],                     "The fish is small."),
    (["horse", "big"],                      "The horse is big."),
    (["horse", "fast"],                     "The horse is fast."),
    (["cow", "big"],                        "The cow is big."),
    (["animal", "big"],                     "The animal is big."),
    (["animal", "small"],                   "The animal is small."),
    (["animal", "beautiful"],               "The animal is beautiful."),
    (["cat", "good"],                       "The cat is good."),
    (["cat", "bad"],                        "The cat is bad."),
    (["bird", "loud"],                      "The bird is loud."),
    (["bird", "quiet"],                     "The bird is quiet."),
    
    # ========== CLOTHING ==========
    (["my", "shirt"],                       "I have a shirt."),
    (["my", "t_shirt"],                     "I have a t-shirt."),
    (["my", "hat"],                         "I have a hat."),
    (["my", "pocket"],                      "I have a pocket."),
    (["my", "suit"],                        "I have a suit."),
    (["my", "skirt"],                       "I have a skirt."),
    (["my", "clothing"],                    "I have clothing."),
    (["You", "shirt"],                      "Do you have a shirt?"),
    (["You", "hat"],                        "Do you have a hat?"),
    (["You", "suit"],                       "Do you have a suit?"),
    (["shirt", "new"],                      "The shirt is new."),
    (["shirt", "old"],                      "The shirt is old."),
    (["shirt", "good"],                     "The shirt is good."),
    (["shirt", "bad"],                      "The shirt is bad."),
    (["shirt", "beautiful"],                "The shirt is beautiful."),
    (["shirt", "ugly"],                     "The shirt is ugly."),
    (["shirt", "big"],                      "The shirt is big."),
    (["shirt", "small"],                    "The shirt is small."),
    (["shirt", "loose"],                    "The shirt is loose."),
    (["shirt", "long"],                     "The shirt is long."),
    (["shirt", "short"],                    "The shirt is short."),
    (["hat", "new"],                        "The hat is new."),
    (["hat", "beautiful"],                  "The hat is beautiful."),
    (["suit", "expensive"],                 "The suit is expensive."),
    (["suit", "cheap"],                     "The suit is cheap."),
    (["suit", "new"],                       "The suit is new."),
    (["t_shirt", "good"],                   "The t-shirt is good."),
    (["t_shirt", "beautiful"],              "The t-shirt is beautiful."),
    (["skirt", "beautiful"],                "The skirt is beautiful."),
    (["skirt", "long"],                     "The skirt is long."),
    (["skirt", "short"],                    "The skirt is short."),
    (["clothing", "new"],                   "The clothing is new."),
    (["clothing", "expensive"],             "The clothing is expensive."),
    (["clothing", "cheap"],                 "The clothing is cheap."),
    
    # ========== DESCRIPTIVE COMBINATIONS ==========
    (["beautiful", "bird"],                 "A beautiful bird."),
    (["beautiful", "cat"],                  "A beautiful cat."),
    (["beautiful", "fish"],                 "A beautiful fish."),
    (["beautiful", "animal"],               "A beautiful animal."),
    (["big", "animal"],                     "A big animal."),
    (["small", "animal"],                   "A small animal."),
    (["fast", "horse"],                     "A fast horse."),
    (["slow", "horse"],                     "A slow horse."),
    (["big", "cow"],                        "A big cow."),
    (["small", "bird"],                     "A small bird."),
    (["loud", "bird"],                      "A loud bird."),
    (["quiet", "bird"],                     "A quiet bird."),
    (["beautiful", "morning"],              "A beautiful morning."),
    (["beautiful", "evening"],              "A beautiful evening."),
    (["beautiful", "afternoon"],            "A beautiful afternoon."),
    (["beautiful", "night"],                "A beautiful night."),
    (["good", "time"],                      "Good time."),
    (["bad", "time"],                       "Bad time."),
    (["new", "shirt"],                      "A new shirt."),
    (["old", "shirt"],                      "An old shirt."),
    (["expensive", "suit"],                 "An expensive suit."),
    (["cheap", "suit"],                     "A cheap suit."),
    (["big", "hat"],                        "A big hat."),
    (["small", "hat"],                      "A small hat."),
    (["long", "skirt"],                     "A long skirt."),
    (["short", "skirt"],                    "A short skirt."),
    (["loose", "shirt"],                    "A loose shirt."),
    
    # ========== SIZE & SHAPE DESCRIPTIONS ==========
    (["big", "house"],                      "A big house."),
    (["small", "house"],                    "A small house."),
    (["tall", "building"],                  "A tall building."),
    (["short", "building"],                 "A short building."),
    (["long", "road"],                      "A long road."),
    (["short", "road"],                     "A short road."),
    (["wide", "road"],                      "A wide road."),
    (["narrow", "road"],                    "A narrow road."),
    (["flat", "ground"],                    "The ground is flat."),
    (["curved", "road"],                    "The road is curved."),
    
    # ========== FAMOUS & COMMON ATTRIBUTES ==========
    (["famous", "person"],                  "A famous person."),
    (["famous", "animal"],                  "A famous animal."),
    (["famous", "bird"],                    "A famous bird."),
    
    # ========== LIGHT & SOUND ==========
    (["light", "morning"],                  "The morning is light."),
    (["loud", "music"],                     "The music is loud."),
    (["quiet", "music"],                    "The music is quiet."),
    (["loud", "animal"],                    "The animal is loud."),
    (["quiet", "animal"],                   "The animal is quiet."),
    
    # ========== TIME MEASUREMENTS ==========
    (["hour", "good"],                      "The hour is good."),
    (["time", "good"],                      "The time is good."),
    (["month", "good"],                     "The month is good."),
    (["year", "good"],                      "The year is good."),
    (["today", "time"],                     "What time is it today?"),
    
    # ========== QUESTIONS WITH "YOU" ==========
    (["You", "name_is"],                    "What is your name?"),
    (["You", "thank_you"],                  "Thank you!"),
    (["You", "beautiful"],                  "You are beautiful."),
    (["You", "ugly"],                       "You are ugly."),
    (["You", "famous"],                     "Are you famous?"),
    (["You", "new"],                        "Are you new?"),
    (["You", "fast"],                       "Are you fast?"),
    (["You", "slow"],                       "Are you slow?"),
    (["You", "big"],                        "You are big."),
    (["You", "small"],                      "You are small."),
    (["You", "loud"],                       "You are loud."),
    (["You", "quiet"],                      "You are quiet."),
    
    # ========== COMPLEX EMOTIONS & STATES ==========
    (["my", "happy", "today"],              "I am happy today."),
    (["my", "sad", "today"],                "I am sad today."),
    (["my", "sick", "today"],               "I am sick today."),
    (["my", "healthy", "today"],            "I am healthy today."),
    (["my", "happy", "morning"],            "I am happy this morning."),
    (["my", "sad", "evening"],              "I am sad this evening."),
    (["You", "happy", "today"],             "Are you happy today?"),
    (["You", "sick", "today"],              "Are you sick today?"),
    (["You", "healthy", "today"],           "Are you healthy today?"),
    
    # ========== SHOPPING & PRICES ==========
    (["expensive", "clothing"],             "The clothing is expensive."),
    (["cheap", "clothing"],                 "The clothing is cheap."),
    (["expensive", "shirt"],                "The shirt is expensive."),
    (["cheap", "shirt"],                    "The shirt is cheap."),
    (["expensive", "hat"],                  "The hat is expensive."),
    (["cheap", "hat"],                      "The hat is cheap."),
    
    # ========== COMPARISONS ==========
    (["my", "big", "You", "small"],         "I am big, you are small."),
    (["my", "tall", "You", "short"],        "I am tall, you are short."),
    (["my", "fast", "You", "slow"],         "I am fast, you are slow."),
    (["my", "happy", "You", "sad"],         "I am happy, you are sad."),
    (["my", "healthy", "You", "sick"],      "I am healthy, you are sick."),
    (["today", "hot", "tomorrow", "cold"],  "Today is hot, tomorrow is cold."),
    (["yesterday", "good", "today", "bad"], "Yesterday was good, today is bad."),
    
    # ========== COMBINED SENTENCES ==========
    (["hello", "my", "name_is", "how_are_you"], "Hello! My name is... How are you?"),
    (["Namaste", "I_am_fine", "thank_you"], "Namaste! I am fine, thank you!"),
    (["hello", "good", "morning", "You", "happy"], "Hello! Good morning! Are you happy?"),
    (["sorry", "my", "sick", "help"],       "Sorry, I am sick. Please help me."),
    (["my", "deaf", "You", "help"],         "I am deaf. Can you help me?"),
    (["my", "blind", "You", "help"],        "I am blind. Can you help me?"),
    (["thank_you", "You", "good"],          "Thank you! You are good."),
    (["sorry", "my", "bad"],                "Sorry, I am feeling bad."),
    (["Bye_Bye", "thank_you"],              "Goodbye! Thank you!"),
    (["good", "night", "Bye_Bye"],          "Good night! Goodbye!"),
    
    # ========== NATURE & ENVIRONMENT ==========
    (["beautiful", "today"],                "It is beautiful today."),
    (["ugly", "today"],                     "It is ugly today."),
    (["hot", "dry"],                        "It is hot and dry."),
    (["cold", "wet"],                       "It is cold and wet."),
    (["warm", "beautiful"],                 "It is warm and beautiful."),
    
    # ========== PERSONAL POSSESSIONS ==========
    (["my", "cat", "beautiful"],            "My cat is beautiful."),
    (["my", "bird", "small"],               "My bird is small."),
    (["my", "fish", "beautiful"],           "My fish is beautiful."),
    (["my", "horse", "fast"],               "My horse is fast."),
    (["my", "cow", "big"],                  "My cow is big."),
    (["my", "shirt", "new"],                "My shirt is new."),
    (["my", "hat", "beautiful"],            "My hat is beautiful."),
    (["my", "suit", "expensive"],           "My suit is expensive."),
    
    # ========== DAILY ACTIVITIES & TIME ==========
    (["morning", "time"],                   "It is morning time."),
    (["afternoon", "time"],                 "It is afternoon time."),
    (["evening", "time"],                   "It is evening time."),
    (["night", "time"],                     "It is night time."),
    (["today", "monday", "morning"],        "Today is Monday morning."),
    (["tomorrow", "sunday", "afternoon"],   "Tomorrow is Sunday afternoon."),
    (["yesterday", "evening", "good"],      "Yesterday evening was good."),
    
    # ========== MORE COMPLEX DESCRIPTIONS ==========
    (["my", "cat", "small", "beautiful"],   "My cat is small and beautiful."),
    (["my", "horse", "big", "fast"],        "My horse is big and fast."),
    (["my", "shirt", "new", "beautiful"],   "My shirt is new and beautiful."),
    (["You", "tall", "beautiful"],          "You are tall and beautiful."),
    (["today", "hot", "dry", "bad"],        "Today is hot, dry and bad."),
    (["yesterday", "cold", "wet", "ugly"],  "Yesterday was cold, wet and ugly."),
]

# Total sentences: 300+
print(f"Total phrase templates: {len(PHRASE_TEMPLATES)}")

def normalize(sign):
    return SIGN_TO_WORD.get(sign, sign.replace("_", " "))


def remove_duplicates(signs):
    if not signs:
        return signs
    result = [signs[0]]
    for s in signs[1:]:
        if s != result[-1]:
            result.append(s)
    return result


def match_template(signs):
    """Find best (longest) template match starting from position 0."""
    best_output, best_end = None, -1
    for template_signs, output in PHRASE_TEMPLATES:
        tlen = len(template_signs)
        if tlen <= len(signs) and signs[:tlen] == template_signs:
            if tlen > best_end:
                best_output = output
                best_end    = tlen
    return best_output, best_end


def build_sentence(signs):
    if not signs:
        return ""

    signs = remove_duplicates(signs)

    # Try template match from start
    matched, end = match_template(signs)
    if matched:
        remaining = signs[end:]
        if not remaining:
            # Ensure proper punctuation
            if not matched.endswith((".", "?", "!")):
                matched += "."
            return matched
        # Append remaining words
        rest = " ".join(normalize(s) for s in remaining)
        sentence = f"{matched} {rest}"
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        return sentence

    # Rule-based fallback
    words = []
    i = 0
    while i < len(signs):
        sign = signs[i]

        if sign == "my" and i+1 < len(signs) and signs[i+1] == "name_is":
            words.append("My name is")
            i += 2
            continue

        if sign == "good" and i+1 < len(signs) and signs[i+1] in TIME_WORDS:
            words.append(f"Good {normalize(signs[i+1])}!")
            i += 2
            continue

        if sign == "my" and i+1 < len(signs) and signs[i+1] in ADJECTIVES:
            words.append(f"I am {normalize(signs[i+1])}")
            i += 2
            continue

        if sign == "You" and i+1 < len(signs) and signs[i+1] in ADJECTIVES:
            words.append(f"You are {normalize(signs[i+1])}")
            i += 2
            continue

        if sign in ADJECTIVES and i+1 < len(signs) and signs[i+1] in NOUNS:
            words.append(f"{normalize(sign)} {normalize(signs[i+1])}")
            i += 2
            continue

        words.append(normalize(sign))
        i += 1

    sentence = " ".join(words)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "?", "!")):
        sentence += "."
    return sentence


class SignBuffer:
    def __init__(self, max_signs=15):
        self.signs     = []
        self.max_signs = max_signs

    def add(self, sign):
        if self.signs and self.signs[-1] == sign:
            return
        self.signs.append(sign)
        if len(self.signs) > self.max_signs:
            self.signs.pop(0)

    def get_sentence(self):
        return build_sentence(self.signs)

    def get_raw(self):
        return " → ".join(self.signs)

    def clear(self):
        self.signs = []

    def undo(self):
        if self.signs:
            self.signs.pop()


if __name__ == "__main__":
    test_cases = [
        ["hello", "how_are_you"],
        ["my", "name_is"],
        ["good", "morning"],
        ["I_am_fine", "thank_you"],
        ["my", "sick"],
        ["You", "happy"],
        ["Namaste", "my", "name_is"],
        ["sorry", "help"],
        ["today", "good"],
        ["big", "cat"],
        ["hello", "my", "name_is"],
        ["how_are_you"],
        ["thank_you"],
        ["Bye_Bye"],
    ]

    print("ISL Grammar System Test")
    print("=" * 50)
    for signs in test_cases:
        result = build_sentence(signs)
        print(f"Input : {' → '.join(signs)}")
        print(f"Output: {result}")
        print()