"""Graded skill suite for GPT-2 small: every item has one definite answer token.
Items are kept only if clean GPT-2 gets them right (top-1), and split into dev (for diagnosis) and test (for scoring)."""
import random, json, os
from common import *

R = random.Random(0)
NAMES = [" Mary", " John", " Sarah", " Tom", " Emma", " James", " Anna", " David", " Lisa", " Michael", " Kate", " Paul"]
CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Russia": "Moscow",
            "China": "Beijing", "Egypt": "Cairo", "England": "London", "Canada": "Ottawa", "Greece": "Athens",
            "Ireland": "Dublin", "Austria": "Vienna", "Norway": "Oslo", "Sweden": "Stockholm", "Poland": "Warsaw",
            "Portugal": "Lisbon", "Hungary": "Budapest", "Thailand": "Bangkok", "Kenya": "Nairobi", "Peru": "Lima",
            "Cuba": "Havana", "Iran": "Tehran", "Iraq": "Baghdad", "Syria": "Damascus", "Afghanistan": "Kabul",
            "Finland": "Helsinki", "Denmark": "Copenhagen", "Belgium": "Brussels", "Argentina": "Buenos",
            "Chile": "Santiago", "Turkey": "Ankara", "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Ukraine": "Kiev",
            "Mexico": "Mexico", "India": "New", "Australia": "Canberra", "Scotland": "Edinburgh", "Israel": "Jerusalem"}
FACTS = [("The Eiffel Tower is located in the city of", " Paris"), ("The Statue of Liberty is located in", " New"),
         ("Romeo and", " Juliet"), ("The Mona Lisa was painted by Leonardo da", " Vin"), ("Apple was founded by Steve", " Jobs"),
         ("Water is made of hydrogen and", " oxygen"), ("Microsoft was founded by Bill", " Gates"), ("Ladies and", " gentlemen"),
         ("The Great Wall of", " China"), ("Albert", " Einstein"), ("The theory of relativity was developed by Albert", " Einstein"),
         ("Neil Armstrong was the first man to walk on the", " moon"), ("The Taj Mahal is located in", " India")]
ANTONYMS = [("hot", "cold"), ("big", "small"), ("up", "down"), ("good", "bad"), ("fast", "slow"), ("old", "new"),
            ("happy", "sad"), ("light", "dark"), ("open", "closed"), ("rich", "poor"), ("hard", "soft"), ("high", "low"),
            ("early", "late"), ("left", "right"), ("strong", "weak"), ("true", "false"), ("full", "empty"), ("wet", "dry"),
            ("love", "hate"), ("win", "lose"), ("black", "white"), ("day", "night"), ("in", "out"), ("yes", "no")]
PLURALS = ["cat", "dog", "car", "book", "tree", "house", "bird", "girl", "boy", "table", "chair", "door", "window",
           "apple", "river", "song", "game", "friend", "student", "teacher", "flower", "computer", "phone", "city"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def build_items():
    it = {k: [] for k in ["facts", "arithmetic", "sequences", "copying", "names", "greater_than", "antonyms", "plurals"]}
    shot = "The capital of Peru is Lima. The capital of Kenya is Nairobi. "
    for c, a in CAPITALS.items():
        if c not in ("Peru", "Kenya"): it["facts"].append((f"{shot}The capital of {c} is", " " + a))
    it["facts"] += FACTS
    for a in range(10):
        for b in range(10):
            if a + b < 10: it["arithmetic"].append((f"1 + 2 = 3\n4 + 4 = 8\n2 + 5 = 7\n{a} + {b} =", f" {a + b}"))
    for n in range(1, 40): it["sequences"].append((f"{n}, {n + 1}, {n + 2}, {n + 3},", f" {n + 4}"))
    for i in range(7): it["sequences"].append((", ".join(DAYS[i:i + 3] if i + 3 <= 7 else DAYS[i:] + DAYS[:i + 3 - 7]) + ",", " " + DAYS[(i + 3) % 7]))
    for i in range(12): it["sequences"].append((", ".join(MONTHS[(i + k) % 12] for k in range(3)) + ",", " " + MONTHS[(i + 3) % 12]))
    for i in range(20): it["sequences"].append((" ".join(chr(65 + i + k) for k in range(4)), " " + chr(65 + i + 4)))
    pool = [w for w in (" " + x for x in "apple river stone cloud table window garden pencil mountain candle silver orange forest bottle engine planet rabbit castle doctor jacket ladder market needle ocean pepper rocket saddle tunnel violin wallet anchor basket cotton dragon falcon helmet island kettle lemon magnet napkin palace pirate puzzle rhythm sponge tomato velvet walnut".split()) if len(tok(w).input_ids) == 1]
    for _ in range(60):
        ws = R.sample(pool, 12); half = R.randint(3, 10)
        it["copying"].append(("".join(ws + ws[:half]), ws[half]))
    for _ in range(80):
        a, b = R.sample(NAMES, 2); place = R.choice(["store", "park", "office", "beach", "school"]); obj = R.choice(["a drink", "a book", "the keys", "a gift"])
        it["names"].append((f"When{a} and{b} went to the {place},{b} gave {obj} to", a))
    for yy in range(10, 90, 2):
        it["greater_than"].append((f"The war lasted from the year 17{yy} to the year 17", yy))
    for w, a in ANTONYMS:
        it["antonyms"].append((f"The opposite of hot is cold. The opposite of big is small. The opposite of {w} is", " " + a))
    for w in PLURALS:
        it["plurals"].append((f"One house, two houses. One book, two books. One {w}, two", " " + w + "s"))
    return it

def first_tok(s): return tok(s).input_ids[0]

def score(model, items, batch=64):
    """Per item: 1 if correct. greater_than: correct if top-1 is a 2-digit number > YY."""
    out = []
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        enc = tok([p for p, _ in chunk], return_tensors="pt", padding=True).to(DEV_)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)  # left padding: positions must start at the first real token
        top = model(**enc, position_ids=pos).logits[:, -1].argmax(-1).tolist()
        for (p, a), t in zip(chunk, top):
            if isinstance(a, int):
                s = tok.decode([t]).strip(); out.append(int(s.isdigit() and len(s) == 2 and int(s) > a))
            else:
                out.append(int(t == first_tok(a)))
    return out

def suite(model, items):
    return {k: sum(score(model, v)) / max(len(v), 1) for k, v in items.items()}

if __name__ == "__main__":
    m = load(); items = build_items(); kept = {}
    print(f"{'skill':14} {'built':>6} {'clean acc':>10} {'kept':>5}")
    for k, v in items.items():
        s = score(m, v); keep = [x for x, ok in zip(v, s) if ok]; R.shuffle(keep)
        kept[k] = {"dev": keep[: len(keep) // 2], "test": keep[len(keep) // 2:]}
        print(f"{k:14} {len(v):6d} {sum(s) / len(v):10.1%} {len(keep):5d}")
    json.dump(kept, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.json"), "w"))
