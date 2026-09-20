"""Generate a tiny synthetic corpus with controlled co-occurrence structure.

Design goals (so notebook results are verifiable, not vibes):
  * topic clusters -> nearest neighbours are checkable
  * gender pronouns + per-pair exclusive role words -> king - man + woman = queen
    works by construction (queen is the unique female sharing king's role words)
  * one paragraph (4 sentences, same topic) per line -> crop-pairs for
    contrastive training in notebook 02
Deterministic: seed fixed. ~6000 paragraphs, vocab ~130 words.
"""
import random

random.seed(0)

M_CTX = ["he", "his", "him"]
F_CTX = ["she", "her", "hers"]
# each (male, female) pair shares an exclusive pair of role words, so every
# royal/common word has ONE canonical opposite-gender counterpart
ROLE = {
    ("king", "queen"):      (["kingdom", "throne"], True),
    ("prince", "princess"): (["banquet", "court"], True),
    ("lord", "lady"):       (["castle", "tower"], True),
    ("man", "woman"):       (["market", "field"], False),
    ("boy", "girl"):        (["river", "meadow"], False),
    ("father", "mother"):   (["home", "hearth"], False),
    ("son", "daughter"):    (["cart", "barn"], False),
}
PAIRS = list(ROLE)
ROYAL_GENERIC = ["palace", "crown", "royal"]
COMMON_GENERIC = ["village", "house", "well"]
VERBS_S = ["rules", "guards", "loves", "visits", "watches"]
VERBS_P = ["rule", "guard", "love", "visit", "watch"]

def person_sentence(pair=None):
    pair = pair or random.choice(PAIRS)
    places, is_royal = ROLE[pair]
    generic = ROYAL_GENERIC if is_royal else COMMON_GENERIC
    place = random.choice(places + places + generic)   # role words weighted 2:1
    if random.random() < 0.25:
        m, f = pair
        return f"the {m} and the {f} {random.choice(VERBS_P)} the {place}"
    g = random.random() < 0.5
    who = pair[0] if g else pair[1]
    ctx = random.choice(M_CTX if g else F_CTX)
    verb = random.choice(VERBS_S)
    pat = random.choice([
        "{ctx} said the {who} {verb} the {place}",
        "the {who} {verb} the {place} and {ctx} smiles",
        "{ctx} saw the {who} near the {place}",
    ])
    return pat.format(who=who, ctx=ctx, verb=verb, place=place)

ANIMALS = ["dog", "cat", "horse", "cow", "sheep", "bird", "fish", "goat"]
ANIMAL_CTX = ["barks", "sleeps", "runs", "grazes", "farm", "pet", "fur", "tail", "feeds"]
FOODS = ["bread", "cheese", "soup", "rice", "tea", "coffee", "apple", "honey", "stew"]
FOOD_CTX = ["eats", "cooks", "kitchen", "warm", "cup", "bowl", "tasty", "bakes", "pot"]
TOOLS = ["hammer", "saw", "rope", "wheel", "plough", "anvil", "ladder", "chisel"]
TOOL_CTX = ["builds", "repairs", "workshop", "wood", "iron", "sharp", "heavy", "bench"]
WEATHER = ["rain", "snow", "wind", "storm", "sun", "cloud", "frost", "mist"]
WEATHER_CTX = ["falls", "blows", "cold", "bright", "sky", "morning", "winter", "summer"]

def topic_sentence(nouns, ctx, n=None):
    n = n or random.choice(nouns)
    c1, c2 = random.choice(ctx), random.choice(ctx)
    pat = random.choice([
        "the {n} {c1} near the {c2}",
        "a {n} and the {c1} by the {c2}",
        "every {n} {c1} when the {c2} comes",
    ])
    return pat.format(n=n, c1=c1, c2=c2)

# each paragraph is about ONE subject (a pair, or a noun) so paragraphs have
# identity — notebook 02 needs retrievable positives, not topic soup
TOPICS = [
    ("people", lambda: person_paragraph()),
    ("people", lambda: person_paragraph()),   # 2x: the analogy grid needs data
    ("animals", lambda: noun_paragraph(ANIMALS, ANIMAL_CTX)),
    ("food",    lambda: noun_paragraph(FOODS, FOOD_CTX)),
    ("tools",   lambda: noun_paragraph(TOOLS, TOOL_CTX)),
    ("weather", lambda: noun_paragraph(WEATHER, WEATHER_CTX)),
]

def person_paragraph(k=4):
    pair = random.choice(PAIRS)
    return " ".join(person_sentence(pair) for _ in range(k))

def noun_paragraph(nouns, ctx, k=4):
    n = random.choice(nouns)
    return " ".join(topic_sentence(nouns, ctx, n) for _ in range(k))

def main(n_paragraphs=6000, path="tiny_corpus.txt"):
    lines = []
    for _ in range(n_paragraphs):
        _, gen = random.choice(TOPICS)
        lines.append(gen())
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    vocab = {w for line in lines for w in line.split()}
    print(f"wrote {len(lines)} paragraphs, vocab={len(vocab)}")

if __name__ == "__main__":
    main()
