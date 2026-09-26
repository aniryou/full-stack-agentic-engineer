"""Load and chunk the NLTK Project Gutenberg corpus for the minifaiss demo.

Provides :func:`load_chunks`, which reads a handful of public-domain books,
splits each into fixed-size word windows ("passages"), and returns the passages
with metadata (which book, where). These passages are the "documents" the demo
turns into TF-IDF vectors and indexes with minifaiss.

Fetch the corpus once with ``python gutenberg_corpus.py`` (or call
:func:`ensure_corpus`). It uses ``nltk.download(..., quiet=True)``, which never
prompts. Avoid ``python -m nltk.downloader gutenberg``: when a download fails it
asks "Retry? [n/y/e]" on stdin, which hangs or crashes in a non-interactive shell.
"""

import numpy as np

# Nice display titles for the Gutenberg file ids we use.
_TITLES = {
    "austen-emma.txt": "Austen — Emma",
    "austen-persuasion.txt": "Austen — Persuasion",
    "austen-sense.txt": "Austen — Sense and Sensibility",
    "bible-kjv.txt": "The King James Bible",
    "blake-poems.txt": "Blake — Poems",
    "bryant-stories.txt": "Bryant — Stories",
    "burgess-busterbrown.txt": "Burgess — Buster Brown",
    "carroll-alice.txt": "Carroll — Alice in Wonderland",
    "chesterton-ball.txt": "Chesterton — The Ball and the Cross",
    "chesterton-brown.txt": "Chesterton — Father Brown",
    "chesterton-thursday.txt": "Chesterton — The Man Who Was Thursday",
    "edgeworth-parents.txt": "Edgeworth — The Parent's Assistant",
    "melville-moby_dick.txt": "Melville — Moby Dick",
    "milton-paradise.txt": "Milton — Paradise Lost",
    "shakespeare-caesar.txt": "Shakespeare — Julius Caesar",
    "shakespeare-hamlet.txt": "Shakespeare — Hamlet",
    "shakespeare-macbeth.txt": "Shakespeare — Macbeth",
    "whitman-leaves.txt": "Whitman — Leaves of Grass",
}

# A diverse default subset: novels, plays, an epic poem, and a children's book.
DEFAULT_BOOKS = [
    "melville-moby_dick.txt",
    "austen-emma.txt",
    "austen-sense.txt",
    "carroll-alice.txt",
    "shakespeare-hamlet.txt",
    "shakespeare-macbeth.txt",
    "chesterton-brown.txt",
    "milton-paradise.txt",
]


def ensure_corpus():
    """Make sure the NLTK Gutenberg corpus is present; download it if not.

    Non-interactive: ``nltk.download(quiet=True)`` returns False instead of
    prompting. Raises ``RuntimeError`` with the fix when the download fails.
    """
    import nltk

    try:
        nltk.data.find("corpora/gutenberg")
        return
    except LookupError:
        pass
    if nltk.download("gutenberg", quiet=True, raise_on_error=False):
        return
    raise RuntimeError(
        "Could not download the NLTK 'gutenberg' corpus. Check network access to "
        "raw.githubusercontent.com. Recent NLTK releases refuse to fetch through an HTTP(S) "
        "proxy; if yours is trusted, set NLTK_ALLOW_PROXIED_URLOPEN=1 and retry. Or unzip "
        "gutenberg.zip from the nltk_data repository into ~/nltk_data/corpora/."
    )


def title_for(fileid):
    """Human-readable title for a Gutenberg file id."""
    return _TITLES.get(fileid, fileid)


def load_chunks(books=None, words_per_chunk=90, max_chunks=1500, seed=1234):
    """Load books and split them into passages.

    Parameters
    ----------
    books : list of str or None
        Gutenberg file ids to load. Defaults to :data:`DEFAULT_BOOKS`.
    words_per_chunk : int
        Number of whitespace-separated words per passage.
    max_chunks : int
        Cap on the total number of passages (randomly subsampled, seeded, if the
        corpus produces more). Keeps the pure-Python demo fast.
    seed : int
        Seed for the subsampling RNG (reproducible).

    Returns
    -------
    texts : list of str
        The passage texts.
    books_of : list of str
        Parallel list: display title of the book each passage came from.
    """
    from nltk.corpus import gutenberg

    if books is None:
        books = DEFAULT_BOOKS

    texts = []
    books_of = []
    for fileid in books:
        words = gutenberg.raw(fileid).split()
        title = title_for(fileid)
        for start in range(0, len(words) - words_per_chunk + 1, words_per_chunk):
            chunk = " ".join(words[start:start + words_per_chunk])
            texts.append(chunk)
            books_of.append(title)

    # Subsample down to max_chunks (reproducibly) if we produced too many.
    if len(texts) > max_chunks:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(texts), size=max_chunks, replace=False))
        texts = [texts[i] for i in keep]
        books_of = [books_of[i] for i in keep]

    return texts, books_of


if __name__ == "__main__":
    ensure_corpus()
    print("NLTK 'gutenberg' corpus is ready.")
