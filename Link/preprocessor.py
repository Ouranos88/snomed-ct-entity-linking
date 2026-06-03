import re
import unicodedata
from typing import List, Tuple

Span = Tuple[int, int, str]  # (start, end, text) — end is exclusive

_STOPWORDS = frozenset(
    "a an the of in on at to for with by from and or but not is are was were be been "
    "have has had do does did will would could should may might shall can "
    "i he she it we they you this that these those per "
    "la le les de du des un une et ou sur par pour avec dans en au aux "
    "qui que quoi dont ou ce se sa son ses ma mon mes ta ton tes "
    "il elle ils elles je tu nous vous on lui y en "
    "aussi donc mais car ni or car apres avant après selon suite "
    "base bases sans plus moins tres tres bien mal ici la "
    "même meme ainsi autre autres tout tous toute toutes "
    "chez vers entre jusqu jusque deja déjà lors encore "
    "patient patients résultat resultat rapport note ".split()
)


def normalize_text(text: str) -> str:
    """NFC + collapse whitespace; preserves character offsets."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("\t", " ").replace("\f", " ")
    text = re.sub(r" {2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def _is_valid_span(text: str) -> bool:
    stripped = text.strip()
    if "\n" in stripped:
        return False
    if len(stripped) < 4:
        return False
    if len(stripped.strip("()[]{}.,;:!?\"'")) < 3:
        return False
    tokens = stripped.lower().split()
    if all(t.strip("()[].,;:") in _STOPWORDS for t in tokens):
        return False
    if tokens[0].strip("()[].,;:+-") in _STOPWORDS:
        return False
    if re.fullmatch(r"[\d\s.,/-]+", stripped):
        return False
    return True


def extract_candidate_spans(
    text: str,
    min_tokens: int = 1,
    max_tokens: int = 8,
) -> List[Span]:
    """Sliding-window span extraction; returns all (start, end, text) tuples."""
    tokens: List[Tuple[str, int, int]] = [
        (m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)
    ]
    spans: List[Span] = []
    n = len(tokens)
    for i in range(n):
        for j in range(i + min_tokens, min(i + max_tokens + 1, n + 1)):
            span_text = text[tokens[i][1]:tokens[j - 1][2]].strip("|:- ").strip()
            if _is_valid_span(span_text):
                spans.append((tokens[i][1], tokens[j - 1][2], span_text))
    return spans


_NEG_CUES = re.compile(
    r"\b(no|not|without|absence\s+of|negative\s+for|denies?|denying|never|"
    r"no\s+evidence\s+of|no\s+sign\s+of|ruled?\s+out|r/o|"
    r"pas\s+de|pas\s+d[''e]|aucun[e]?|sans|absence\s+de|n[ée]gatif|n[ée]gative|"
    r"aucune?|ni\s+[a-z]|exclut?|exclu[e]?|éliminé[e]?)\b",
    re.IGNORECASE,
)

_NEG_WINDOW = 60


def is_negated(text: str, span_start: int) -> bool:
    window = text[max(0, span_start - _NEG_WINDOW):span_start]
    return bool(_NEG_CUES.search(window))
