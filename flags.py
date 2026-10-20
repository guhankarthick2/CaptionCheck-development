"""Find possible caption issues that can change what a viewer understands.

A row is a possible issue. The recognizer score is evidence about the second
transcript. It never creates a row by itself.
"""

from __future__ import annotations

import re
from decimal import Decimal

from srt_io import Cue


CONTRACTIONS = {
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "can't": "cannot",
    "cant": "cannot",
    "cannot": "can not",
    "won't": "will not",
    "wont": "will not",
    "isn't": "is not",
    "isnt": "is not",
    "wasn't": "was not",
    "wasnt": "was not",
    "aren't": "are not",
    "arent": "are not",
    "weren't": "were not",
    "werent": "were not",
    "haven't": "have not",
    "havent": "have not",
    "hasn't": "has not",
    "hasnt": "has not",
    "hadn't": "had not",
    "hadnt": "had not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "couldn't": "could not",
    "couldnt": "could not",
    "shouldn't": "should not",
    "shouldnt": "should not",
}

NEGATION = {"not", "no", "never", "without"}
ONES = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
DIGIT_WORDS = {key: str(value) for key, value in ONES.items() if value < 10}
DIGIT_WORDS["oh"] = "0"
NUMBER_WORDS = set(ONES) | set(TENS) | set(SCALES) | {"point", "and"}
FILLERS = {"uh", "um", "like", "the", "a"}
PROMO_RE = re.compile(
    r"\b(subscribe|like and share|thanks for watching|thank you for watching|"
    r"follow us|hit the bell|leave a comment|smash that like)\b",
    re.I,
)
NONSPEECH_RE = re.compile(
    r"^\[?\s*(music|silence|silence,? wind|wind|applause|laughter|inaudible|"
    r"background|sound|noise|beeping|explosion|rumbling)\s*\]?$",
    re.I,
)

_KIND_ORDER = {"number": 0, "negation": 1, "glossary": 2, "missing": 3}


def normalize_caption_text(text: str) -> str:
    cleaned = text.replace("\ufeff", "")
    cleaned = cleaned.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("�", "'")
    cleaned = re.sub(r"^>+", "", cleaned.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    lowered = normalize_caption_text(text).lower().replace("-", " ")
    for source, target in CONTRACTIONS.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered)
    lowered = re.sub(r"[^a-z0-9.'\s]", " ", lowered)
    tokens = []
    for raw in lowered.split():
        token = raw.strip("'")
        if re.fullmatch(r"\.?\d+(?:\.\d+)?\.?", token):
            token = token.rstrip(".")
        else:
            token = token.replace(".", "")
        if token:
            tokens.append(token)
    return tokens


def _parse_spoken_chunk(tokens: list[str], start: int) -> tuple[Decimal | None, int]:
    """Parse a spoken number starting at tokens[start]. Returns (value, next_index)."""
    index = start
    if index >= len(tokens):
        return None, start

    # Digits already present, optionally split as "0" ".05"
    if re.fullmatch(r"\d+", tokens[index]):
        if index + 1 < len(tokens) and re.fullmatch(r"\.\d+", tokens[index + 1]):
            return Decimal(tokens[index] + tokens[index + 1]), index + 2
        if index + 1 < len(tokens) and tokens[index + 1] == "point":
            frac = []
            cursor = index + 2
            while cursor < len(tokens) and tokens[cursor] in ONES and ONES[tokens[cursor]] < 10:
                frac.append(str(ONES[tokens[cursor]]))
                cursor += 1
            if frac:
                return Decimal(f"{tokens[index]}.{''.join(frac)}"), cursor
        return Decimal(tokens[index]), index + 1
    if re.fullmatch(r"\.\d+", tokens[index]):
        return Decimal("0" + tokens[index]), index + 1
    if re.fullmatch(r"\d+\.\d+", tokens[index]):
        return Decimal(tokens[index]), index + 1

    # Leading "point five" style decimals
    if tokens[index] == "point":
        frac = []
        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor] in ONES and ONES[tokens[cursor]] < 10:
            frac.append(str(ONES[tokens[cursor]]))
            cursor += 1
        if frac:
            return Decimal("0." + "".join(frac)), cursor
        return None, start

    total = 0
    current = 0
    consumed = False
    cursor = index
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == "and" and consumed:
            cursor += 1
            continue
        if token in ONES:
            current += ONES[token]
            consumed = True
            cursor += 1
            continue
        if token in TENS:
            current += TENS[token]
            consumed = True
            cursor += 1
            continue
        if token == "hundred" and consumed:
            current *= 100
            cursor += 1
            continue
        if token in SCALES and token != "hundred" and consumed:
            total += current * SCALES[token]
            current = 0
            cursor += 1
            continue
        if token == "point" and consumed:
            frac = []
            cursor += 1
            while cursor < len(tokens) and tokens[cursor] in ONES and ONES[tokens[cursor]] < 10:
                frac.append(str(ONES[tokens[cursor]]))
                cursor += 1
            value = Decimal(total + current)
            if frac:
                value = Decimal(f"{format(value, 'f')}.{''.join(frac)}")
            return value, cursor
        break
    if not consumed:
        return None, start
    return Decimal(total + current), cursor


def extract_numbers(text: str) -> set[Decimal]:
    tokens = tokenize(text)
    found: set[Decimal] = set()
    index = 0
    while index < len(tokens):
        value, next_index = _parse_spoken_chunk(tokens, index)
        if value is not None and next_index > index:
            found.add(value)
            index = next_index
            continue
        index += 1
    return found


def format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def audio_text(words: list[dict]) -> str:
    return " ".join(word["word"].strip() for word in words if word["word"].strip())


def overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def words_in_cue(words: list[dict], cue: Cue) -> list[dict]:
    return [word for word in words if overlap(word["start"], word["end"], cue.start, cue.end) > 0]


def is_nonspeech_cue(cue: Cue) -> bool:
    text = normalize_caption_text(cue.text)
    stripped = text.strip("[] ").lower()
    if NONSPEECH_RE.match(stripped) or NONSPEECH_RE.match(text):
        return True
    if text.startswith("[") and text.endswith("]") and not re.search(r"[a-zA-Z]{4,}", text.strip("[]")):
        return True
    return bool(re.fullmatch(r"\[.*\]", text) and not re.search(r"[A-Za-z]{3,}", text.strip("[]")))


def merge_cues(cues: list[Cue], max_gap: float = 0.35) -> list[Cue]:
    """Combine nearby short cues so fragmented captions compare as phrases."""
    if not cues:
        return []
    ordered = sorted(cues, key=lambda cue: (cue.start, cue.id))
    merged: list[Cue] = []
    current = Cue(ordered[0].id, ordered[0].start, ordered[0].end, ordered[0].text)
    for cue in ordered[1:]:
        gap = cue.start - current.end
        short = (current.end - current.start) < 2.5 or (cue.end - cue.start) < 2.5
        if gap <= max_gap and short and not is_nonspeech_cue(current) and not is_nonspeech_cue(cue):
            current = Cue(
                current.id,
                current.start,
                max(current.end, cue.end),
                normalize_caption_text(f"{current.text} {cue.text}"),
            )
        else:
            merged.append(current)
            current = Cue(cue.id, cue.start, cue.end, cue.text)
    merged.append(current)
    return merged


def speech_runs(words: list[dict], max_gap: float = 0.6) -> list[dict]:
    runs: list[dict] = []
    for word in words:
        if runs and word["start"] - runs[-1]["end"] < max_gap:
            runs[-1]["end"] = max(runs[-1]["end"], word["end"])
            runs[-1]["words"].append(word)
        else:
            runs.append({"start": word["start"], "end": word["end"], "words": [word]})
    return runs


def covered_duration(start: float, end: float, cues: list[Cue]) -> float:
    intervals = []
    for cue in cues:
        left = max(start, cue.start)
        right = min(end, cue.end)
        if right > left:
            intervals.append((left, right))
    intervals.sort()
    total = 0.0
    current_start = None
    current_end = None
    for left, right in intervals:
        if current_start is None:
            current_start, current_end = left, right
        elif left <= current_end:
            current_end = max(current_end, right)
        else:
            total += current_end - current_start
            current_start, current_end = left, right
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, char_left in enumerate(left, start=1):
        current = [index]
        for inner, char_right in enumerate(right, start=1):
            insert = current[inner - 1] + 1
            delete = previous[inner] + 1
            substitute = previous[inner - 1] + (char_left != char_right)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _score(words: list[dict]) -> float | None:
    if not words:
        return None
    return round(sum(float(word["probability"]) for word in words) / len(words), 4)


def _flag(
    kind: str,
    possible_issue: str,
    why: str,
    start: float,
    end: float,
    cue_id: int | None,
    caption_text: str,
    heard: str,
    score: float | None,
    suggestion: str | None,
    detail: str = "",
) -> dict:
    return {
        "id": f"{kind}-{start:.2f}",
        "kind": kind,
        "possibleIssue": possible_issue,
        "why": why,
        "detail": detail,
        "start": start,
        "end": end,
        "cueId": cue_id,
        "captionText": caption_text,
        "audioText": heard,
        "score": score,
        "suggestion": suggestion,
    }


def _number_flag(cue: Cue, overlapping: list[dict]) -> dict | None:
    heard = audio_text(overlapping)
    caption_values = extract_numbers(cue.text)
    audio_values = extract_numbers(heard)
    if caption_values == audio_values:
        return None
    if not caption_values and not audio_values:
        return None
    number_tokens = set(NUMBER_WORDS)
    relevant = []
    for word in overlapping:
        pieces = tokenize(word["word"])
        if any(re.search(r"\d", piece) or piece in number_tokens for piece in pieces):
            relevant.append(word)
    anchor = relevant or overlapping
    start = anchor[0]["start"] if anchor else cue.start
    end = anchor[-1]["end"] if anchor else cue.end
    caption_label = ", ".join(format_decimal(value) for value in sorted(caption_values)) or "none"
    audio_label = ", ".join(format_decimal(value) for value in sorted(audio_values)) or "none"
    return _flag(
        "number",
        "Different number",
        "Different measurement",
        start,
        end,
        cue.id,
        cue.text,
        heard,
        _score(relevant or overlapping),
        None,
        f"Caption has {caption_label}. Audio has {audio_label}.",
    )


def _negation_flag(cue: Cue, overlapping: list[dict]) -> dict | None:
    heard = audio_text(overlapping)
    caption_neg = set(tokenize(cue.text)) & NEGATION
    audio_tokens = tokenize(heard)
    audio_neg = set(audio_tokens) & NEGATION
    if caption_neg == audio_neg:
        return None
    only_audio = audio_neg - caption_neg
    only_caption = caption_neg - audio_neg
    relevant = []
    for word in overlapping:
        if set(tokenize(word["word"])) & only_audio:
            relevant.append(word)
    if only_audio and not only_caption:
        issue = "Negation may be missing"
    elif only_caption and not only_audio:
        issue = "Possible extra negation"
    else:
        issue = "Negation differs"
    start = relevant[0]["start"] if relevant else cue.start
    end = relevant[-1]["end"] if relevant else (overlapping[-1]["end"] if overlapping else cue.end)
    return _flag(
        "negation",
        issue,
        "Reverses an instruction",
        start,
        end,
        cue.id,
        cue.text,
        heard,
        _score(relevant or overlapping),
        None,
    )


def _contains_phrase(tokens: list[str], phrase: list[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    return any(tokens[index : index + width] == phrase for index in range(len(tokens) - width + 1))


def _phrase_near_miss(tokens: list[str], phrase: list[str]) -> bool:
    """True when a multi-word glossary phrase appears with a 1-2 letter edit on one word."""
    if len(phrase) < 2 or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    for index in range(len(tokens) - width + 1):
        window = tokens[index : index + width]
        distance = sum(levenshtein(left, right) for left, right in zip(window, phrase))
        if 1 <= distance <= 2 and all(len(word) >= 4 for word in phrase):
            return True
    return False


def _glossary_flags(cue: Cue, overlapping: list[dict], glossary: list[str]) -> list[dict]:
    heard = audio_text(overlapping)
    audio_tokens = tokenize(heard)
    caption_tokens = tokenize(cue.text)
    flags = []
    for term in glossary:
        term_tokens = tokenize(term)
        if not term_tokens:
            continue
        if len(term_tokens) == 1:
            term_token = term_tokens[0]
            in_audio = term_token in audio_tokens
            in_caption = term_token in caption_tokens
            if in_audio and not in_caption:
                relevant = [word for word in overlapping if term_token in tokenize(word["word"])]
                anchor = relevant[0] if relevant else None
                flags.append(
                    _flag(
                        "glossary",
                        "Glossary term missing from caption",
                        "Changes a required term",
                        anchor["start"] if anchor else cue.start,
                        anchor["end"] if anchor else cue.end,
                        cue.id,
                        cue.text,
                        heard,
                        _score(relevant),
                        None,
                    )
                )
            elif in_caption and not in_audio and len(term_token) >= 6 and audio_tokens:
                closest = min(audio_tokens, key=lambda token: levenshtein(term_token, token))
                distance = levenshtein(term_token, closest)
                if closest != term_token and distance in (1, 2):
                    relevant = [word for word in overlapping if closest in tokenize(word["word"])]
                    anchor = relevant[0] if relevant else None
                    flags.append(
                        _flag(
                            "glossary",
                            "Glossary term may be misspelled",
                            "Changes a required term",
                            anchor["start"] if anchor else cue.start,
                            anchor["end"] if anchor else cue.end,
                            cue.id,
                            cue.text,
                            heard,
                            _score(relevant),
                            None,
                        )
                    )
        else:
            in_audio = _contains_phrase(audio_tokens, term_tokens)
            in_caption = _contains_phrase(caption_tokens, term_tokens)
            if in_audio and not in_caption:
                flags.append(
                    _flag(
                        "glossary",
                        "Glossary term missing from caption",
                        "Changes a required term",
                        overlapping[0]["start"] if overlapping else cue.start,
                        overlapping[-1]["end"] if overlapping else cue.end,
                        cue.id,
                        cue.text,
                        heard,
                        _score(overlapping),
                        None,
                    )
                )
            elif in_caption and not in_audio and _phrase_near_miss(audio_tokens, term_tokens):
                flags.append(
                    _flag(
                        "glossary",
                        "Glossary term may be misspelled",
                        "Changes a required term",
                        overlapping[0]["start"] if overlapping else cue.start,
                        overlapping[-1]["end"] if overlapping else cue.end,
                        cue.id,
                        cue.text,
                        heard,
                        _score(overlapping),
                        None,
                    )
                )
    return flags


def _missing_flags(cues: list[Cue], words: list[dict]) -> list[dict]:
    flags = []
    for run in speech_runs(words):
        duration = run["end"] - run["start"]
        run_words = run["words"]
        if duration < 1.0 or len(run_words) < 3:
            continue
        if covered_duration(run["start"], run["end"], cues) >= 0.3 * duration:
            continue
        average = _score(run_words) or 0
        if average < 0.55:
            continue
        content = []
        for word in run_words:
            tokens = tokenize(word["word"])
            if tokens and any(token not in FILLERS for token in tokens):
                content.append(word)
        if len(content) < 2:
            continue
        heard = audio_text(run_words)
        if PROMO_RE.search(heard):
            continue
        flags.append(
            _flag(
                "missing",
                "Speech with no caption",
                "Viewer hears speech with no caption",
                run["start"],
                run["end"],
                None,
                "",
                heard,
                average,
                heard,
            )
        )
    return flags


def build_flags(
    cues: list[Cue],
    words: list[dict],
    glossary: list[str] | None = None,
    dismissed: set[str] | None = None,
) -> list[dict]:
    glossary = glossary or []
    dismissed = dismissed or set()
    # Preserve original cues for UI edits; analyze on merged nonspeech-aware groups.
    analysis_cues = [cue for cue in merge_cues(cues) if not is_nonspeech_cue(cue)]
    coverage_cues = list(cues)
    flags: list[dict] = []
    for cue in analysis_cues:
        overlapping = words_in_cue(words, cue)
        if not overlapping:
            continue
        number = _number_flag(cue, overlapping)
        if number:
            flags.append(number)
        negation = _negation_flag(cue, overlapping)
        if negation:
            flags.append(negation)
        flags.extend(_glossary_flags(cue, overlapping, glossary))
    flags.extend(_missing_flags(coverage_cues, words))
    seen: dict[str, int] = {}
    unique = []
    for flag in flags:
        base = flag["id"]
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            flag["id"] = f"{base}-{count}"
        if flag["id"] not in dismissed:
            unique.append(flag)
    unique.sort(key=lambda flag: (flag["start"], _KIND_ORDER.get(flag["kind"], 9), flag["id"]))
    return unique
