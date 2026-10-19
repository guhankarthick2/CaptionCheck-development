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
DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
FILLERS = {"uh", "um", "like", "the", "a"}

_KIND_ORDER = {"number": 0, "negation": 1, "glossary": 2, "missing": 3}


def tokenize(text: str) -> list[str]:
    lowered = text.lower().replace("’", "'")
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


def extract_numbers(text: str) -> set[Decimal]:
    tokens = tokenize(text)
    found: set[Decimal] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            re.fullmatch(r"\d+", token)
            and index + 1 < len(tokens)
            and re.fullmatch(r"\.\d+", tokens[index + 1])
        ):
            found.add(Decimal(token + tokens[index + 1]))
            index += 2
            continue
        if re.fullmatch(r"\.\d+", token):
            found.add(Decimal("0" + token))
            index += 1
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            found.add(Decimal(token))
            index += 1
            continue
        if token not in DIGIT_WORDS and token != "point":
            index += 1
            continue
        parts: list[str] = []
        saw_point = False
        cursor = index
        while cursor < len(tokens) and (tokens[cursor] in DIGIT_WORDS or tokens[cursor] == "point"):
            if tokens[cursor] == "point":
                if saw_point:
                    break
                saw_point = True
                parts.append(".")
            else:
                parts.append(DIGIT_WORDS[tokens[cursor]])
            cursor += 1
        literal = "".join(parts)
        if literal.startswith("."):
            literal = "0" + literal
        if saw_point and re.fullmatch(r"\d+\.\d+", literal):
            found.add(Decimal(literal))
        index = max(cursor, index + 1)
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
    number_tokens = set(DIGIT_WORDS) | {"point"}
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
        elif _contains_phrase(audio_tokens, term_tokens) and not _contains_phrase(caption_tokens, term_tokens):
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
    flags: list[dict] = []
    for cue in cues:
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
    flags.extend(_missing_flags(cues, words))
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
