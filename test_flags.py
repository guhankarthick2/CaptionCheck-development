import unittest
from decimal import Decimal

from flags import build_flags, extract_numbers, tokenize
from srt_io import Cue, parse_srt, write_srt


def word(start, end, text, probability=0.9):
    return {"start": start, "end": end, "word": text, "probability": probability}


class FlagTests(unittest.TestCase):
    def test_spoken_dose_and_digits_match(self):
        self.assertEqual(extract_numbers("Use 0.5 milligrams."), {Decimal("0.5")})
        self.assertEqual(extract_numbers("zero point zero five"), {Decimal("0.05")})
        self.assertEqual(extract_numbers("point oh five"), {Decimal("0.05")})
        self.assertEqual(extract_numbers("0 .05 milligrams"), {Decimal("0.05")})
        self.assertEqual(extract_numbers("Use 0.50 milligrams."), extract_numbers("zero point five"))
        self.assertEqual(extract_numbers("one hundred and one days"), {Decimal("101")})
        self.assertEqual(extract_numbers("101 days"), {Decimal("101")})
        self.assertEqual(extract_numbers("one-point-two"), {Decimal("1.2")})
        self.assertEqual(
            extract_numbers("after traveling for two years, one hundred and one days,"),
            extract_numbers("after traveling for two years, 101 days,"),
        )

    def test_number_mismatch(self):
        cues = [Cue(2, 4.0, 7.0, "Use 0.5 milligrams.")]
        words = [
            word(4.1, 4.4, "Use"),
            word(4.4, 4.8, "zero"),
            word(4.8, 5.1, "point"),
            word(5.1, 5.4, "zero"),
            word(5.4, 5.8, "five"),
            word(5.8, 6.4, "milligrams"),
        ]
        flags = build_flags(cues, words)
        self.assertEqual([flag["kind"] for flag in flags], ["number"])
        self.assertIn("0.5", flags[0]["detail"])
        self.assertIn("0.05", flags[0]["detail"])
        self.assertAlmostEqual(flags[0]["start"], 4.4)

    def test_same_number_is_not_flagged(self):
        cues = [Cue(1, 0, 2, "Use 0.05 milligrams.")]
        words = [word(0.1, 0.4, "zero"), word(0.4, 0.6, "point"), word(0.6, 0.8, "zero"), word(0.8, 1.1, "five")]
        self.assertEqual(build_flags(cues, words), [])

    def test_spelled_out_days_match_digits(self):
        cues = [Cue(1, 6.0, 10.5, "after traveling for two years,\none hundred and one days,")]
        words = [
            word(6.1, 6.4, "after"),
            word(6.4, 6.8, "traveling"),
            word(6.8, 7.0, "for"),
            word(7.0, 7.3, "two"),
            word(7.3, 7.7, "years"),
            word(7.7, 8.1, "101"),
            word(8.1, 8.5, "days"),
        ]
        self.assertEqual(build_flags(cues, words), [])

    def test_subscribe_outro_is_not_missing_caption(self):
        cues = [Cue(1, 0, 2, "Hello class."), Cue(2, 2, 5, "[ MUSIC ]")]
        words = [
            word(0.1, 0.4, "Hello"),
            word(0.4, 0.9, "class"),
            word(6.0, 6.4, "subscribe"),
            word(6.4, 6.7, "for"),
            word(6.7, 7.0, "more"),
            word(7.0, 7.5, "space"),
        ]
        self.assertEqual([flag["kind"] for flag in build_flags(cues, words)], [])

    def test_multiword_glossary_missing(self):
        cues = [Cue(1, 0, 3, "Things like surface pressure.")]
        words = [
            word(0.1, 0.4, "Things"),
            word(0.4, 0.7, "like"),
            word(0.7, 1.2, "surface"),
            word(1.2, 1.8, "tension"),
        ]
        flags = build_flags(cues, words, ["surface tension"])
        self.assertEqual(flags[0]["kind"], "glossary")
        self.assertEqual(flags[0]["possibleIssue"], "Glossary term missing from caption")

    def test_missing_negation(self):
        cues = [Cue(1, 0.2, 3.0, "Mix these chemicals.")]
        words = [
            word(0.3, 0.6, "Do"),
            word(0.6, 0.9, "not", 0.93),
            word(0.9, 1.2, "mix"),
            word(1.2, 1.6, "these"),
            word(1.6, 2.2, "chemicals"),
        ]
        flags = build_flags(cues, words)
        self.assertEqual(flags[0]["kind"], "negation")
        self.assertEqual(flags[0]["possibleIssue"], "Negation may be missing")
        self.assertAlmostEqual(flags[0]["score"], 0.93)

    def test_contraction_agreement_is_quiet(self):
        cues = [Cue(1, 0, 2, "Don't mix these chemicals.")]
        words = [word(0.1, 0.3, "Do"), word(0.3, 0.5, "not"), word(0.5, 0.8, "mix"), word(0.8, 1.2, "these")]
        self.assertEqual([flag["kind"] for flag in build_flags(cues, words)], [])

    def test_dropped_filler_is_not_a_row(self):
        cues = [Cue(1, 0, 2, "Mix chemicals now.")]
        words = [word(0.1, 0.3, "Mix"), word(0.3, 0.5, "the"), word(0.5, 1.0, "chemicals"), word(1.0, 1.4, "now")]
        self.assertEqual(build_flags(cues, words), [])

    def test_missing_caption_run(self):
        cues = [Cue(1, 0.0, 2.0, "Mix these chemicals.")]
        words = [
            word(0.2, 0.5, "Mix"),
            word(0.5, 0.8, "these"),
            word(0.8, 1.4, "chemicals"),
            word(8.0, 8.4, "Rinse"),
            word(8.4, 8.7, "with"),
            word(8.7, 9.1, "water"),
            word(9.1, 9.4, "right"),
            word(9.4, 9.9, "away"),
        ]
        flags = [flag for flag in build_flags(cues, words) if flag["kind"] == "missing"]
        self.assertEqual(len(flags), 1)
        self.assertIn("Rinse", flags[0]["audioText"])
        self.assertEqual(flags[0]["suggestion"], flags[0]["audioText"])

    def test_low_probability_hallucination_is_dropped(self):
        cues = [Cue(1, 0, 1.5, "Hello there friend.")]
        words = [
            word(5, 5.4, "thanks", 0.2),
            word(5.4, 5.8, "for", 0.2),
            word(5.8, 6.4, "watching", 0.21),
        ]
        self.assertEqual(build_flags(cues, words), [])

    def test_filler_run_is_dropped(self):
        cues = []
        words = [word(0, 0.4, "uh"), word(0.4, 0.8, "um"), word(0.8, 1.2, "the"), word(1.2, 1.6, "a")]
        self.assertEqual(build_flags(cues, words), [])

    def test_glossary_exact_and_near_miss(self):
        cues = [Cue(1, 0, 3, "This process is meiosis.")]
        words = [word(0.2, 0.5, "This"), word(0.5, 0.9, "process"), word(0.9, 1.1, "is"), word(1.1, 1.8, "mitosis")]
        flags = build_flags(cues, words, ["mitosis"])
        self.assertEqual(flags[0]["kind"], "glossary")
        self.assertEqual(flags[0]["possibleIssue"], "Glossary term missing from caption")

        swapped = build_flags(
            [Cue(1, 0, 3, "This process is mitosis.")],
            [word(0.2, 0.5, "This"), word(0.5, 0.9, "process"), word(0.9, 1.1, "is"), word(1.1, 1.8, "meiosis")],
            ["mitosis"],
        )
        self.assertEqual(swapped[0]["possibleIssue"], "Glossary term may be misspelled")

    def test_dismissed_flag_stays_hidden(self):
        cues = [Cue(1, 0.2, 3.0, "Mix these chemicals.")]
        words = [word(0.3, 0.6, "Do"), word(0.6, 0.9, "not"), word(0.9, 1.4, "mix"), word(1.4, 2.0, "chemicals")]
        flag_id = build_flags(cues, words)[0]["id"]
        self.assertEqual(build_flags(cues, words, dismissed={flag_id}), [])

    def test_srt_round_trip(self):
        original = "1\n00:00:01,000 --> 00:00:03,500\nMix these chemicals.\n\n2\n00:00:05,000 --> 00:00:08,000\nUse 0.5 milligrams.\n"
        cues = parse_srt(original)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[1].text, "Use 0.5 milligrams.")
        self.assertIn("00:00:05,000", write_srt(cues))
        self.assertIn("not", tokenize("Don't mix"))


if __name__ == "__main__":
    unittest.main()
