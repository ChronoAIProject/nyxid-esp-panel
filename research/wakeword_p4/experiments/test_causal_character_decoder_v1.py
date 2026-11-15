"""Protocol cases: CTC collapse, repeat separation, boundaries, continuous state."""
import unittest
from causal_character_decoder_v1 import ALPHABET, Decoder

class DecoderTest(unittest.TestCase):
    def emit(self, decoder, text):
        events = []
        for char in text:
            token = ALPHABET.index(char) + 1
            events += [decoder.step(token), decoder.step(token), decoder.step(0)]
        return sum(events)

    def test_continuous_keyword_commands_and_repeats(self):
        d = Decoder()
        self.assertEqual(sum(d.step(0) for _ in range(100)), 0)
        self.assertEqual(self.emit(d, 'HEY NICKS TURN ON THE LIGHT HEY NICKS'), 2)

    def test_nearwords_and_left_word_boundary(self):
        for text in ('HEY NICK', 'HEY NEXT', 'HEY MIX', 'HEY MUSE', 'THEY NICKS', 'AHEY NICKS'):
            self.assertEqual(self.emit(Decoder(), text), 0, text)

    def test_blank_separates_identical_ctc_characters(self):
        d = Decoder()
        token = ALPHABET.index('H') + 1
        for value in (token, token, 0, token):
            d.step(value)
        self.assertEqual(''.join(d.suffix), 'HH')

    def test_source_boundary_does_not_reset_partial_keyword(self):
        d = Decoder()
        self.assertEqual(self.emit(d, 'HEY '), 0)
        self.assertEqual(self.emit(d, 'NICKS'), 1)
        d.reset()
        self.assertEqual(self.emit(d, 'NICKS'), 0)

    def test_right_continuation_is_documented_prefix_behavior(self):
        self.assertEqual(self.emit(Decoder(), 'HEY NICKSON'), 1)

if __name__ == '__main__':
    unittest.main()
