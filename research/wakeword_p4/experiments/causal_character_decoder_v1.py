"""Fixed streaming greedy CTC keyword matcher, independent of audio boundaries."""
from collections import deque

ALPHABET = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
KEYWORD = 'HEY NICKS'

class Decoder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_token = 0
        self.suffix = deque(maxlen=len(KEYWORD) + 1)
        self.collapsed_characters = 0

    def step(self, token):
        """Consume one argmax token (blank=0); return a keyword event boolean."""
        if not isinstance(token, int) or not 0 <= token <= len(ALPHABET):
            raise ValueError('Expected integer token in [0, 27]')
        previous = self.previous_token
        self.previous_token = token
        if token == 0 or token == previous:
            return False
        self.suffix.append(ALPHABET[token - 1])
        self.collapsed_characters += 1
        text = ''.join(self.suffix)
        # Left boundary only. Fire as soon as final S is emitted: trailing
        # word continuation is deliberately not anticipated or endpoint-gated.
        return text == KEYWORD or text == ' ' + KEYWORD
