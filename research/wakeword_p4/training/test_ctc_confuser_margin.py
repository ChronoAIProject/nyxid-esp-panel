"""Small exhaustive checks of CTC hypothesis scores, without model weights."""
import itertools
import math
import unittest

import torch

from probe_ctc_confuser_margin import phone_scores


class CtcHypothesisTest(unittest.TestCase):
    def test_matches_all_paths_including_repeated_phones(self):
        vocab = {'a': 1, 'b': 2}
        templates = {'nyx': [('a', 'a'), ('a',)], 'mix': [('b',)],
                     'next': [('a', 'b')], 'nick': [('b', 'a')]}
        logits = torch.tensor([[.4, .1, -.2], [-.3, .2, .7], [.1, .9, .2]])
        probabilities = logits.softmax(-1).tolist()
        scored = phone_scores(logits, vocab, 0, templates)

        def collapse(path):
            return tuple(token for i, token in enumerate(path)
                         if token and (i == 0 or token != path[i-1]))

        for name, rows in templates.items():
            total = 0.
            for row in rows:
                target = tuple(vocab[token] for token in row)
                total += sum(math.prod(probabilities[i][token] for i, token in enumerate(path))
                             for path in itertools.product(range(3), repeat=3) if collapse(path) == target)
            expected = math.log(total/len(rows))
            self.assertAlmostEqual(expected, scored['log_probability'][name], places=6)

    def test_template_count_does_not_change_class_prior(self):
        logits = torch.tensor([[0., 2., -1.], [2., 0., -1.]])
        templates = {'nyx': [('a',)], 'mix': [('a',), ('a',)],
                     'next': [('b',)], 'nick': [('b',)]}
        scored = phone_scores(logits, {'a': 1, 'b': 2}, 0, templates)
        self.assertAlmostEqual(scored['log_probability']['nyx'], scored['log_probability']['mix'], places=6)
        self.assertAlmostEqual(scored['nyx_margin'], 0., places=6)


if __name__ == '__main__':
    unittest.main()
