"""Review-source boundaries and raw prediction ordering, without model access."""
import copy
import unittest

from phone_sequences import collapse
from review_phoneme_consistency import selected_ids, triplet_comparisons


class ReviewTest(unittest.TestCase):
    def test_ctc_repeat_before_blank_removal_and_known_edit_orders(self):
        self.assertEqual(collapse([0, 3, 3, 0, 3, 4, 4, 0]), [3, 3, 4])
        row = {'anchor_id': 0, 'positive_id': 1, 'negative_id': 2, 'margin': -.1}
        values = {0: {'tokens': [4, 5, 6]}, 1: {'tokens': [4, 5, 6]}, 2: {'tokens': [4, 5]}}
        result = triplet_comparisons([row], values)[0]
        self.assertEqual(result['phone_order'], 'positive_closer')
        self.assertAlmostEqual(result['phone_margin'], 1/3)
        values[1], values[2] = values[2], values[1]
        self.assertEqual(triplet_comparisons([row], values)[0]['phone_order'], 'negative_closer')
        values[1] = values[2]
        self.assertEqual(triplet_comparisons([row], values)[0]['phone_order'], 'tie')
        values[2] = {'tokens': []}
        self.assertEqual(triplet_comparisons([row], values)[0]['phone_order'], 'empty_prediction')

    def test_selection_keeps_shared_sources_once_and_all_triplets(self):
        inventory = [{'id': i, 'class': int(i == 2), 'source_sha256': str(i), 'voice_group': ['same']} for i in range(3)]
        row = {'anchor_id': 0, 'positive_id': 1, 'negative_id': 2, 'class': 1}
        self.assertEqual(selected_ids(inventory, [row, row]), [0, 1, 2])
        self.assertEqual(inventory[0]['id'], 0)

    def test_ambiguous_ids_roles_duplicate_audio_or_voices_are_rejected(self):
        inventory = [{'id': i, 'class': int(i == 2), 'source_sha256': str(i), 'voice_group': ['same']} for i in range(3)]
        row = {'anchor_id': 0, 'positive_id': 1, 'negative_id': 2, 'class': 1}
        for key, value in [('anchor_id', -1), ('positive_id', 2), ('negative_id', True), ('class', 2)]:
            with self.assertRaises(ValueError):
                selected_ids(inventory, [{**row, key: value}])
        for key, value in [('id', 9), ('voice_group', ['other']), ('source_sha256', '0')]:
            changed = copy.deepcopy(inventory); changed[2][key] = value
            with self.assertRaises(ValueError):
                selected_ids(changed, [row])


if __name__ == '__main__':
    unittest.main()
