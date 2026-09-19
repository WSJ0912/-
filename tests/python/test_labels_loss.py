from __future__ import annotations

import unittest

import torch
from cxr_research.labels import (
    CHEXPERT_LABELS,
    encode_row,
    parse_chexpert_value,
    validate_label_order,
)
from cxr_research.losses import masked_bce_with_logits


class LabelsAndLossTests(unittest.TestCase):
    def test_exact_fourteen_labels_and_mask_policy(self) -> None:
        self.assertEqual(len(CHEXPERT_LABELS), 14)
        self.assertEqual(parse_chexpert_value(-1), (0.0, 0.0))
        self.assertEqual(parse_chexpert_value(""), (0.0, 0.0))
        self.assertEqual(parse_chexpert_value(1), (1.0, 1.0))
        targets, mask = encode_row({"Atelectasis": -1, "Cardiomegaly": 1, "Edema": 0})
        self.assertEqual(targets[:3], [0.0, 1.0, 0.0])
        self.assertEqual(mask[:2], [0.0, 1.0])
        self.assertEqual(mask[3], 1.0)  # Edema=0 is observed; uncertainty is not negative.

    def test_reordered_dictionary_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_label_order(list(reversed(CHEXPERT_LABELS)))

    def test_masked_bce_ignores_uncertain_label(self) -> None:
        logits = torch.tensor([[0.0, 10.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0]])
        loss = masked_bce_with_logits(logits, targets, mask)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :1], targets[:, :1])
        self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=6)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[0, 1]), 0.0, places=7)

    def test_class_weight_is_capped(self) -> None:
        targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        mask = torch.ones_like(targets)
        from cxr_research.losses import capped_inverse_prevalence

        weights = capped_inverse_prevalence(targets, mask, max_weight=3.0)
        self.assertLessEqual(float(weights.max()), 3.0)


if __name__ == "__main__":
    unittest.main()
