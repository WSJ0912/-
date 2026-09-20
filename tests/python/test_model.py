from __future__ import annotations

import unittest

import onnxruntime as ort
import torch
from cxr_research.export import export_onnx, verify_pytorch_onnx
from cxr_research.labels import CHEXPERT_LABELS
from cxr_research.mixstyle import MixStyle
from cxr_research.model import DenseNet121MultiLabel
from test_utils import case_directory


class ModelTests(unittest.TestCase):
    def test_mixstyle_shape_and_fixed_parameters(self) -> None:
        layer = MixStyle(p=0.5, alpha=0.1)
        self.assertEqual(layer.p, 0.5)
        self.assertEqual(layer.alpha, 0.1)
        layer.train()
        output = layer(torch.randn(2, 8, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 8, 8, 8))

    def test_densenet_has_fourteen_outputs_and_cam_features(self) -> None:
        model = DenseNet121MultiLabel(mixstyle=True, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits, features = model(torch.zeros(1, 1, 320, 320))
            cams = model.cam(features)
        self.assertEqual(tuple(logits.shape), (1, len(CHEXPERT_LABELS)))
        self.assertEqual(cams.shape[0], 1)
        self.assertEqual(cams.shape[1], len(CHEXPERT_LABELS))
        self.assertGreater(cams.shape[-1], 1)

    def test_real_densenet_onnx_matches_pytorch_probabilities(self) -> None:
        root = case_directory("onnx-export")
        torch.manual_seed(17)
        model = DenseNet121MultiLabel(mixstyle=True, pretrained=False)
        sample = torch.randn(1, 1, 320, 320)
        output_path = export_onnx(model, root / "model.onnx")

        verification = verify_pytorch_onnx(model, output_path, sample)
        self.assertTrue(verification["within_tolerance"], verification)
        self.assertLessEqual(verification["max_probability_error"], 1e-4)

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: sample.numpy()})
        self.assertEqual(tuple(outputs[0].shape), (1, len(CHEXPERT_LABELS)))
        self.assertEqual(outputs[1].shape[0], 1)
        self.assertEqual(outputs[1].shape[1], model.classifier.in_features)
        self.assertGreater(outputs[1].shape[-1], 1)
        self.assertEqual(outputs[2].shape[0], 1)
        self.assertEqual(outputs[2].shape[1], len(CHEXPERT_LABELS))
        self.assertEqual(outputs[2].shape[-2:], outputs[1].shape[-2:])
        self.assertTrue((outputs[2] >= 0).all())


if __name__ == "__main__":
    unittest.main()
