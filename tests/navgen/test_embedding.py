import shutil
import tempfile
import unittest
from pathlib import Path

from navgen_pipeline.embedding import LocalEmbedding, nearest_duplicate_pairs

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


@unittest.skipUnless(Image is not None, "Pillow is required")
class EmbeddingTests(unittest.TestCase):
    def test_duplicate_detector_requires_pixel_stability(self):
        with tempfile.TemporaryDirectory(prefix="navgen-embed-") as directory:
            root = Path(directory)
            first = root / "first.jpg"
            same = root / "same.jpg"
            changed = root / "changed.jpg"
            Image.new("RGB", (32, 18), (30, 40, 50)).save(first)
            Image.new("RGB", (32, 18), (30, 40, 50)).save(same)
            Image.new("RGB", (32, 18), (200, 40, 50)).save(changed)
            pairs = nearest_duplicate_pairs([first, same, changed], threshold=0.99)
            self.assertEqual(len(pairs), 1)
            self.assertEqual({Path(pairs[0][0]).name, Path(pairs[0][1]).name}, {"first.jpg", "same.jpg"})

    def test_embedding_dimension_is_stable(self):
        self.assertEqual(LocalEmbedding.dimension, 112)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
