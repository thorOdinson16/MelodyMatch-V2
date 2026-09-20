from pathlib import Path
import sys

import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Make src/melodymatch importable on Streamlit Cloud
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from melodymatch.data.labels import GENRE_TO_INDEX
from melodymatch.data.preprocessing import audio_to_mel
from melodymatch.models.cnn import CNNBaseline


# ============================================================
# Model configuration
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "cnn_decoupled.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INDEX_TO_GENRE = {
    index: genre
    for genre, index in GENRE_TO_INDEX.items()
}


# ============================================================
# Inference
# ============================================================

class MelodyMatchInference:

    def __init__(self):
        self.device = DEVICE

        self.model = CNNBaseline(
            num_classes=len(GENRE_TO_INDEX)
        ).to(self.device)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")
        self.val_macro_f1 = checkpoint.get("val_macro_f1")
        self.stage = checkpoint.get("stage")

    def _predict_window(self, audio_path: str):
        """
        Predict one 30-second audio window.
        """

        mel = audio_to_mel(audio_path)

        if not isinstance(mel, torch.Tensor):
            mel = torch.as_tensor(mel)

        # [1, 128, 1292]
        #       ↓
        # [1, 1, 128, 1292]
        mel = mel.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        with torch.no_grad():
            logits = self.model(mel)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

        return probabilities.cpu()

    def predict(
        self,
        audio_path: str,
        top_k: int = 3,
    ):
        """
        Predict genre for a single 30-second window.
        """

        probabilities = self._predict_window(audio_path)

        return self._format_predictions(
            probabilities,
            top_k,
        )

    def predict_windows(
        self,
        audio_paths: list[str],
        top_k: int = 3,
    ):
        """
        Predict several 30-second windows and
        average their probability distributions.
        """

        if not audio_paths:
            raise ValueError(
                "No audio windows were provided."
            )

        probabilities = torch.stack(
            [
                self._predict_window(path)
                for path in audio_paths
            ]
        ).mean(dim=0)

        return self._format_predictions(
            probabilities,
            top_k,
        )

    @staticmethod
    def _format_predictions(
        probabilities: torch.Tensor,
        top_k: int,
    ):
        top_k = min(
            top_k,
            len(INDEX_TO_GENRE),
        )

        values, indices = torch.topk(
            probabilities,
            k=top_k,
        )

        return [
            {
                "genre": INDEX_TO_GENRE[index.item()],
                "probability": probability.item(),
            }
            for probability, index in zip(
                values,
                indices,
            )
        ]