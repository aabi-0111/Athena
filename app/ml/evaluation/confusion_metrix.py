"""
Athena v1.0
--------------------
Confusion Matrix

Responsibilities
----------------
1. Compute confusion matrix.
2. Optionally normalize the matrix.
3. Optionally save a visualization.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

__all__ = ["ConfusionMatrixGenerator", "ConfusionMatrixError"]

# Fixed label order for the binary fraud/legitimate matrix. Passed
# explicitly to sklearn's confusion_matrix() so the output is always
# 2x2 with a stable row/column order, regardless of which classes are
# actually present in a given y_true/y_pred slice. Without this, an
# evaluation batch containing only one class (very possible given how
# rare fraud is) silently collapses to a 1x1 matrix.
_BINARY_LABELS: Final[tuple[int, int]] = (0, 1)


class ConfusionMatrixError(ValueError):
    """Raised when the confusion matrix cannot be computed or saved."""


class ConfusionMatrixGenerator:
    """
    Generate confusion matrices for binary classification models.
    """

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and normalize prediction arrays.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.ndim != 1:
            raise ConfusionMatrixError(
                f"y_true must be 1-dimensional, got shape {y_true.shape}."
            )
        if y_pred.ndim != 1:
            raise ConfusionMatrixError(
                f"y_pred must be 1-dimensional, got shape {y_pred.shape}."
            )
        if y_true.size == 0:
            raise ConfusionMatrixError("y_true is empty.")
        if y_pred.size == 0:
            raise ConfusionMatrixError("y_pred is empty.")
        if y_true.size != y_pred.size:
            raise ConfusionMatrixError(
                f"y_true and y_pred must have the same length "
                f"(got {y_true.size} and {y_pred.size})."
            )
        return y_true, y_pred

    @classmethod
    def compute(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        normalize: str | None = None,
        labels: tuple[int, int] = _BINARY_LABELS,
    ) -> np.ndarray:
        """
        Compute the confusion matrix.

        Parameters
        ----------
        y_true : ndarray
            Ground-truth labels.
        y_pred : ndarray
            Predicted labels.
        normalize : {"true", "pred", "all"} | None
            Normalization strategy. Validated by sklearn itself.
        labels : tuple[int, int]
            The two class labels defining row/column order, fixed to
            (0, 1) by default. Passed explicitly to sklearn so the
            matrix is always 2x2 even if only one class is present in
            y_true/y_pred (e.g. a small or time-windowed evaluation
            slice with zero fraud cases) — without this, sklearn infers
            the label set from the data and silently returns a smaller
            matrix.

        Returns
        -------
        ndarray
            2x2 confusion matrix (unless `labels` is overridden).

        Raises
        ------
        ConfusionMatrixError
            If inputs are empty, mismatched in length, or not 1-D.
        """
        y_true, y_pred = cls._validate_inputs(y_true, y_pred)
        return confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            normalize=normalize,
        )

    @staticmethod
    def save_plot(
        matrix: np.ndarray,
        *,
        output_path: str | Path,
        class_labels: tuple[str, str] = ("Legitimate", "Fraud"),
        title: str = "Confusion Matrix",
    ) -> Path:
        """
        Save the confusion matrix as a PNG image.

        Writes atomically: the plot is rendered to a temporary file in
        the destination directory, then moved into place with
        os.replace, so a failure or interruption mid-render never
        leaves a corrupt or partial PNG at `output_path`.

        Parameters
        ----------
        matrix : ndarray
            Confusion matrix. Must be 2-D and square.
        output_path : str | Path
            Destination PNG path.
        class_labels : tuple[str, str]
            Display labels. Must match matrix dimensions.
        title : str
            Plot title.

        Returns
        -------
        Path
            Saved image path.

        Raises
        ------
        ConfusionMatrixError
            If `matrix` is not 2-D/square, or if its size doesn't match
            `class_labels`.
        """
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ConfusionMatrixError(
                f"matrix must be a 2-D square array, got shape {matrix.shape}."
            )
        if matrix.shape[0] != len(class_labels):
            raise ConfusionMatrixError(
                f"matrix has {matrix.shape[0]} classes but class_labels has "
                f"{len(class_labels)} entries; they must match."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=class_labels,
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            disp.plot(
                cmap="Blues",
                colorbar=False,
                ax=ax,
            )
            ax.set_title(title)
            fig.tight_layout()

            fd, tmp_path_str = tempfile.mkstemp(
                dir=output_path.parent, prefix=".tmp-", suffix=".png"
            )
            os.close(fd)
            tmp_path = Path(tmp_path_str)
            try:
                fig.savefig(
                    tmp_path,
                    dpi=300,
                    bbox_inches="tight",
                    format="png",
                )
                os.replace(tmp_path, output_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        finally:
            plt.close(fig)

        return output_path
    