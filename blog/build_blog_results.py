"""Freeze production pipeline evidence used by the bilingual blog post.

The script runs the same target construction, feasible persistence benchmark,
and purged walk-forward evaluation as the command-line research workflow. It
then copies only the article's compact evidence files into ``blog/data``.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from futures_roll_overlay.pipeline.run_research_pipeline import run_research_pipeline


BLOG_ROOT = Path(__file__).resolve().parent
DATA_DIR = BLOG_ROOT / "data"


def main() -> None:
    """Run production research and freeze the blog's tabular evidence.

    Returns
    -------
    None
        The function writes metrics, predictions, and coefficients under
        ``blog/data``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="futures-roll-blog-") as temporary_directory:
        result = run_research_pipeline(output_root=Path(temporary_directory))
        result["metrics"].to_csv(DATA_DIR / "metrics.csv", index=False)
        result["predictions"].reset_index().to_parquet(
            DATA_DIR / "predictions.parquet",
            index=False,
        )
        result["feature_importance"].to_csv(
            DATA_DIR / "feature_importance.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
