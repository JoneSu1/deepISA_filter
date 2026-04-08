"""FASTA reader — unified approach using bioframe (deepISA native)."""
import bioframe as bf
from pathlib import Path
from typing import Optional, Union


class FastaReader:
    """
    Lightweight FASTA reader using bioframe.

    Supports two input modes (matching deepISA's native approach):
      - Single FASTA file: bf.load_fasta("/path/to/hg38.fa")
      - Directory of chr*.fa files: bf.load_fasta([...chr1.fa, ...])

    Parameters
    ----------
    fasta_path : str or Path
        Either:
          - Path to a single FASTA file (e.g. /path/to/hg38.fa or hg38.fa.gz)
          - Path to a directory containing chr*.fa files
    """

    def __init__(self, fasta_path: Union[str, Path]):
        fasta_path = Path(fasta_path)
        if fasta_path.is_file():
            self._fasta = bf.load_fasta(fasta_path)
            self._mode = "single"
        elif fasta_path.is_dir():
            fasta_files = sorted(fasta_path.glob("chr*.fa"))
            if not fasta_files:
                raise FileNotFoundError(
                    f"No 'chr*.fa' files found in directory: {fasta_path}"
                )
            self._fasta = bf.load_fasta(fasta_files)
            self._mode = "split"
        else:
            raise FileNotFoundError(
                f"FASTA path does not exist: {fasta_path}"
            )

    def fetch(self, chrom: str, start: int, end: int) -> Optional[str]:
        """Fetch uppercase sequence from FASTA, or None on failure."""
        try:
            return str(self._fasta[chrom][start:end]).upper()
        except Exception:
            return None
