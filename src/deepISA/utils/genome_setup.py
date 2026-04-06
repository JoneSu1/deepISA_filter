"""
hg38 genome auto-setup.

Downloads hg38 from UCSC, decompresses, and splits into per-chromosome FASTA files
required by the pipeline.

Usage:
    from deepISA.utils.genome_setup import ensure_hg38
    genome_dir = ensure_hg38("./data/genome")
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
CHROMS_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chromFa.tar.gz"


def _find_samtools() -> str:
    """Locate samtools binary. Raises RuntimeError if not found."""
    path = os.environ.get("SAMTOOLS", "")
    if path and Path(path).exists():
        return path

    for candidate in [
        "samtools",
        "/usr/bin/samtools",
        "/usr/local/bin/samtools",
        "/home/linuxbrew/.linuxbrew/bin/samtools",
    ]:
        if Path(candidate).exists():
            return candidate
        try:
            subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                timeout=5,
            )
            return candidate
        except Exception:
            pass

    raise RuntimeError(
        "samtools not found. Install: sudo apt install samtools  # Linux\n"
        "or brew install samtools  # Mac\n"
        "Or set SAMTOOLS env var to the samtools binary path."
    )


def _run_cmd(cmd: list, cwd: str | None = None, desc: str = "") -> None:
    """Run a shell command, streaming output."""
    print(f"[genome_setup] {desc}: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_chroms_from_fasta(
    fasta_path: Union[str, Path],
    chroms_dir: Union[str, Path],
    quiet: bool = False,
) -> str:
    """
    Split a multi-chromosome FASTA file into per-chromosome chr*.fa files.

    This is useful when you have a single hg38.fa file (e.g., from deepISA)
    but need chr*.fa files for the motif filter pipeline.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the source FASTA file (e.g., /path/to/hg38.fa).
    chroms_dir : str or Path
        Target directory for per-chromosome FASTA files.
        Will be created if it doesn't exist.
    quiet : bool
        Suppress progress output.

    Returns
    -------
    str
        Path to the chroms directory (chr*.fa files).

    Raises
    ------
    FileNotFoundError
        If the source FASTA file doesn't exist.
    RuntimeError
        If samtools is not installed.

    Examples
    --------
    >>> from deepISA.utils.genome_setup import prepare_chroms_from_fasta
    >>> chroms_dir = prepare_chroms_from_fasta(
    ...     fasta_path="/path/to/hg38.fa",
    ...     chroms_dir="./data/genome/hg38/chroms",
    ... )
    """
    fasta_path = Path(fasta_path)
    chroms_dir = Path(chroms_dir)

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"Source FASTA not found: {fasta_path}\n"
            "Please provide the path to your hg38.fa file."
        )

    # Check if already done
    if chroms_dir.exists() and list(chroms_dir.glob("chr1.fa")):
        if not quiet:
            print(f"[genome_setup] chr*.fa already exist at {chroms_dir}")
        return str(chroms_dir)

    samtools = _find_samtools()

    # Index if needed
    fai_file = fasta_path.with_suffix(fasta_path.suffix + ".fai")
    if not fai_file.exists():
        if not quiet:
            print(f"[genome_setup] Indexing {fasta_path} with samtools ...")
        subprocess.run([samtools, "faidx", str(fasta_path)], check=True)

    # Create output dir
    chroms_dir.mkdir(parents=True, exist_ok=True)

    # Split
    with open(fai_file) as f:
        chrom_names = [line.split()[0] for line in f if line.strip()]

    if not quiet:
        print(f"[genome_setup] Splitting into {len(chrom_names)} chromosome files ...")

    for chrom in chrom_names:
        subprocess.run(
            [samtools, "faidx", str(fasta_path), chrom],
            stdout=(chroms_dir / f"{chrom}.fa").open("w"),
            check=True,
        )

    if not quiet:
        print(f"[genome_setup] Created {len(chrom_names)} files in {chroms_dir}")

    return str(chroms_dir)


def ensure_hg38(
    genome_dir: str = "./data/genome",
    chroms_subdir: str = "chroms",
    quiet: bool = False,
) -> str:
    """
    Ensures hg38 genome is available locally.

    If the genome is not present, downloads and decompresses hg38 from UCSC,
    then splits it into per-chromosome FASTA files required by the pipeline.

    Parameters
    ----------
    genome_dir : str
        Base directory to store genome data (default: ./data/genome).
        Structure: genome_dir/hg38.fa.gz and genome_dir/chroms/chr*.fa
    chroms_subdir : str
        Subdirectory name for per-chromosome FASTA files (default: chroms).
    quiet : bool
        Suppress download progress output.

    Returns
    -------
    str
        Path to the chroms directory (chr*.fa files).

    Raises
    ------
    RuntimeError
        If samtools is not installed or download fails.
    """
    base = Path(genome_dir)
    chroms_dir = base / chroms_subdir

    if chroms_dir.exists() and list(chroms_dir.glob("chr1.fa")):
        print(f"[genome_setup] hg38 chroms already exist at {chroms_dir}")
        return str(chroms_dir)

    base.mkdir(parents=True, exist_ok=True)

    samtools = _find_samtools()

    # ── Download hg38 ─────────────────────────────────────────
    fa_gz = base / "hg38.fa.gz"
    if not fa_gz.exists():
        print(f"[genome_setup] Downloading hg38 (~1 GB) ...")
        print(f"[genome_setup] Source: {HG38_URL}")
        try:
            subprocess.run(
                ["wget", "-O", str(fa_gz), HG38_URL],
                check=True,
            )
        except Exception:
            # fallback: try curl
            try:
                subprocess.run(
                    ["curl", "-L", "-o", str(fa_gz), HG38_URL],
                    check=True,
                )
            except Exception as e:
                fa_gz.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download failed. Install wget or curl, then retry.\n{e}"
                )
    else:
        print(f"[genome_setup] hg38.tar.gz already exists, skipping download.")

    # ── Decompress ───────────────────────────────────────────
    fa_file = base / "hg38.fa"
    if not fa_file.exists():
        print("[genome_setup] Decompressing ...")
        with open(fa_file, "wb") as dst:
            proc = subprocess.Popen(
                ["gunzip", "-c", str(fa_gz)],
                stdout=dst,
            )
            proc.wait()
        print(f"[genome_setup] Decompressed to {fa_file}")
    else:
        print(f"[genome_setup] hg38.fa already exists.")

    # ── Index ────────────────────────────────────────────────
    fai_file = Path(str(fa_file) + ".fai")
    if not fai_file.exists():
        print("[genome_setup] Indexing with samtools faidx ...")
        subprocess.run(
            [samtools, "faidx", str(fa_file)],
            check=True,
        )
        print(f"[genome_setup] Index written to {fai_file}")
    else:
        print("[genome_setup] Index already exists.")

    # ── Split into chr*.fa ──────────────────────────────────
    if not chroms_dir.exists():
        chroms_dir.mkdir(parents=True, exist_ok=True)
        print("[genome_setup] Splitting into per-chromosome FASTA files ...")

        with open(fai_file) as f:
            chrom_names = [line.split()[0] for line in f if line.strip()]

        for chrom in chrom_names:
            subprocess.run(
                [samtools, "faidx", str(fa_file), chrom],
                stdout=(chroms_dir / f"{chrom}.fa").open("w"),
                check=True,
            )

        print(
            f"[genome_setup] Created {len(chrom_names)} chromosome files in {chroms_dir}"
        )
    else:
        print(f"[genome_setup] Per-chromosome files already exist at {chroms_dir}")

    return str(chroms_dir)


def check_samtools() -> bool:
    """Return True if samtools is installed and usable."""
    try:
        _find_samtools()
        return True
    except RuntimeError:
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and setup hg38 genome")
    parser.add_argument(
        "--genome-dir",
        default="./data/genome",
        help="Base directory for genome files (default: ./data/genome)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if samtools is installed, do not download",
    )
    args = parser.parse_args()

    if args.check:
        try:
            samtools = _find_samtools()
            print(f"samtools found: {samtools}")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        path = ensure_hg38(args.genome_dir)
        print(f"Done. Genome ready at: {path}")
