"""MinHash + banded LSH near-duplicate detection.

Exact-digest deduplication removes only byte-identical rows after
normalisation: on the Official view it dropped 2 rows out of 1,265,981, while
the reference standard expects 10-30% of a web-scale corpus to be duplicated.
Anything that differs by a timestamp, a boilerplate header, or one edited
sentence survives, and the model then sees it repeatedly.

Character shingles rather than words: the corpus is majority Chinese, where
word segmentation is itself a model and a wrong split silently changes the
signature. Character n-grams need no segmenter and behave the same across
scripts.

No new dependency. datasketch would do this, but the lock file is a controlled
artefact deployed to two hosts, and the numpy formulation below is short enough
that adding a dependency costs more than it saves.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np

MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1
# Reserved signature for a document with no shingles.
EMPTY_SENTINEL = MAX_HASH


@dataclass(frozen=True)
class MinHashConfig:
    """Signature geometry and the similarity threshold it implies.

    With ``bands`` b and ``rows`` r per band, two documents of Jaccard
    similarity s share a bucket with probability ``1 - (1 - s**r)**b``. The
    default 8x8 puts the steep part of that curve near 0.85: a 0.9-similar pair
    is caught 99% of the time, a 0.5-similar pair 3% of the time.
    """

    permutations: int = 64
    bands: int = 8
    shingle_size: int = 5
    seed: int = 20260511

    def __post_init__(self) -> None:
        if self.permutations <= 0 or self.bands <= 0:
            raise ValueError("permutations and bands must be positive")
        if self.permutations % self.bands:
            raise ValueError("permutations must divide evenly into bands")
        if self.shingle_size <= 0:
            raise ValueError("shingle_size must be positive")

    @property
    def rows_per_band(self) -> int:
        return self.permutations // self.bands

    def detection_probability(self, similarity: float) -> float:
        """Probability that a pair at this similarity lands in a shared bucket."""
        return 1.0 - (1.0 - similarity**self.rows_per_band) ** self.bands


def normalize_for_shingles(text: str) -> str:
    """NFKC, collapse whitespace, casefold.

    Matching should not depend on full-width versus half-width digits or on
    stray spacing, all of which differ across scrapes of the same document.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def shingle_hashes(text: str, shingle_size: int) -> np.ndarray:
    """Hash the set of character n-grams of a document."""
    normalized = normalize_for_shingles(text)
    if not normalized:
        return np.empty(0, dtype=np.uint64)
    if len(normalized) <= shingle_size:
        shingles = {normalized}
    else:
        shingles = {
            normalized[index : index + shingle_size]
            for index in range(len(normalized) - shingle_size + 1)
        }
    return np.fromiter(
        (hash_shingle(item) for item in shingles), dtype=np.uint64, count=len(shingles)
    )


def hash_shingle(shingle: str) -> int:
    """Stable 32-bit hash. Python's hash() is salted per process and unusable."""
    import hashlib

    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=4).digest(), "big")


def permutation_parameters(config: MinHashConfig) -> tuple[np.ndarray, np.ndarray]:
    """Draw the (a, b) coefficients of the universal hash family."""
    generator = np.random.default_rng(config.seed)
    a = generator.integers(1, MERSENNE_PRIME, size=config.permutations, dtype=np.uint64)
    b = generator.integers(0, MERSENNE_PRIME, size=config.permutations, dtype=np.uint64)
    return a, b


def signature(hashes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum of each permutation over the document's shingle hashes."""
    if hashes.size == 0:
        # Sentinel, recognised by duplicate_indices and excluded from
        # bucketing. Returning a constant here without that exclusion would
        # give every empty document an identical signature and collapse them
        # all into one near-duplicate group.
        return np.full(a.size, EMPTY_SENTINEL, dtype=np.uint32)
    permuted = (np.outer(a, hashes) + b[:, None]) % MERSENNE_PRIME
    return (permuted.min(axis=1) % (MAX_HASH + 1)).astype(np.uint32)


def band_keys(signatures: np.ndarray, config: MinHashConfig) -> np.ndarray:
    """Hash each band of each signature into a bucket key.

    ``signatures`` is (documents, permutations); the result is
    (documents, bands) of uint64 keys.
    """
    documents = signatures.shape[0]
    reshaped = signatures.reshape(documents, config.bands, config.rows_per_band)
    # FNV-1a over the band's bytes, vectorised across documents and bands.
    keys = np.full((documents, config.bands), np.uint64(14695981039346656037), dtype=np.uint64)
    for row in range(config.rows_per_band):
        keys ^= reshaped[:, :, row].astype(np.uint64)
        keys *= np.uint64(1099511628211)
    return keys


def duplicate_indices(signatures: np.ndarray, config: MinHashConfig) -> set[int]:
    """Indices to drop, keeping the first occurrence of each near-duplicate group.

    Order matters: keeping the earliest index makes the result independent of
    how the work was chunked across processes.
    """
    keys = band_keys(signatures, config)
    empty = np.all(signatures == EMPTY_SENTINEL, axis=1)
    duplicates: set[int] = set()
    representative: dict[tuple[int, int], int] = {}
    for band in range(config.bands):
        column = keys[:, band]
        order = np.argsort(column, kind="stable")
        sorted_keys = column[order]
        boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
        for raw_group in np.split(order, boundaries):
            group = raw_group[~empty[raw_group]]
            if group.size < 2:
                continue
            earliest = int(group.min())
            bucket = (band, int(column[group[0]]))
            representative.setdefault(bucket, earliest)
            for index in group:
                if int(index) != earliest:
                    duplicates.add(int(index))
    return duplicates
