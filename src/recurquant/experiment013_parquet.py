"""Immutable Parquet access for Experiment 013 calibration rows.

The checked-in materialization manifest is both schema-validated and bound by
its canonical file digest.  Live reads authenticate the immutable Hub source
and conversion commits plus every selected Parquet object's Git/LFS metadata
before and after one projected row-group read.

Network and Parquet operations sit behind small protocols so the identity and
offset contracts can be tested without network access or row data.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

EXPERIMENT013_PARQUET_MANIFEST_SCHEMA = (
    "recurquant.experiment013.parquet-materializations.v1"
)
EXPERIMENT013_PARQUET_MANIFEST_SHA256 = (
    "ee5628e50e5d3516fd79077542d355fd915455ac0e53128d372f4177ad63d39c"
)
EXPERIMENT013_PARQUET_MANIFEST_SIZE_BYTES = 3918
EXPERIMENT013_PARQUET_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "experiment013-parquet-materializations.json"
)

_TOP_LEVEL_FIELDS = frozenset({"schema", "datasets"})
_DATASET_FIELDS = frozenset(
    {
        "conversion_revision",
        "dataset_id",
        "failed",
        "files",
        "partial",
        "pending",
        "selected_splits",
        "source_revision",
    }
)
_FILE_FIELDS = frozenset(
    {
        "config",
        "git_blob_oid",
        "immutable_path",
        "lfs_sha256",
        "lfs_size_bytes",
        "logical_split",
        "size_bytes",
    }
)
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Experiment013ParquetError(RuntimeError):
    """Raised when immutable Experiment 013 Parquet access fails closed."""


class Experiment013ParquetOffsetError(IndexError):
    """Raised when a split-relative global row offset is outside its bounds."""


@dataclass(frozen=True, slots=True)
class Experiment013ParquetFile:
    """One exact Parquet materialization object."""

    config: str
    logical_split: str
    immutable_path: str
    size_bytes: int
    git_blob_oid: str
    lfs_sha256: str
    lfs_size_bytes: int


@dataclass(frozen=True, slots=True)
class Experiment013ParquetDataset:
    """Frozen source/conversion identity for one calibration dataset."""

    key: str
    dataset_id: str
    source_revision: str
    conversion_revision: str
    selected_splits: tuple[str, ...]
    partial: bool
    files: tuple[Experiment013ParquetFile, ...]


@dataclass(frozen=True, slots=True)
class Experiment013ParquetManifest:
    """Validated immutable materialization inventory."""

    schema: str
    datasets: tuple[Experiment013ParquetDataset, ...]

    def dataset(self, key: str) -> Experiment013ParquetDataset:
        for dataset in self.datasets:
            if dataset.key == key:
                return dataset
        raise Experiment013ParquetError(f"unknown Experiment 013 dataset key: {key!r}")


@dataclass(frozen=True, slots=True)
class HubFileMetadata:
    """Point-of-use Hub metadata for one immutable LFS object."""

    path: str
    commit_hash: str
    size_bytes: int
    git_blob_oid: str
    lfs_sha256: str
    lfs_size_bytes: int
    etag: str


@dataclass(frozen=True, slots=True)
class HubDatasetMetadata:
    """One ordered metadata snapshot at an exact conversion commit."""

    commit_hash: str
    files: tuple[HubFileMetadata, ...]


class HubMetadataBackend(Protocol):
    """Injectable metadata-only Hugging Face Hub boundary."""

    def resolve_dataset_revision(self, *, repo_id: str, revision: str) -> str:
        """Resolve an exact dataset revision without opening row content."""

    def snapshot_parquet_files(
        self,
        *,
        repo_id: str,
        revision: str,
        paths: tuple[str, ...],
    ) -> HubDatasetMetadata:
        """Return ordered Hub/LFS metadata for the requested immutable paths."""


@dataclass(frozen=True, slots=True)
class ParquetFileLayout:
    """Footer-only layout needed to locate a row."""

    row_group_rows: tuple[int, ...]
    columns: tuple[str, ...]


class ParquetBackend(Protocol):
    """Injectable Parquet footer and projected row-group boundary."""

    def inspect(self, uri: str) -> ParquetFileLayout:
        """Read only Parquet footer/schema metadata."""

    def read_row(
        self,
        uri: str,
        *,
        row_group_index: int,
        row_index_in_group: int,
        columns: tuple[str, ...],
    ) -> Mapping[str, object]:
        """Read one projected row from exactly one row group."""

    def read_row_group_projection(
        self,
        uri: str,
        *,
        row_group_index: int,
        columns: tuple[str, ...],
    ) -> Sequence[Mapping[str, object]]:
        """Read requested columns from exactly one ordered row group."""


@dataclass(frozen=True, slots=True)
class Experiment013ParquetRowLocation:
    """Deterministic split-relative global-offset resolution."""

    dataset_key: str
    dataset_id: str
    logical_split: str
    global_offset: int
    split_row_count: int
    manifest_file_index: int
    split_file_index: int
    immutable_path: str
    immutable_uri: str
    file_row_index: int
    row_group_index: int
    row_index_in_group: int
    row_group_row_count: int


@dataclass(frozen=True, slots=True)
class Experiment013ParquetRow:
    """One authenticated, projected Experiment 013 row."""

    location: Experiment013ParquetRowLocation
    columns: tuple[str, ...]
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Experiment013ParquetProjectionRow:
    """One immutable ID-only projection row in split-global order."""

    global_offset: int
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Experiment013ParquetProjection:
    """Complete deterministic projection and its canonical commitment."""

    dataset_key: str
    dataset_id: str
    logical_split: str
    columns: tuple[str, ...]
    rows: tuple[Experiment013ParquetProjectionRow, ...]
    canonical_projection_sha256: str


_EXPECTED_MANIFEST = Experiment013ParquetManifest(
    schema=EXPERIMENT013_PARQUET_MANIFEST_SCHEMA,
    datasets=(
        Experiment013ParquetDataset(
            key="humaneval_plus",
            dataset_id="evalplus/humanevalplus",
            source_revision="d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
            conversion_revision="1cf4467306a94e0828b355ff1f32e9222d2d588a",
            selected_splits=("test",),
            partial=False,
            files=(
                Experiment013ParquetFile(
                    config="default",
                    logical_split="test",
                    immutable_path="default/test/0000.parquet",
                    size_bytes=2_902_210,
                    git_blob_oid="9877db06683d4245bc39aed18ee7cbad013ba5fa",
                    lfs_sha256=(
                        "4436f5c03d77c17e0cbc57543b90665b5c1266f55a43992a5ed7922cd34a7558"
                    ),
                    lfs_size_bytes=2_902_210,
                ),
            ),
        ),
        Experiment013ParquetDataset(
            key="pg19",
            dataset_id="emozilla/pg19",
            source_revision="c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
            conversion_revision="b3624dc44b60cb01e74876e8869234d2660812cf",
            selected_splits=("train", "validation"),
            partial=True,
            files=(
                Experiment013ParquetFile(
                    config="default",
                    logical_split="train",
                    immutable_path="default/partial-train/0000.parquet",
                    size_bytes=603_127_902,
                    git_blob_oid="00245b214ff9806a04f32debff0fd2e7b0737997",
                    lfs_sha256=(
                        "ea701af2e8a11bb8601150a47affff658452d687494dbed52a82d3b1fcf48811"
                    ),
                    lfs_size_bytes=603_127_902,
                ),
                Experiment013ParquetFile(
                    config="default",
                    logical_split="train",
                    immutable_path="default/partial-train/0001.parquet",
                    size_bytes=526_793_959,
                    git_blob_oid="1169b6deb8c1cd46a46f0ae752b68806c9b5cca9",
                    lfs_sha256=(
                        "5c1c025f46b4a6b52b56167efeb89a2b9378f9ea8a50cdf5ddcbca8c4e17db1f"
                    ),
                    lfs_size_bytes=526_793_959,
                ),
                Experiment013ParquetFile(
                    config="default",
                    logical_split="train",
                    immutable_path="default/partial-train/0002.parquet",
                    size_bytes=576_668_259,
                    git_blob_oid="b9413777553240574488fa36b74f4bd286c06719",
                    lfs_sha256=(
                        "80cc198a2ef5239bf22a496eb10e6afd6fba075c4f6dd3d26dae7ed82c3bb1ad"
                    ),
                    lfs_size_bytes=576_668_259,
                ),
                Experiment013ParquetFile(
                    config="default",
                    logical_split="train",
                    immutable_path="default/partial-train/0003.parquet",
                    size_bytes=583_939_098,
                    git_blob_oid="08f92b1ad15eb9f90268fa7cf9823523f1fb056a",
                    lfs_sha256=(
                        "326718129b7d13a9f45ae8e6e68ae90d95c15bf40fa457a053716832e4d07c1c"
                    ),
                    lfs_size_bytes=583_939_098,
                ),
                Experiment013ParquetFile(
                    config="default",
                    logical_split="train",
                    immutable_path="default/partial-train/0004.parquet",
                    size_bytes=588_756_614,
                    git_blob_oid="9cb48d05cf6568879582eb5bd894a7d1b34aee7b",
                    lfs_sha256=(
                        "c4dff8b2cd993d1bb6bded41eb0eef56dff5449753ea13c79e730b7a9e1f6907"
                    ),
                    lfs_size_bytes=588_756_614,
                ),
                Experiment013ParquetFile(
                    config="default",
                    logical_split="train",
                    immutable_path="default/partial-train/0005.parquet",
                    size_bytes=321_273_724,
                    git_blob_oid="6a1b9b38d31ca025cbf06192a3dfd66067eb7571",
                    lfs_sha256=(
                        "9ab4d07d379720a9b18e7e3a060a948e2338b7aa338e9534a12d52fbc4fd8e2e"
                    ),
                    lfs_size_bytes=321_273_724,
                ),
                Experiment013ParquetFile(
                    config="default",
                    logical_split="validation",
                    immutable_path="default/partial-validation/0000.parquet",
                    size_bytes=10_803_864,
                    git_blob_oid="3e86263f595fae38387a938ec882417649c2bbd4",
                    lfs_sha256=(
                        "81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef"
                    ),
                    lfs_size_bytes=10_803_864,
                ),
            ),
        ),
    ),
)

# Bulk projection exists only to rank the complete frozen populations without
# decoding dataset content.  Keep this allow-list separate from the generic
# one-row reader, which deliberately projects the selected row's content.
_FROZEN_ID_PROJECTION_COLUMNS = {
    ("humaneval_plus", "test"): ("task_id",),
    ("pg19", "train"): ("url",),
    ("pg19", "validation"): ("url",),
}


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    if any(not isinstance(field, str) for field in value):
        raise Experiment013ParquetError(f"{name} field names must be strings")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Experiment013ParquetError(
            f"{name} fields drifted (missing={missing}, extra={extra})"
        )


def _expect_string(value: object, expected: str, *, name: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise Experiment013ParquetError(f"{name} drifted from its frozen value")
    return value


def _expect_commit(value: object, expected: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise Experiment013ParquetError(f"{name} must be an immutable lowercase commit SHA")
    return _expect_string(value, expected, name=name)


def _expect_sha1(value: object, expected: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise Experiment013ParquetError(f"{name} must be a lowercase SHA-1 object ID")
    return _expect_string(value, expected, name=name)


def _expect_sha256(value: object, expected: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment013ParquetError(f"{name} must be a lowercase SHA-256 digest")
    return _expect_string(value, expected, name=name)


def _expect_size(value: object, expected: int, *, name: str) -> int:
    if type(value) is not int or value != expected:  # bool is intentionally rejected
        raise Experiment013ParquetError(f"{name} drifted from its frozen byte size")
    return value


def _expect_path(value: object, expected: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Experiment013ParquetError(f"{name} must be a canonical immutable path")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise Experiment013ParquetError(f"{name} must be a canonical immutable POSIX path")
    return _expect_string(value, expected, name=name)


def validate_experiment013_parquet_manifest(
    manifest: Mapping[str, object],
) -> Experiment013ParquetManifest:
    """Strictly validate every field and ordered value in the frozen manifest."""

    if not isinstance(manifest, Mapping):
        raise Experiment013ParquetError("Parquet materialization manifest must be a mapping")
    _exact_fields(manifest, _TOP_LEVEL_FIELDS, name="Parquet materialization manifest")
    _expect_string(
        manifest["schema"],
        EXPERIMENT013_PARQUET_MANIFEST_SCHEMA,
        name="Parquet materialization schema",
    )
    raw_datasets = manifest["datasets"]
    if not isinstance(raw_datasets, Mapping):
        raise Experiment013ParquetError("datasets must be a mapping")
    expected_keys = tuple(dataset.key for dataset in _EXPECTED_MANIFEST.datasets)
    if tuple(raw_datasets) != expected_keys:
        raise Experiment013ParquetError("dataset keys or their canonical order drifted")

    for expected_dataset in _EXPECTED_MANIFEST.datasets:
        raw_dataset = raw_datasets[expected_dataset.key]
        name = f"datasets.{expected_dataset.key}"
        if not isinstance(raw_dataset, Mapping):
            raise Experiment013ParquetError(f"{name} must be a mapping")
        _exact_fields(raw_dataset, _DATASET_FIELDS, name=name)
        _expect_string(
            raw_dataset["dataset_id"],
            expected_dataset.dataset_id,
            name=f"{name}.dataset_id",
        )
        _expect_commit(
            raw_dataset["source_revision"],
            expected_dataset.source_revision,
            name=f"{name}.source_revision",
        )
        _expect_commit(
            raw_dataset["conversion_revision"],
            expected_dataset.conversion_revision,
            name=f"{name}.conversion_revision",
        )
        if type(raw_dataset["selected_splits"]) is not list or tuple(
            raw_dataset["selected_splits"]  # type: ignore[arg-type]
        ) != expected_dataset.selected_splits:
            raise Experiment013ParquetError(f"{name}.selected_splits or order drifted")
        if (
            type(raw_dataset["partial"]) is not bool
            or raw_dataset["partial"] is not expected_dataset.partial
        ):
            raise Experiment013ParquetError(f"{name}.partial drifted")
        for field in ("failed", "pending"):
            if type(raw_dataset[field]) is not list or raw_dataset[field]:
                raise Experiment013ParquetError(f"{name}.{field} must remain an empty list")

        raw_files = raw_dataset["files"]
        if type(raw_files) is not list or len(raw_files) != len(expected_dataset.files):
            raise Experiment013ParquetError(f"{name}.files inventory size drifted")
        for index, expected_file in enumerate(expected_dataset.files):
            raw_file = raw_files[index]
            file_name = f"{name}.files[{index}]"
            if not isinstance(raw_file, Mapping):
                raise Experiment013ParquetError(f"{file_name} must be a mapping")
            _exact_fields(raw_file, _FILE_FIELDS, name=file_name)
            _expect_string(raw_file["config"], expected_file.config, name=f"{file_name}.config")
            _expect_string(
                raw_file["logical_split"],
                expected_file.logical_split,
                name=f"{file_name}.logical_split",
            )
            _expect_path(
                raw_file["immutable_path"],
                expected_file.immutable_path,
                name=f"{file_name}.immutable_path",
            )
            _expect_size(
                raw_file["size_bytes"], expected_file.size_bytes, name=f"{file_name}.size_bytes"
            )
            _expect_sha1(
                raw_file["git_blob_oid"],
                expected_file.git_blob_oid,
                name=f"{file_name}.git_blob_oid",
            )
            _expect_sha256(
                raw_file["lfs_sha256"],
                expected_file.lfs_sha256,
                name=f"{file_name}.lfs_sha256",
            )
            _expect_size(
                raw_file["lfs_size_bytes"],
                expected_file.lfs_size_bytes,
                name=f"{file_name}.lfs_size_bytes",
            )
    return _EXPECTED_MANIFEST


def _manifest_as_dict(manifest: Experiment013ParquetManifest) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for dataset in manifest.datasets:
        datasets[dataset.key] = {
            "conversion_revision": dataset.conversion_revision,
            "dataset_id": dataset.dataset_id,
            "failed": [],
            "files": [
                {
                    "config": file.config,
                    "git_blob_oid": file.git_blob_oid,
                    "immutable_path": file.immutable_path,
                    "lfs_sha256": file.lfs_sha256,
                    "lfs_size_bytes": file.lfs_size_bytes,
                    "logical_split": file.logical_split,
                    "size_bytes": file.size_bytes,
                }
                for file in dataset.files
            ],
            "partial": dataset.partial,
            "pending": [],
            "selected_splits": list(dataset.selected_splits),
            "source_revision": dataset.source_revision,
        }
    return {"datasets": datasets, "schema": manifest.schema}


def canonical_experiment013_parquet_manifest_bytes(
    manifest: Mapping[str, object],
) -> bytes:
    """Validate and encode the exact canonical checked-in manifest bytes."""

    normalized = validate_experiment013_parquet_manifest(manifest)
    return (
        json.dumps(_manifest_as_dict(normalized), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _reject_json_number(value: str) -> object:
    raise Experiment013ParquetError(f"non-integer JSON number is forbidden: {value}")


def _reject_json_constant(value: str) -> object:
    raise Experiment013ParquetError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Experiment013ParquetError(f"duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def load_experiment013_parquet_manifest(
    path: str | Path = EXPERIMENT013_PARQUET_MANIFEST_PATH,
) -> Experiment013ParquetManifest:
    """Load the byte-bound canonical manifest, rejecting any local drift."""

    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise Experiment013ParquetError("Parquet materialization manifest must not be a symlink")
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise Experiment013ParquetError("Parquet materialization manifest is unreadable") from error
    if len(raw) != EXPERIMENT013_PARQUET_MANIFEST_SIZE_BYTES:
        raise Experiment013ParquetError("Parquet materialization manifest byte size drifted")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPERIMENT013_PARQUET_MANIFEST_SHA256:
        raise Experiment013ParquetError("Parquet materialization manifest SHA-256 drifted")
    try:
        text = raw.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment013ParquetError(
            "Parquet materialization manifest is not strict UTF-8 JSON"
        ) from error
    if not isinstance(parsed, Mapping):
        raise Experiment013ParquetError("Parquet materialization manifest must be a JSON object")
    normalized = validate_experiment013_parquet_manifest(parsed)
    if canonical_experiment013_parquet_manifest_bytes(parsed) != raw:
        raise Experiment013ParquetError("Parquet materialization manifest is not canonical JSON")
    return normalized


def _object_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


class HuggingFaceHubMetadataBackend:
    """Metadata-only implementation using official Hub APIs and exact revisions."""

    def __init__(self, *, token: str | bool | None = None) -> None:
        self._token = token

    def resolve_dataset_revision(self, *, repo_id: str, revision: str) -> str:
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=self._token).dataset_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=False,
            )
            resolved = info.sha
        except Exception as error:
            raise Experiment013ParquetError(
                f"Hub could not resolve immutable dataset revision for {repo_id}"
            ) from error
        if not isinstance(resolved, str):
            raise Experiment013ParquetError("Hub returned no resolved dataset commit")
        return resolved

    def snapshot_parquet_files(
        self,
        *,
        repo_id: str,
        revision: str,
        paths: tuple[str, ...],
    ) -> HubDatasetMetadata:
        try:
            from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

            info = HfApi(token=self._token).dataset_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=True,
            )
            siblings = {
                sibling.rfilename: sibling
                for sibling in (info.siblings or ())
                if isinstance(sibling.rfilename, str)
            }
            files: list[HubFileMetadata] = []
            for path in paths:
                sibling = siblings.get(path)
                if sibling is None:
                    raise Experiment013ParquetError(f"Hub metadata omitted immutable path: {path}")
                lfs = sibling.lfs
                if lfs is None:
                    raise Experiment013ParquetError(
                        f"Hub path is not an authenticated LFS object: {path}"
                    )
                url = hf_hub_url(
                    repo_id=repo_id,
                    filename=path,
                    repo_type="dataset",
                    revision=revision,
                )
                head = get_hf_file_metadata(url, token=self._token)
                files.append(
                    HubFileMetadata(
                        path=path,
                        commit_hash=head.commit_hash,
                        size_bytes=head.size,
                        git_blob_oid=sibling.blob_id,
                        lfs_sha256=_object_field(lfs, "sha256"),  # type: ignore[arg-type]
                        lfs_size_bytes=_object_field(lfs, "size"),  # type: ignore[arg-type]
                        etag=head.etag,
                    )
                )
            commit_hash = info.sha
        except Experiment013ParquetError:
            raise
        except Exception as error:
            raise Experiment013ParquetError(
                f"Hub could not authenticate immutable Parquet metadata for {repo_id}"
            ) from error
        if not isinstance(commit_hash, str):
            raise Experiment013ParquetError("Hub returned no conversion commit")
        return HubDatasetMetadata(commit_hash=commit_hash, files=tuple(files))


class PyArrowParquetBackend:
    """Read immutable ``hf://`` Parquet files through footer/range access."""

    @staticmethod
    def _dependencies() -> tuple[object, object]:
        try:
            import fsspec
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise Experiment013ParquetError(
                "PyArrow Parquet access requires fsspec and pyarrow"
            ) from error
        return fsspec, parquet

    def inspect(self, uri: str) -> ParquetFileLayout:
        fsspec, parquet = self._dependencies()
        try:
            with fsspec.open(uri, mode="rb") as stream:  # type: ignore[attr-defined]
                parquet_file = parquet.ParquetFile(stream)  # type: ignore[attr-defined]
                metadata = parquet_file.metadata
                row_group_rows = tuple(
                    metadata.row_group(index).num_rows
                    for index in range(metadata.num_row_groups)
                )
                columns = tuple(parquet_file.schema_arrow.names)
        except Exception as error:
            raise Experiment013ParquetError(
                f"could not inspect immutable Parquet footer: {uri}"
            ) from error
        return ParquetFileLayout(row_group_rows=row_group_rows, columns=columns)

    def read_row(
        self,
        uri: str,
        *,
        row_group_index: int,
        row_index_in_group: int,
        columns: tuple[str, ...],
    ) -> Mapping[str, object]:
        fsspec, parquet = self._dependencies()
        try:
            with fsspec.open(uri, mode="rb") as stream:  # type: ignore[attr-defined]
                parquet_file = parquet.ParquetFile(stream)  # type: ignore[attr-defined]
                table = parquet_file.read_row_group(row_group_index, columns=list(columns))
                if row_index_in_group >= table.num_rows:
                    raise Experiment013ParquetError("Parquet row-group layout changed during read")
                rows = table.slice(row_index_in_group, 1).to_pylist()
        except Experiment013ParquetError:
            raise
        except Exception as error:
            raise Experiment013ParquetError(
                f"could not read immutable Parquet row: {uri}"
            ) from error
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise Experiment013ParquetError("Parquet backend did not return exactly one row")
        return rows[0]

    def read_row_group_projection(
        self,
        uri: str,
        *,
        row_group_index: int,
        columns: tuple[str, ...],
    ) -> Sequence[Mapping[str, object]]:
        fsspec, parquet = self._dependencies()
        try:
            with fsspec.open(uri, mode="rb") as stream:  # type: ignore[attr-defined]
                parquet_file = parquet.ParquetFile(stream)  # type: ignore[attr-defined]
                rows = parquet_file.read_row_group(
                    row_group_index,
                    columns=list(columns),
                ).to_pylist()
        except Exception as error:
            raise Experiment013ParquetError(
                f"could not read immutable Parquet projection row group: {uri}"
            ) from error
        return rows


def _immutable_uri(
    dataset: Experiment013ParquetDataset,
    file: Experiment013ParquetFile,
) -> str:
    return (
        f"hf://datasets/{dataset.dataset_id}@{dataset.conversion_revision}/"
        f"{file.immutable_path}"
    )


def _selected_files(
    dataset: Experiment013ParquetDataset,
    logical_split: str,
) -> tuple[tuple[int, Experiment013ParquetFile], ...]:
    if not isinstance(logical_split, str) or logical_split not in dataset.selected_splits:
        raise Experiment013ParquetError(
            f"split {logical_split!r} is not frozen for dataset {dataset.key!r}"
        )
    files = tuple(
        (index, file)
        for index, file in enumerate(dataset.files)
        if file.logical_split == logical_split
    )
    if not files:
        raise Experiment013ParquetError("frozen split has no Parquet files")
    return files


def _authenticate_hub_snapshot(
    hub: HubMetadataBackend,
    dataset: Experiment013ParquetDataset,
    files: tuple[tuple[int, Experiment013ParquetFile], ...],
    *,
    phase: str,
) -> HubDatasetMetadata:
    try:
        source_commit = hub.resolve_dataset_revision(
            repo_id=dataset.dataset_id,
            revision=dataset.source_revision,
        )
        snapshot = hub.snapshot_parquet_files(
            repo_id=dataset.dataset_id,
            revision=dataset.conversion_revision,
            paths=tuple(file.immutable_path for _, file in files),
        )
    except Experiment013ParquetError:
        raise
    except Exception as error:
        raise Experiment013ParquetError(
            f"Hub metadata backend failed {phase} row access"
        ) from error
    if source_commit != dataset.source_revision:
        raise Experiment013ParquetError(f"source commit drifted {phase} row access")
    if not isinstance(snapshot, HubDatasetMetadata):
        raise Experiment013ParquetError(f"Hub metadata snapshot is malformed {phase} row access")
    if snapshot.commit_hash != dataset.conversion_revision:
        raise Experiment013ParquetError(f"conversion commit drifted {phase} row access")
    if len(snapshot.files) != len(files):
        raise Experiment013ParquetError(f"Hub file inventory drifted {phase} row access")
    for index, (observed, (_, expected)) in enumerate(zip(snapshot.files, files, strict=True)):
        name = f"Hub files[{index}] {phase} row access"
        if not isinstance(observed, HubFileMetadata):
            raise Experiment013ParquetError(f"{name} is malformed")
        _expect_path(observed.path, expected.immutable_path, name=f"{name}.path")
        _expect_commit(
            observed.commit_hash,
            dataset.conversion_revision,
            name=f"{name}.commit_hash",
        )
        _expect_size(observed.size_bytes, expected.size_bytes, name=f"{name}.size_bytes")
        _expect_sha1(
            observed.git_blob_oid,
            expected.git_blob_oid,
            name=f"{name}.git_blob_oid",
        )
        _expect_sha256(
            observed.lfs_sha256,
            expected.lfs_sha256,
            name=f"{name}.lfs_sha256",
        )
        _expect_size(
            observed.lfs_size_bytes,
            expected.lfs_size_bytes,
            name=f"{name}.lfs_size_bytes",
        )
        _expect_sha256(observed.etag, expected.lfs_sha256, name=f"{name}.etag")
    return snapshot


def _validated_layout(layout: object, *, uri: str) -> ParquetFileLayout:
    if not isinstance(layout, ParquetFileLayout):
        raise Experiment013ParquetError(f"Parquet backend returned a malformed layout: {uri}")
    if not layout.row_group_rows:
        raise Experiment013ParquetError(f"Parquet file has no row groups: {uri}")
    if any(type(count) is not int or count <= 0 for count in layout.row_group_rows):
        raise Experiment013ParquetError(f"Parquet row-group counts are invalid: {uri}")
    if (
        not layout.columns
        or any(not isinstance(column, str) or not column for column in layout.columns)
        or len(set(layout.columns)) != len(layout.columns)
    ):
        raise Experiment013ParquetError(f"Parquet schema columns are invalid: {uri}")
    return layout


def _inspect_selected_files(
    parquet: ParquetBackend,
    dataset: Experiment013ParquetDataset,
    files: tuple[tuple[int, Experiment013ParquetFile], ...],
) -> tuple[tuple[int, Experiment013ParquetFile, str, ParquetFileLayout], ...]:
    inspected: list[tuple[int, Experiment013ParquetFile, str, ParquetFileLayout]] = []
    for manifest_index, file in files:
        uri = _immutable_uri(dataset, file)
        try:
            raw_layout = parquet.inspect(uri)
        except Experiment013ParquetError:
            raise
        except Exception as error:
            raise Experiment013ParquetError(
                "Parquet backend failed while reading a footer"
            ) from error
        inspected.append((manifest_index, file, uri, _validated_layout(raw_layout, uri=uri)))
    return tuple(inspected)


def _locate_offset(
    dataset: Experiment013ParquetDataset,
    logical_split: str,
    global_offset: int,
    inspected: tuple[tuple[int, Experiment013ParquetFile, str, ParquetFileLayout], ...],
) -> Experiment013ParquetRowLocation:
    if type(global_offset) is not int or global_offset < 0:
        raise Experiment013ParquetOffsetError("global row offset must be a non-negative integer")
    split_row_count = sum(
        sum(layout.row_group_rows) for _, _, _, layout in inspected
    )
    if global_offset >= split_row_count:
        raise Experiment013ParquetOffsetError(
            f"global row offset {global_offset} is outside split row count {split_row_count}"
        )
    remaining = global_offset
    for split_file_index, (manifest_index, file, uri, layout) in enumerate(inspected):
        file_row_count = sum(layout.row_group_rows)
        if remaining >= file_row_count:
            remaining -= file_row_count
            continue
        file_row_index = remaining
        for row_group_index, row_group_row_count in enumerate(layout.row_group_rows):
            if remaining < row_group_row_count:
                return Experiment013ParquetRowLocation(
                    dataset_key=dataset.key,
                    dataset_id=dataset.dataset_id,
                    logical_split=logical_split,
                    global_offset=global_offset,
                    split_row_count=split_row_count,
                    manifest_file_index=manifest_index,
                    split_file_index=split_file_index,
                    immutable_path=file.immutable_path,
                    immutable_uri=uri,
                    file_row_index=file_row_index,
                    row_group_index=row_group_index,
                    row_index_in_group=remaining,
                    row_group_row_count=row_group_row_count,
                )
            remaining -= row_group_row_count
    raise Experiment013ParquetError("validated Parquet offset could not be resolved")


def _normalize_projection(
    columns: Sequence[str] | None,
    layout: ParquetFileLayout,
) -> tuple[str, ...]:
    if columns is None:
        return layout.columns
    if isinstance(columns, (str, bytes, bytearray)) or not isinstance(columns, Sequence):
        raise Experiment013ParquetError("columns must be a sequence of unique column names")
    normalized = tuple(columns)
    if (
        not normalized
        or any(not isinstance(column, str) or not column for column in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise Experiment013ParquetError("columns must be a non-empty unique string sequence")
    missing = [column for column in normalized if column not in layout.columns]
    if missing:
        raise Experiment013ParquetError(
            f"projected columns are absent from Parquet schema: {missing}"
        )
    return normalized


def _projection_sha256(
    *,
    dataset_key: str,
    logical_split: str,
    columns: tuple[str, ...],
    rows: tuple[Experiment013ParquetProjectionRow, ...],
) -> str:
    payload = {
        "columns": list(columns),
        "dataset_key": dataset_key,
        "logical_split": logical_split,
        "rows": [
            {"global_offset": row.global_offset, "values": list(row.values)} for row in rows
        ],
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_experiment013_parquet_columns(
    dataset_key: str,
    logical_split: str,
    *,
    columns: Sequence[str],
    expected_count: int | None = None,
    hub_backend: HubMetadataBackend | None = None,
    parquet_backend: ParquetBackend | None = None,
    manifest_path: str | Path = EXPERIMENT013_PARQUET_MANIFEST_PATH,
) -> Experiment013ParquetProjection:
    """Authenticate and read a complete ordered string-column projection.

    Hub metadata and all relevant Parquet footers are authenticated once
    before any projected values are decoded and once after the final row
    group. Only ``columns`` are passed to the backend. Values are restricted to
    strings because this surface exists solely for frozen canonical IDs.
    """

    if isinstance(columns, (str, bytes, bytearray)) or not isinstance(columns, Sequence):
        raise Experiment013ParquetError("ID projection columns must be a sequence")
    requested_columns = tuple(columns)
    frozen_columns = (
        _FROZEN_ID_PROJECTION_COLUMNS.get((dataset_key, logical_split))
        if isinstance(dataset_key, str) and isinstance(logical_split, str)
        else None
    )
    if requested_columns != frozen_columns:
        raise Experiment013ParquetError(
            "bulk Parquet projection is restricted to the frozen canonical-ID column"
        )
    if expected_count is not None and (
        type(expected_count) is not int or expected_count < 0
    ):
        raise Experiment013ParquetError("expected projection count must be a non-negative integer")
    hub = hub_backend if hub_backend is not None else HuggingFaceHubMetadataBackend()
    parquet = parquet_backend if parquet_backend is not None else PyArrowParquetBackend()
    manifest = load_experiment013_parquet_manifest(manifest_path)
    dataset = manifest.dataset(dataset_key)
    files = _selected_files(dataset, logical_split)
    before = _authenticate_hub_snapshot(hub, dataset, files, phase="before")
    inspected = _inspect_selected_files(parquet, dataset, files)
    projection: tuple[str, ...] | None = None
    split_row_count = sum(sum(layout.row_group_rows) for _, _, _, layout in inspected)
    if expected_count is not None and split_row_count != expected_count:
        raise Experiment013ParquetError(
            "Parquet footer row count differs from the frozen projection population"
        )
    rows: list[Experiment013ParquetProjectionRow] = []
    for _manifest_index, _file, uri, layout in inspected:
        normalized = _normalize_projection(requested_columns, layout)
        if projection is None:
            projection = normalized
        elif normalized != projection:
            raise Experiment013ParquetError("projected Parquet schemas differ across files")
        for row_group_index, row_group_count in enumerate(layout.row_group_rows):
            try:
                raw_rows = parquet.read_row_group_projection(
                    uri,
                    row_group_index=row_group_index,
                    columns=projection,
                )
            except Experiment013ParquetError:
                raise
            except Exception as error:
                raise Experiment013ParquetError(
                    "Parquet backend failed while reading an ID projection"
                ) from error
            if (
                isinstance(raw_rows, (str, bytes, bytearray))
                or not isinstance(raw_rows, Sequence)
                or len(raw_rows) != row_group_count
            ):
                raise Experiment013ParquetError(
                    "Parquet projection backend row count differs from authenticated footer"
                )
            for raw_row in raw_rows:
                if not isinstance(raw_row, Mapping) or set(raw_row) != set(projection):
                    raise Experiment013ParquetError(
                        "Parquet projection backend returned columns outside the projection"
                    )
                values: list[str] = []
                for column in projection:
                    value = raw_row[column]
                    if not isinstance(value, str) or not value:
                        raise Experiment013ParquetError(
                            "Parquet canonical-ID projection values must be non-empty strings"
                        )
                    values.append(value)
                rows.append(
                    Experiment013ParquetProjectionRow(
                        global_offset=len(rows),
                        values=tuple(values),
                    )
                )
    if projection is None:
        raise Experiment013ParquetError("Parquet projection contains no files")
    frozen_rows = tuple(rows)
    if len(frozen_rows) != split_row_count:
        raise Experiment013ParquetError("Parquet projection population drifted")
    after = _authenticate_hub_snapshot(hub, dataset, files, phase="after")
    if after != before:
        raise Experiment013ParquetError("Hub metadata changed during Parquet projection")
    return Experiment013ParquetProjection(
        dataset_key=dataset.key,
        dataset_id=dataset.dataset_id,
        logical_split=logical_split,
        columns=projection,
        rows=frozen_rows,
        canonical_projection_sha256=_projection_sha256(
            dataset_key=dataset.key,
            logical_split=logical_split,
            columns=projection,
            rows=frozen_rows,
        ),
    )


def locate_experiment013_parquet_row(
    dataset_key: str,
    logical_split: str,
    global_offset: int,
    *,
    hub_backend: HubMetadataBackend | None = None,
    parquet_backend: ParquetBackend | None = None,
    manifest_path: str | Path = EXPERIMENT013_PARQUET_MANIFEST_PATH,
) -> Experiment013ParquetRowLocation:
    """Authenticate and map one split-relative global offset without reading rows."""

    hub = hub_backend if hub_backend is not None else HuggingFaceHubMetadataBackend()
    parquet = parquet_backend if parquet_backend is not None else PyArrowParquetBackend()
    manifest = load_experiment013_parquet_manifest(manifest_path)
    dataset = manifest.dataset(dataset_key)
    files = _selected_files(dataset, logical_split)
    before = _authenticate_hub_snapshot(hub, dataset, files, phase="before")
    inspected = _inspect_selected_files(parquet, dataset, files)
    location = _locate_offset(dataset, logical_split, global_offset, inspected)
    after = _authenticate_hub_snapshot(hub, dataset, files, phase="after")
    if after != before:
        raise Experiment013ParquetError("Hub metadata changed during Parquet offset resolution")
    return location


def read_experiment013_parquet_row(
    dataset_key: str,
    logical_split: str,
    global_offset: int,
    *,
    columns: Sequence[str] | None = None,
    hub_backend: HubMetadataBackend | None = None,
    parquet_backend: ParquetBackend | None = None,
    manifest_path: str | Path = EXPERIMENT013_PARQUET_MANIFEST_PATH,
) -> Experiment013ParquetRow:
    """Read exactly one projected row from an immutable conversion commit.

    The split-relative offset is resolved from ordered Parquet footer row-group
    counts.  Only the containing row group and requested columns are decoded;
    the returned mapping contains exactly one row.
    """

    hub = hub_backend if hub_backend is not None else HuggingFaceHubMetadataBackend()
    parquet = parquet_backend if parquet_backend is not None else PyArrowParquetBackend()
    manifest = load_experiment013_parquet_manifest(manifest_path)
    dataset = manifest.dataset(dataset_key)
    files = _selected_files(dataset, logical_split)
    before = _authenticate_hub_snapshot(hub, dataset, files, phase="before")
    inspected = _inspect_selected_files(parquet, dataset, files)
    location = _locate_offset(dataset, logical_split, global_offset, inspected)
    target_layout = inspected[location.split_file_index][3]
    projection = _normalize_projection(columns, target_layout)
    try:
        raw_values = parquet.read_row(
            location.immutable_uri,
            row_group_index=location.row_group_index,
            row_index_in_group=location.row_index_in_group,
            columns=projection,
        )
    except Experiment013ParquetError:
        raise
    except Exception as error:
        raise Experiment013ParquetError("Parquet backend failed while reading one row") from error
    after = _authenticate_hub_snapshot(hub, dataset, files, phase="after")
    if after != before:
        raise Experiment013ParquetError("Hub metadata changed during Parquet row access")
    if not isinstance(raw_values, Mapping):
        raise Experiment013ParquetError("Parquet backend returned a non-mapping row")
    if set(raw_values) != set(projection):
        raise Experiment013ParquetError("Parquet backend returned columns outside the projection")
    values = {column: raw_values[column] for column in projection}
    return Experiment013ParquetRow(location=location, columns=projection, values=values)


__all__ = [
    "EXPERIMENT013_PARQUET_MANIFEST_PATH",
    "EXPERIMENT013_PARQUET_MANIFEST_SCHEMA",
    "EXPERIMENT013_PARQUET_MANIFEST_SHA256",
    "EXPERIMENT013_PARQUET_MANIFEST_SIZE_BYTES",
    "Experiment013ParquetDataset",
    "Experiment013ParquetError",
    "Experiment013ParquetFile",
    "Experiment013ParquetManifest",
    "Experiment013ParquetOffsetError",
    "Experiment013ParquetProjection",
    "Experiment013ParquetProjectionRow",
    "Experiment013ParquetRow",
    "Experiment013ParquetRowLocation",
    "HubDatasetMetadata",
    "HubFileMetadata",
    "HubMetadataBackend",
    "HuggingFaceHubMetadataBackend",
    "ParquetBackend",
    "ParquetFileLayout",
    "PyArrowParquetBackend",
    "canonical_experiment013_parquet_manifest_bytes",
    "load_experiment013_parquet_manifest",
    "locate_experiment013_parquet_row",
    "project_experiment013_parquet_columns",
    "read_experiment013_parquet_row",
    "validate_experiment013_parquet_manifest",
]
