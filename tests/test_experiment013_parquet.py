from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

import recurquant.experiment013_parquet as parquet_module
from recurquant.experiment013_parquet import (
    EXPERIMENT013_PARQUET_MANIFEST_PATH,
    EXPERIMENT013_PARQUET_MANIFEST_SCHEMA,
    EXPERIMENT013_PARQUET_MANIFEST_SHA256,
    EXPERIMENT013_PARQUET_MANIFEST_SIZE_BYTES,
    Experiment013ParquetError,
    Experiment013ParquetOffsetError,
    HubDatasetMetadata,
    HubFileMetadata,
    ParquetFileLayout,
    canonical_experiment013_parquet_manifest_bytes,
    load_experiment013_parquet_manifest,
    locate_experiment013_parquet_row,
    project_experiment013_parquet_columns,
    read_experiment013_parquet_row,
    validate_experiment013_parquet_manifest,
)


def _raw_manifest() -> dict[str, object]:
    payload = json.loads(EXPERIMENT013_PARQUET_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class _FakeHubBackend:
    def __init__(self, *, events: list[str] | None = None) -> None:
        self.manifest = load_experiment013_parquet_manifest()
        self.events = events if events is not None else []
        self.calls: list[tuple[object, ...]] = []
        self.revision_calls: defaultdict[str, int] = defaultdict(int)
        self.snapshot_calls: defaultdict[str, int] = defaultdict(int)
        self.revision_overrides: dict[tuple[str, int], str] = {}
        self.snapshot_commit_overrides: dict[tuple[str, int], str] = {}
        self.file_overrides: dict[tuple[str, int], dict[str, object]] = {}
        self.reverse_snapshot: set[tuple[str, int]] = set()

    def resolve_dataset_revision(self, *, repo_id: str, revision: str) -> str:
        call_index = self.revision_calls[revision]
        self.revision_calls[revision] += 1
        self.calls.append(("resolve", repo_id, revision, call_index))
        self.events.append(f"hub-resolve-{call_index}")
        return self.revision_overrides.get((revision, call_index), revision)

    def snapshot_parquet_files(
        self,
        *,
        repo_id: str,
        revision: str,
        paths: tuple[str, ...],
    ) -> HubDatasetMetadata:
        call_index = self.snapshot_calls[revision]
        self.snapshot_calls[revision] += 1
        self.calls.append(("snapshot", repo_id, revision, paths, call_index))
        self.events.append(f"hub-snapshot-{call_index}")
        dataset = next(
            dataset
            for dataset in self.manifest.datasets
            if dataset.dataset_id == repo_id and dataset.conversion_revision == revision
        )
        expected_by_path = {file.immutable_path: file for file in dataset.files}
        metadata: list[HubFileMetadata] = []
        for path in paths:
            file = expected_by_path[path]
            observed = HubFileMetadata(
                path=file.immutable_path,
                commit_hash=dataset.conversion_revision,
                size_bytes=file.size_bytes,
                git_blob_oid=file.git_blob_oid,
                lfs_sha256=file.lfs_sha256,
                lfs_size_bytes=file.lfs_size_bytes,
                etag=file.lfs_sha256,
            )
            override = self.file_overrides.get((path, call_index))
            if override:
                observed = replace(observed, **override)
            metadata.append(observed)
        if (revision, call_index) in self.reverse_snapshot:
            metadata.reverse()
        return HubDatasetMetadata(
            commit_hash=self.snapshot_commit_overrides.get(
                (revision, call_index), dataset.conversion_revision
            ),
            files=tuple(metadata),
        )


class _FakeParquetBackend:
    def __init__(
        self,
        layouts: dict[str, ParquetFileLayout],
        *,
        events: list[str] | None = None,
    ) -> None:
        self.layouts = layouts
        self.events = events if events is not None else []
        self.inspect_calls: list[str] = []
        self.read_calls: list[tuple[str, int, int, tuple[str, ...]]] = []
        self.projection_calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.projection_overreturn = False
        self.projection_extra_column = False

    def inspect(self, uri: str) -> ParquetFileLayout:
        self.inspect_calls.append(uri)
        self.events.append("parquet-inspect")
        return self.layouts[uri]

    def read_row(
        self,
        uri: str,
        *,
        row_group_index: int,
        row_index_in_group: int,
        columns: tuple[str, ...],
    ) -> dict[str, object]:
        self.read_calls.append((uri, row_group_index, row_index_in_group, columns))
        self.events.append("parquet-read")
        return {
            column: f"{column}:group={row_group_index}:row={row_index_in_group}"
            for column in columns
        }

    def read_row_group_projection(
        self,
        uri: str,
        *,
        row_group_index: int,
        columns: tuple[str, ...],
    ) -> tuple[dict[str, object], ...]:
        self.projection_calls.append((uri, row_group_index, columns))
        self.events.append("parquet-project")
        count = self.layouts[uri].row_group_rows[row_group_index]
        if self.projection_overreturn:
            count += 1
        rows = []
        for row_index in range(count):
            row: dict[str, object] = {
                column: f"{column}:{uri.rsplit('/', 1)[-1]}:{row_group_index}:{row_index}"
                for column in columns
            }
            if self.projection_extra_column:
                row["text"] = "must-not-be-returned"
            rows.append(row)
        return tuple(rows)


def _uri(dataset_id: str, revision: str, path: str) -> str:
    return f"hf://datasets/{dataset_id}@{revision}/{path}"


def _parquet_backend(
    dataset_key: str,
    *,
    layouts_by_path: dict[str, tuple[int, ...]] | None = None,
    events: list[str] | None = None,
) -> _FakeParquetBackend:
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset(dataset_key)
    schema_columns = (
        ("url", "text")
        if dataset_key == "pg19"
        else ("prompt", "task_id", "canonical_solution")
    )
    layouts: dict[str, ParquetFileLayout] = {}
    for file in dataset.files:
        row_groups = (1,)
        if layouts_by_path is not None:
            row_groups = layouts_by_path.get(file.immutable_path, row_groups)
        layouts[_uri(dataset.dataset_id, dataset.conversion_revision, file.immutable_path)] = (
            ParquetFileLayout(
                row_group_rows=row_groups,
                columns=schema_columns,
            )
        )
    return _FakeParquetBackend(layouts, events=events)


def test_checked_in_manifest_is_canonical_byte_bound_and_exact() -> None:
    raw = EXPERIMENT013_PARQUET_MANIFEST_PATH.read_bytes()
    manifest = load_experiment013_parquet_manifest()
    parsed = _raw_manifest()

    assert len(raw) == EXPERIMENT013_PARQUET_MANIFEST_SIZE_BYTES
    assert hashlib.sha256(raw).hexdigest() == EXPERIMENT013_PARQUET_MANIFEST_SHA256
    assert canonical_experiment013_parquet_manifest_bytes(parsed) == raw
    assert manifest.schema == EXPERIMENT013_PARQUET_MANIFEST_SCHEMA
    assert tuple(dataset.key for dataset in manifest.datasets) == ("humaneval_plus", "pg19")
    assert manifest.dataset("humaneval_plus").dataset_id == "evalplus/humanevalplus"
    assert manifest.dataset("pg19").dataset_id == "emozilla/pg19"


def test_loader_rejects_locally_changed_manifest_bytes(tmp_path: Path) -> None:
    raw = bytearray(EXPERIMENT013_PARQUET_MANIFEST_PATH.read_bytes())
    marker = raw.index(b"humaneval_plus")
    raw[marker] = ord("j")
    changed = tmp_path / "materializations.json"
    changed.write_bytes(raw)

    with pytest.raises(Experiment013ParquetError, match="SHA-256"):
        load_experiment013_parquet_manifest(changed)


@pytest.mark.parametrize("level", ["top", "dataset", "file"])
def test_validation_rejects_extra_fields_at_every_level(level: str) -> None:
    payload = _raw_manifest()
    if level == "top":
        payload["mutable_revision"] = "main"
    elif level == "dataset":
        payload["datasets"]["pg19"]["endpoint"] = "viewer"  # type: ignore[index]
    else:
        payload["datasets"]["pg19"]["files"][0]["url"] = "mutable"  # type: ignore[index]

    with pytest.raises(Experiment013ParquetError, match="fields drifted"):
        validate_experiment013_parquet_manifest(payload)


@pytest.mark.parametrize(
    "case",
    [
        "dataset_repo",
        "source_alias",
        "conversion_alias",
        "split",
        "file_path",
        "file_order",
        "size",
        "git_hash",
        "lfs_hash",
        "lfs_size",
        "partial_type",
        "pending",
    ],
)
def test_validation_rejects_every_frozen_identity_drift(case: str) -> None:
    payload = copy.deepcopy(_raw_manifest())
    pg19 = payload["datasets"]["pg19"]  # type: ignore[index]
    files = pg19["files"]
    if case == "dataset_repo":
        pg19["dataset_id"] = "someone/pg19"
    elif case == "source_alias":
        pg19["source_revision"] = "main"
    elif case == "conversion_alias":
        pg19["conversion_revision"] = "refs/convert/parquet"
    elif case == "split":
        pg19["selected_splits"] = ["validation", "train"]
    elif case == "file_path":
        files[0]["immutable_path"] = "default/train/0000.parquet"
    elif case == "file_order":
        files[0], files[1] = files[1], files[0]
    elif case == "size":
        files[0]["size_bytes"] += 1
    elif case == "git_hash":
        files[0]["git_blob_oid"] = "0" * 40
    elif case == "lfs_hash":
        files[0]["lfs_sha256"] = "0" * 64
    elif case == "lfs_size":
        files[0]["lfs_size_bytes"] += 1
    elif case == "partial_type":
        pg19["partial"] = 1
    else:
        pg19["pending"] = ["default/partial-train/0007.parquet"]

    with pytest.raises(Experiment013ParquetError):
        validate_experiment013_parquet_manifest(payload)


def test_validation_rejects_malformed_manifest_shapes() -> None:
    payload = _raw_manifest()
    payload["datasets"]["pg19"]["files"] = "not-a-file-list"  # type: ignore[index]
    with pytest.raises(Experiment013ParquetError, match="files inventory"):
        validate_experiment013_parquet_manifest(payload)

    with pytest.raises(Experiment013ParquetError, match="must be a mapping"):
        validate_experiment013_parquet_manifest([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("offset", "file_index", "group_index", "row_in_group"),
    [
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (2, 0, 1, 0),
        (3, 1, 0, 0),
        (4, 1, 0, 1),
        (8, 5, 0, 0),
    ],
)
def test_global_offset_maps_across_file_and_row_group_boundaries(
    offset: int,
    file_index: int,
    group_index: int,
    row_in_group: int,
) -> None:
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("pg19")
    train_paths = [file.immutable_path for file in dataset.files if file.logical_split == "train"]
    layouts = {path: (1,) for path in train_paths}
    layouts[train_paths[0]] = (2, 1)
    layouts[train_paths[1]] = (2,)
    hub = _FakeHubBackend()
    parquet = _parquet_backend("pg19", layouts_by_path=layouts)

    location = locate_experiment013_parquet_row(
        "pg19",
        "train",
        offset,
        hub_backend=hub,
        parquet_backend=parquet,
    )

    assert location.split_row_count == 9
    assert location.split_file_index == file_index
    assert location.row_group_index == group_index
    assert location.row_index_in_group == row_in_group


@pytest.mark.parametrize("offset", [-1, 9, True])
def test_global_offset_rejects_negative_upper_boundary_and_bool(offset: int) -> None:
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("pg19")
    train_paths = [file.immutable_path for file in dataset.files if file.logical_split == "train"]
    layouts = {path: (1,) for path in train_paths}
    layouts[train_paths[0]] = (2, 1)
    layouts[train_paths[1]] = (2,)

    with pytest.raises(Experiment013ParquetOffsetError):
        locate_experiment013_parquet_row(
            "pg19",
            "train",
            offset,
            hub_backend=_FakeHubBackend(),
            parquet_backend=_parquet_backend("pg19", layouts_by_path=layouts),
        )


def test_read_projects_one_row_group_and_uses_only_immutable_hf_uris() -> None:
    events: list[str] = []
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("humaneval_plus")
    file = dataset.files[0]
    hub = _FakeHubBackend(events=events)
    parquet = _parquet_backend(
        "humaneval_plus",
        layouts_by_path={file.immutable_path: (2, 3)},
        events=events,
    )

    row = read_experiment013_parquet_row(
        "humaneval_plus",
        "test",
        3,
        columns=("task_id", "prompt"),
        hub_backend=hub,
        parquet_backend=parquet,
    )

    expected_uri = _uri(dataset.dataset_id, dataset.conversion_revision, file.immutable_path)
    assert row.location.immutable_uri == expected_uri
    assert row.location.row_group_index == 1
    assert row.location.row_index_in_group == 1
    assert row.columns == ("task_id", "prompt")
    assert tuple(row.values) == row.columns
    assert parquet.read_calls == [(expected_uri, 1, 1, ("task_id", "prompt"))]
    assert events == [
        "hub-resolve-0",
        "hub-snapshot-0",
        "parquet-inspect",
        "parquet-read",
        "hub-resolve-1",
        "hub-snapshot-1",
    ]
    contacted = repr(hub.calls + parquet.inspect_calls + parquet.read_calls)
    assert "/rows" not in contacted
    assert "@~parquet" not in contacted
    assert all(uri.startswith("hf://datasets/") for uri in parquet.inspect_calls)


def test_implementation_contains_no_dataset_viewer_or_mutable_parquet_endpoint() -> None:
    source = Path(parquet_module.__file__).read_text(encoding="utf-8")

    assert "/rows" not in source
    assert "@~parquet" not in source


def test_bulk_projection_is_ordered_counted_immutable_and_authenticated_once() -> None:
    events: list[str] = []
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("pg19")
    train_files = tuple(file for file in dataset.files if file.logical_split == "train")
    layouts = {file.immutable_path: (1,) for file in train_files}
    layouts[train_files[0].immutable_path] = (2, 1)
    hub = _FakeHubBackend(events=events)
    parquet = _parquet_backend("pg19", layouts_by_path=layouts, events=events)

    projection = project_experiment013_parquet_columns(
        "pg19",
        "train",
        columns=("url",),
        expected_count=8,
        hub_backend=hub,
        parquet_backend=parquet,
    )

    assert projection.columns == ("url",)
    assert len(projection.rows) == 8
    assert tuple(row.global_offset for row in projection.rows) == tuple(range(8))
    assert projection.rows[0].values[0].endswith("0000.parquet:0:0")
    assert projection.rows[1].values[0].endswith("0000.parquet:0:1")
    assert projection.rows[2].values[0].endswith("0000.parquet:1:0")
    assert projection.rows[3].values[0].endswith("0001.parquet:0:0")
    assert len(projection.canonical_projection_sha256) == 64
    assert hub.revision_calls[dataset.source_revision] == 2
    assert hub.snapshot_calls[dataset.conversion_revision] == 2
    assert len(parquet.inspect_calls) == 6
    assert len(parquet.projection_calls) == 7
    assert events[:2] == ["hub-resolve-0", "hub-snapshot-0"]
    assert events[-2:] == ["hub-resolve-1", "hub-snapshot-1"]
    assert all(call[2] == ("url",) for call in parquet.projection_calls)
    assert "/rows" not in repr(parquet.projection_calls)
    assert "@~parquet" not in repr(parquet.projection_calls)


@pytest.mark.parametrize(
    ("dataset_key", "logical_split", "columns"),
    [
        ("pg19", "train", ("text",)),
        ("humaneval_plus", "test", ("prompt", "canonical_solution")),
    ],
)
def test_bulk_projection_rejects_content_columns_before_external_access(
    dataset_key: str,
    logical_split: str,
    columns: tuple[str, ...],
) -> None:
    hub = _FakeHubBackend()
    parquet = _parquet_backend(dataset_key)

    with pytest.raises(Experiment013ParquetError, match="canonical-ID"):
        project_experiment013_parquet_columns(
            dataset_key,
            logical_split,
            columns=columns,
            hub_backend=hub,
            parquet_backend=parquet,
        )

    assert hub.calls == []
    assert parquet.inspect_calls == []
    assert parquet.projection_calls == []


def test_bulk_projection_rejects_schema_count_drift_before_reading_values() -> None:
    manifest = load_experiment013_parquet_manifest()
    file = manifest.dataset("humaneval_plus").files[0]
    hub = _FakeHubBackend()
    parquet = _parquet_backend(
        "humaneval_plus",
        layouts_by_path={file.immutable_path: (2, 3)},
    )

    with pytest.raises(Experiment013ParquetError, match="population"):
        project_experiment013_parquet_columns(
            "humaneval_plus",
            "test",
            columns=("task_id",),
            expected_count=4,
            hub_backend=hub,
            parquet_backend=parquet,
        )
    assert parquet.projection_calls == []

    with pytest.raises(Experiment013ParquetError, match="canonical-ID"):
        project_experiment013_parquet_columns(
            "humaneval_plus",
            "test",
            columns=("unknown_id",),
            hub_backend=_FakeHubBackend(),
            parquet_backend=parquet,
        )


@pytest.mark.parametrize("failure", ["overreturn", "extra_column"])
def test_bulk_projection_rejects_backend_overreturn(failure: str) -> None:
    parquet = _parquet_backend("humaneval_plus")
    if failure == "overreturn":
        parquet.projection_overreturn = True
        message = "row count"
    else:
        parquet.projection_extra_column = True
        message = "outside the projection"

    with pytest.raises(Experiment013ParquetError, match=message):
        project_experiment013_parquet_columns(
            "humaneval_plus",
            "test",
            columns=("task_id",),
            expected_count=1,
            hub_backend=_FakeHubBackend(),
            parquet_backend=parquet,
        )


def test_bulk_projection_rejects_metadata_drift_after_projection() -> None:
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("humaneval_plus")
    file = dataset.files[0]
    hub = _FakeHubBackend()
    hub.file_overrides[(file.immutable_path, 1)] = {"etag": "0" * 64}

    with pytest.raises(Experiment013ParquetError, match="after"):
        project_experiment013_parquet_columns(
            "humaneval_plus",
            "test",
            columns=("task_id",),
            expected_count=1,
            hub_backend=hub,
            parquet_backend=_parquet_backend("humaneval_plus"),
        )


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("path", "default/test/9999.parquet", "path"),
        ("commit_hash", "0" * 40, "commit"),
        ("size_bytes", 1, "size"),
        ("git_blob_oid", "0" * 40, "git_blob_oid"),
        ("lfs_sha256", "0" * 64, "lfs_sha256"),
        ("lfs_size_bytes", 1, "lfs_size_bytes"),
        ("etag", "0" * 64, "etag"),
    ],
)
def test_point_of_use_metadata_rejects_wrong_file_identity_before_read(
    field: str,
    wrong_value: object,
    message: str,
) -> None:
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("humaneval_plus")
    file = dataset.files[0]
    hub = _FakeHubBackend()
    hub.file_overrides[(file.immutable_path, 0)] = {field: wrong_value}
    parquet = _parquet_backend("humaneval_plus")

    with pytest.raises(Experiment013ParquetError, match=message):
        read_experiment013_parquet_row(
            "humaneval_plus",
            "test",
            0,
            hub_backend=hub,
            parquet_backend=parquet,
        )

    assert parquet.inspect_calls == []
    assert parquet.read_calls == []


def test_point_of_use_metadata_rejects_source_conversion_and_order_drift() -> None:
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("pg19")

    wrong_source = _FakeHubBackend()
    wrong_source.revision_overrides[(dataset.source_revision, 0)] = "0" * 40
    with pytest.raises(Experiment013ParquetError, match="source commit"):
        locate_experiment013_parquet_row(
            "pg19",
            "train",
            0,
            hub_backend=wrong_source,
            parquet_backend=_parquet_backend("pg19"),
        )

    wrong_conversion = _FakeHubBackend()
    wrong_conversion.snapshot_commit_overrides[(dataset.conversion_revision, 0)] = "0" * 40
    with pytest.raises(Experiment013ParquetError, match="conversion commit"):
        locate_experiment013_parquet_row(
            "pg19",
            "train",
            0,
            hub_backend=wrong_conversion,
            parquet_backend=_parquet_backend("pg19"),
        )

    wrong_order = _FakeHubBackend()
    wrong_order.reverse_snapshot.add((dataset.conversion_revision, 0))
    with pytest.raises(Experiment013ParquetError, match="path"):
        locate_experiment013_parquet_row(
            "pg19",
            "train",
            0,
            hub_backend=wrong_order,
            parquet_backend=_parquet_backend("pg19"),
        )


@pytest.mark.parametrize("drift", ["etag", "source", "conversion"])
def test_metadata_drift_after_row_read_fails_closed(drift: str) -> None:
    manifest = load_experiment013_parquet_manifest()
    dataset = manifest.dataset("humaneval_plus")
    file = dataset.files[0]
    hub = _FakeHubBackend()
    if drift == "etag":
        hub.file_overrides[(file.immutable_path, 1)] = {"etag": "0" * 64}
    elif drift == "source":
        hub.revision_overrides[(dataset.source_revision, 1)] = "0" * 40
    else:
        hub.snapshot_commit_overrides[(dataset.conversion_revision, 1)] = "0" * 40
    parquet = _parquet_backend("humaneval_plus")

    with pytest.raises(Experiment013ParquetError, match="after"):
        read_experiment013_parquet_row(
            "humaneval_plus",
            "test",
            0,
            columns=("prompt",),
            hub_backend=hub,
            parquet_backend=parquet,
        )

    assert len(parquet.read_calls) == 1


def test_projection_rejects_unknown_duplicate_and_backend_extra_columns() -> None:
    hub = _FakeHubBackend()
    parquet = _parquet_backend("humaneval_plus")
    with pytest.raises(Experiment013ParquetError, match="absent"):
        read_experiment013_parquet_row(
            "humaneval_plus",
            "test",
            0,
            columns=("missing",),
            hub_backend=hub,
            parquet_backend=parquet,
        )

    with pytest.raises(Experiment013ParquetError, match="unique"):
        read_experiment013_parquet_row(
            "humaneval_plus",
            "test",
            0,
            columns=("prompt", "prompt"),
            hub_backend=_FakeHubBackend(),
            parquet_backend=_parquet_backend("humaneval_plus"),
        )


def test_rejects_mutable_or_unknown_selection_at_api_boundary() -> None:
    with pytest.raises(Experiment013ParquetError, match="unknown"):
        locate_experiment013_parquet_row(
            "pg19@main",
            "train",
            0,
            hub_backend=_FakeHubBackend(),
            parquet_backend=_parquet_backend("pg19"),
        )
    with pytest.raises(Experiment013ParquetError, match="not frozen"):
        locate_experiment013_parquet_row(
            "pg19",
            "train@latest",
            0,
            hub_backend=_FakeHubBackend(),
            parquet_backend=_parquet_backend("pg19"),
        )
