"""Offline Talker codec-token cache in ``<root>/<backbone>/<encoder_tag>/``, keyed by ``sample_id``."""

from __future__ import annotations

import fcntl
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

N_CODEBOOKS = {"qwen2_5_omni": 1, "qwen3_omni": 16}
SHARD_SIZE = 1024
MAX_CACHED_SHARDS = 8
INDEX_COLUMNS = ["sample_id", "speaker", "shard", "row_in_shard"]


@dataclass
class CacheRecord:
    sample_id: str
    codes: list[int] | list[list[int]]
    speaker: str | None = None


class CodecCache:
    """Several readers, one writer at a time: ``flush()`` holds an exclusive file lock."""

    def __init__(self, cache_root: str | Path, backbone: str, encoder_tag: str):
        if backbone not in N_CODEBOOKS:
            raise ValueError(f"Unknown backbone {backbone!r}; expected one of {sorted(N_CODEBOOKS)}")
        self.n_codebooks = N_CODEBOOKS[backbone]
        self.dir = Path(cache_root) / backbone / encoder_tag
        self.index_path = self.dir / "index.parquet"
        self._index: dict[str, dict] | None = None
        self._pending: dict[str, CacheRecord] = {}
        self._shard_tables: OrderedDict[str, pa.Table] = OrderedDict()

    @classmethod
    def from_dir(cls, cache_dir: str | Path) -> CodecCache:
        cache_dir = Path(cache_dir)
        return cls(cache_dir.parent.parent, cache_dir.parent.name, cache_dir.name)

    def _read_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        rows = pq.read_table(self.index_path, columns=INDEX_COLUMNS).to_pylist()
        return {row["sample_id"]: row for row in rows}

    @property
    def index(self) -> dict[str, dict]:
        if self._index is None:
            self._index = self._read_index()
        return self._index

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._pending or sample_id in self.index

    def lookup(self, sample_id: str) -> CacheRecord | None:
        if sample_id in self._pending:
            return self._pending[sample_id]
        meta = self.index.get(sample_id)
        if meta is None:
            return None
        codes = self._shard_table(meta["shard"]).column("codes")[int(meta["row_in_shard"])].as_py()
        return CacheRecord(sample_id, codes, meta["speaker"])

    def _shard_table(self, shard: str) -> pa.Table:
        table = self._shard_tables.get(shard)
        if table is None:
            table = pq.read_table(self.dir / shard, columns=["codes"])
            self._shard_tables[shard] = table
            if len(self._shard_tables) > MAX_CACHED_SHARDS:
                self._shard_tables.popitem(last=False)
        else:
            self._shard_tables.move_to_end(shard)
        return table

    def put(self, record: CacheRecord) -> None:
        if not record.codes:
            raise ValueError(f"{record.sample_id}: no codec frames")
        width = 1 if isinstance(record.codes[0], int) else len(record.codes[0])
        if width != self.n_codebooks:
            raise ValueError(f"{record.sample_id}: {width} codebooks per frame, expected {self.n_codebooks}")
        if record.sample_id in self:
            return
        self._pending[record.sample_id] = record
        if len(self._pending) >= SHARD_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ".write.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._flush_locked()

    def _flush_locked(self) -> None:
        index = self._read_index()
        new_rows: dict[str, list[dict]] = {}
        for record in self._pending.values():
            if record.sample_id in index:
                continue
            position = len(index)
            shard = f"shards/{position // SHARD_SIZE:06d}.parquet"
            index[record.sample_id] = {
                "sample_id": record.sample_id,
                "speaker": record.speaker,
                "shard": shard,
                "row_in_shard": position % SHARD_SIZE,
            }
            new_rows.setdefault(shard, []).append({"sample_id": record.sample_id, "codes": record.codes})
        self._pending = {}
        (self.dir / "shards").mkdir(exist_ok=True)
        for shard, rows in new_rows.items():
            path = self.dir / shard
            table = pa.Table.from_pylist(rows)
            if path.exists():
                table = pa.concat_tables([pq.read_table(path, columns=["sample_id", "codes"]), table])
            _write_atomic(table, path)
            self._shard_tables.pop(shard, None)
        if new_rows:
            _write_atomic(pa.Table.from_pylist(list(index.values())), self.index_path)
        self._index = index


def _write_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
