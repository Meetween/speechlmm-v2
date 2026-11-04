"""Utils for data preparation and preprocessing scripts for SpeechLMM-v2."""

from pathlib import Path
import json
import logging

from speechlmm.data.utils import datasets_dir

DATASETS_VERSION = "v1.0"
DEFAULT_CORPUS_VERSION = "v1.0"

# examples of "versions.json"
#{
#    "v1.1": { "commit": "4edd2b3c8e12778cebce8f2063a4e09c86843aff", "corpora": { "ami": "v1.1", "icsi": "v1.1" } },
#    "v1.0": { "commit": "4edd2b3c8e12778cebce8f2063a4e09c86843aff", "corpora": {} }
#}

logger = logging.getLogger(__name__)


def get_versions_or_defaults(corpora: list[str], version: str | None = None, path: Path | None = None) -> tuple[str | None, dict[str, str]]:
    path = datasets_dir() / "versions.json" if path is None else path
    commit = None
    version = DATASETS_VERSION if version is None else version
    versions = {corpus: DEFAULT_CORPUS_VERSION for corpus in corpora}
    datasets_versions = None

    if path.exists():
        datasets_versions = json.loads(path.read_text())
    else:
        logger.info("Version file %s is missing; using default corpus versions", path)

    if datasets_versions is not None and version in datasets_versions:
        entry = datasets_versions[version] or {}
        if "corpora" in entry:
            versions.update(entry["corpora"])
        if "commit" in entry:
            commit = entry["commit"]

    if datasets_versions is not None and commit is None:
        logger.error("dataset version %s described in %s does not contain the code commit", version, path)

    return commit, versions
