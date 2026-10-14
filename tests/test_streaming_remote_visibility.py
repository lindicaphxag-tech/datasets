# Copyright 2026 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Retry only remote discovery, never a partially consumed sample stream."""

from unittest.mock import Mock

import pytest

from datasets.download.download_config import DownloadConfig
from datasets.utils import file_utils


REMOTE = "zstd://train.jsonl::hf://datasets/my-org/my-dataset@revision/train.jsonl.zst"


def test_transient_remote_shard_false_negative_recovers_without_duplicate(monkeypatch):
    probe = Mock(side_effect=[False, False, True])
    sleeps = []
    monkeypatch.setattr(file_utils, "xisfile", probe)
    monkeypatch.setattr(file_utils, "xisdir", Mock(return_value=False))
    monkeypatch.setattr(file_utils.time, "sleep", sleeps.append)

    paths = list(file_utils.FilesIterable._iter_from_urlpaths([REMOTE], DownloadConfig(max_retries=2)))
    assert paths == [REMOTE]
    assert probe.call_count == 3
    assert sleeps == [1, 2]


def test_permanently_absent_remote_shard_raises_and_never_skips(monkeypatch):
    probe = Mock(return_value=False)
    sleeps = []
    monkeypatch.setattr(file_utils, "xisfile", probe)
    monkeypatch.setattr(file_utils, "xisdir", Mock(return_value=False))
    monkeypatch.setattr(file_utils.time, "sleep", sleeps.append)

    with pytest.raises(FileNotFoundError, match="train.jsonl"):
        list(file_utils.FilesIterable._iter_from_urlpaths(REMOTE, DownloadConfig(max_retries=3)))
    assert probe.call_count == 4
    assert sleeps == [1, 2, 4]


@pytest.mark.parametrize("retries", [0, 1, 9])
def test_missing_local_file_is_not_retried(monkeypatch, tmp_path, retries):
    absent = str(tmp_path / "missing.jsonl")
    sleeps = Mock()
    monkeypatch.setattr(file_utils.time, "sleep", sleeps)
    with pytest.raises(FileNotFoundError):
        list(file_utils.FilesIterable._iter_from_urlpaths(absent, DownloadConfig(max_retries=retries)))
    sleeps.assert_not_called()


def test_remote_directory_listing_remains_unchanged_and_not_retried(monkeypatch):
    dirpath = "hf://datasets/example/testdir"
    sleeps = Mock()
    monkeypatch.setattr(file_utils, "xisfile", Mock(return_value=False))
    monkeypatch.setattr(file_utils, "xisdir", Mock(return_value=True))
    monkeypatch.setattr(
        file_utils,
        "xwalk",
        Mock(return_value=iter([(dirpath, ["subdir", ".ignored"], ["b.jsonl", ".hidden", "a.jsonl"])])),
    )
    monkeypatch.setattr(file_utils.time, "sleep", sleeps)
    paths = list(file_utils.FilesIterable._iter_from_urlpaths(dirpath, DownloadConfig(max_retries=2)))
    assert paths == [dirpath + "/a.jsonl", dirpath + "/b.jsonl"]
    sleeps.assert_not_called()


def test_iteration_does_not_restart_a_partially_consumed_directory(monkeypatch):
    directory = "hf://datasets/example/dataset"

    def broken_walk(*args, **kwargs):
        yield (directory, [], ["first.txt"])
        raise OSError("streaming interrupted after first yielded file")

    directory_probe = Mock(return_value=True)
    sleeps = Mock()
    monkeypatch.setattr(file_utils, "xisfile", Mock(return_value=False))
    monkeypatch.setattr(file_utils, "xisdir", directory_probe)
    monkeypatch.setattr(file_utils, "xwalk", broken_walk)
    monkeypatch.setattr(file_utils.time, "sleep", sleeps)
    stream = file_utils.FilesIterable._iter_from_urlpaths(directory, DownloadConfig(max_retries=5))
    assert next(stream) == directory + "/first.txt"
    with pytest.raises(OSError, match="streaming interrupted"):
        next(stream)
    assert directory_probe.call_count == 1
    sleeps.assert_not_called()


def test_retry_budget_is_per_shard_and_does_not_repeat_recovered_files(monkeypatch):
    one, two = "hf://datasets/example/one.jsonl", "hf://datasets/example/two.jsonl"
    probe = Mock(side_effect=[False, True, False, True])
    sleeps = []
    monkeypatch.setattr(file_utils, "xisfile", probe)
    monkeypatch.setattr(file_utils, "xisdir", Mock(return_value=False))
    monkeypatch.setattr(file_utils.time, "sleep", sleeps.append)
    found = list(file_utils.FilesIterable._iter_from_urlpaths([one, two], DownloadConfig(max_retries=1)))
    assert found == [one, two]
    assert probe.call_count == 4
    assert sleeps == [1, 1]


def test_warning_does_not_leak_remote_signed_urls(monkeypatch):
    secret_url = "hf://datasets/example/secret.jsonl?access_token=DO_NOT_LOG_THIS"
    monkeypatch.setattr(file_utils, "xisfile", Mock(side_effect=[False, True]))
    monkeypatch.setattr(file_utils, "xisdir", Mock(return_value=False))
    warning = Mock()
    monkeypatch.setattr(file_utils.logger, "warning", warning)
    monkeypatch.setattr(file_utils.time, "sleep", Mock())
    assert list(file_utils.FilesIterable._iter_from_urlpaths(secret_url, DownloadConfig(max_retries=1))) == [
        secret_url
    ]
    assert warning.call_count == 1
    assert "DO_NOT_LOG_THIS" not in str(warning.call_args)
