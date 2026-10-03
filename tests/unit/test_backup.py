import zipfile
from pathlib import Path

import pytest


def test_backup_restores_exact_files_to_new_directory_excluding_secrets(tmp_path):
    from annual_report_agent.storage.backup import backup, restore
    source = tmp_path / "source"
    (source / "vectors").mkdir(parents=True)
    (source / "application.sqlite").write_bytes(b"sqlite-example")
    (source / "vectors" / "index.bin").write_bytes(b"vectors")
    (source / ".env").write_text("secret", encoding="utf-8")
    archive = backup(source, tmp_path / "backup.zip")
    target = restore(archive, tmp_path / "restored")
    assert (target / "vectors/index.bin").read_bytes() == b"vectors"
    assert not (target / ".env").exists()
    with pytest.raises(ValueError, match="empty"):
        restore(archive, target)


def test_restore_rejects_archive_path_traversal_before_writing(tmp_path):
    from annual_report_agent.storage.backup import restore
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", "oops")
    with pytest.raises(ValueError):
        restore(archive, tmp_path / "restored")
    assert not (tmp_path / "outside.txt").exists()


def test_restore_relocates_stored_pdf_path_without_overwriting_original(tmp_path):
    from annual_report_agent.storage.backup import backup, restore
    from annual_report_agent.storage.sqlite_repo import Repository
    source = tmp_path / "source"
    (source / "files").mkdir(parents=True)
    pdf = source / "files" / "hash.pdf"
    pdf.write_bytes(b"%PDF-demo")
    repo = Repository(source / "application.sqlite")
    repo.put("report", "r", {"id": "r", "path": str(pdf.resolve())})
    target = restore(backup(source, tmp_path / "b.zip"), tmp_path / "restored")
    restored = Repository(target / "application.sqlite").get("report", "r")
    assert Path(restored["path"]) == target / "files/hash.pdf"
    assert repo.get("report", "r")["path"] == str(pdf.resolve())
