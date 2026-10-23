"""Notebook05 exact-member checkpoints and externally rooted completion index.

One controller owns the journal. Index identities are returned and printed for
the operator/owner to retain outside the runtime. A cold resume requires that
external reference; discovery, a self-hash or the newest filename is not trust.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import uuid
import zipfile

import stage7_joint_recipe_v1 as joint

h, auth, t = joint.h, joint.auth, joint.t
SCHEMA = "stage7_response_checkpoint_v1"


def relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name or name == "."):
        raise ValueError("checkpoint_unsafe_relative_member")
    return name


def member_map(members):
    if not isinstance(members, dict) or not members:
        raise ValueError("checkpoint_exact_nonempty_members_required")
    names = set(members)
    for name, identity in members.items():
        relative(name)
        auth.validate_identity(identity)
        if any(parent.as_posix() in names for parent in PurePosixPath(name).parents if parent.as_posix() != "."):
            raise ValueError("checkpoint_file_directory_collision")


def archive_inventory(archive, members):
    member_map(members)
    infos = archive.infolist()
    if len(infos) != len(members) or {item.filename for item in infos} != set(members):
        raise ValueError("checkpoint_archive_inventory_mismatch")
    for info in infos:
        if (info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
                or info.file_size != members[info.filename]["bytes"]):
            raise ValueError("checkpoint_archive_member_type_or_size_mismatch")


def _stream(source, destination, expected, progress=None):
    digest, count = hashlib.sha256(), 0
    while chunk := source.read(1024 * 1024):
        count += len(chunk)
        if count > expected["bytes"]:
            raise ValueError("checkpoint_member_larger_than_expected")
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
        if progress is not None:
            progress(len(chunk))
    if dict(bytes=count, sha256=digest.hexdigest()) != expected:
        raise ValueError("checkpoint_member_byte_identity_mismatch")


def verify_archive(path, members):
    with zipfile.ZipFile(path) as archive:
        archive_inventory(archive, members)
        for name in sorted(members):
            with archive.open(name) as source:
                _stream(source, None, members[name])


def publish(root, members, durable, label, binding):
    """Payload and two complete raw readbacks precede the immutable manifest."""
    root, durable = Path(root), Path(durable)
    if not isinstance(label, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,100}", label):
        raise ValueError("checkpoint_fixed_label_required")
    member_map(members)
    manifest_path = durable / (label + ".checkpoint.json")
    if manifest_path.exists():
        raise ValueError("existing_checkpoint_requires_external_identity")
    for name, expected in members.items():
        t.verify(root / name, expected)
    durable.mkdir(parents=True, exist_ok=True)
    staging = root / (label + ".checkpoint.zip.partial." + uuid.uuid4().hex)
    total, completed, last = sum(item["bytes"] for item in members.values()), 0, time.monotonic()
    def progress(count):
        nonlocal completed, last
        completed += count
        if time.monotonic() - last >= 20:
            print(h.incumbent.canonical(dict(phase="checkpoint_compress", unit=label, bytes=completed, total=total)), flush=True)
            last = time.monotonic()
    with zipfile.ZipFile(staging, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            with (root / name).open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                _stream(source, destination, members[name], progress)
    destination = durable / (label + ".checkpoint.zip")
    if destination.exists():
        # A different lossless ZIP wrapper is acceptable only for the same
        # fully verified member inventory; never overwrite it for its hash.
        verify_archive(destination, members)
        transport_identity = t.identity(destination)
        t.verify(destination, transport_identity)
        t.verify(destination, transport_identity)
    else:
        transport_identity = {key: value for key, value in t.durable_copy(staging, destination).items() if key in ("bytes", "sha256")}
    manifest = dict(schema_version=SCHEMA, binding=binding, file_name=destination.name,
                    transport=transport_identity, members=members, raw_byte_readbacks=2)
    local_manifest = root / (label + ".checkpoint.json.partial." + uuid.uuid4().hex)
    t.publish_local_bytes(local_manifest, (h.incumbent.canonical(manifest) + "\n").encode())
    result = t.durable_copy(local_manifest, manifest_path)
    reference = dict(file_name=manifest_path.name, identity={key: result[key] for key in ("bytes", "sha256")})
    staging.unlink(); local_manifest.unlink()  # Only owned, verified transport scratch.
    return reference


def read_manifest(durable, reference, binding):
    if set(reference) != {"file_name", "identity"}:
        raise ValueError("checkpoint_external_reference_required")
    relative(reference["file_name"])
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,100}\.checkpoint\.json", reference["file_name"]):
        raise ValueError("checkpoint_manifest_name_mismatch")
    manifest = joint.verified_json(Path(durable) / reference["file_name"], reference["identity"])
    if (set(manifest) != {"schema_version", "binding", "file_name", "transport", "members", "raw_byte_readbacks"}
            or manifest["schema_version"] != SCHEMA or manifest["binding"] != binding
            or type(manifest["raw_byte_readbacks"]) is not int or manifest["raw_byte_readbacks"] != 2
            or manifest["file_name"] != reference["file_name"].replace(".json", ".zip")):
        raise ValueError("checkpoint_manifest_contract_mismatch")
    member_map(manifest["members"])
    auth.validate_identity(manifest["transport"])
    return manifest


def restore(root, durable, reference, binding):
    root, durable = Path(root), Path(durable)
    manifest = read_manifest(durable, reference, binding)
    source_path = durable / manifest["file_name"]
    t.verify(source_path, manifest["transport"])
    with zipfile.ZipFile(source_path) as archive:
        archive_inventory(archive, manifest["members"])
        for name, expected in sorted(manifest["members"].items()):
            destination = root / name
            if destination.exists():
                t.verify(destination, expected)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial." + uuid.uuid4().hex)
            with archive.open(name) as source, partial.open("xb") as target:
                _stream(source, target, expected)
                target.flush(); os.fsync(target.fileno())
            try:
                os.link(partial, destination)
            except FileExistsError:
                t.verify(destination, expected)
            partial.unlink()
    return manifest


class Journal:
    """Immutable serial completion indices; no latest-file trust on restart."""
    def __init__(self, root, durable, binding, expected_index=None):
        self.root, self.durable, self.binding = Path(root), Path(durable), binding
        self.entries, self.sequence, self.reference = {}, 0, None
        if expected_index is None:
            if self.durable.exists() and any(self.durable.iterdir()):
                raise ValueError("external_checkpoint_index_required_for_existing_state")
            return
        if set(expected_index) != {"file_name", "identity"} or not re.fullmatch(r"index_[0-9]{6}\.json", expected_index["file_name"]):
            raise ValueError("exact_external_completion_index_required")
        value = joint.verified_json(self.durable / expected_index["file_name"], expected_index["identity"])
        if (set(value) != {"schema_version", "binding", "sequence", "previous", "entries"}
                or value["schema_version"] != SCHEMA + "_index" or value["binding"] != binding
                or type(value["sequence"]) is not int or value["sequence"] < 1
                or expected_index["file_name"] != f'index_{value["sequence"]:06d}.json'
                or not isinstance(value["entries"], dict) or len(value["entries"]) != value["sequence"]):
            raise ValueError("completion_index_contract_mismatch")
        self.entries, self.sequence, self.reference = value["entries"], value["sequence"], expected_index
        for path in self.durable.iterdir():
            match = re.fullmatch(r"index_([0-9]{6})\.json", path.name)
            if match and int(match[1]) > self.sequence:
                raise ValueError("newer_completion_index_requires_external_authentication")
        if any(not isinstance(unit, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,100}", unit) for unit in self.entries):
            raise ValueError("completion_unit_identity_required")

    def record(self, unit, snapshot, receipt_path, receipt_identity):
        if not isinstance(unit, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,100}", unit):
            raise ValueError("completion_unit_identity_required")
        relative(receipt_path); auth.validate_identity(receipt_identity)
        entry = dict(snapshot=snapshot, receipt_path=receipt_path, receipt_identity=receipt_identity)
        if unit in self.entries:
            if self.entries[unit] != entry:
                raise ValueError("different_existing_completion")
            return self.reference
        checkpoint_binding = dict(controller=self.binding, unit=unit, receipt_identity=receipt_identity)
        manifest = read_manifest(self.durable, snapshot, checkpoint_binding)
        if manifest["members"].get(receipt_path) != receipt_identity:
            raise ValueError("completion_receipt_not_in_checkpoint")
        t.verify(self.root / receipt_path, receipt_identity)
        sequence = self.sequence + 1
        entries = dict(self.entries, **{unit: entry})
        value = dict(schema_version=SCHEMA + "_index", binding=self.binding, sequence=sequence,
                     previous=self.reference, entries=entries)
        local = self.root / (f"index_{sequence:06d}.json.partial." + uuid.uuid4().hex)
        t.publish_local_bytes(local, (h.incumbent.canonical(value) + "\n").encode())
        destination = self.durable / f"index_{sequence:06d}.json"
        copied = t.durable_copy(local, destination)
        reference = dict(file_name=destination.name, identity={key: copied[key] for key in ("bytes", "sha256")})
        self.entries, self.sequence, self.reference = entries, sequence, reference
        print(h.incumbent.canonical(dict(phase="durable_completion", unit=unit, completion_index=reference,
                                         receipt_identity=receipt_identity, completed_units=len(entries))), flush=True)
        local.unlink()
        return reference

    def restore_unit(self, unit, expected_receipt_path=None):
        entry = self.entries[unit]
        if set(entry) != {"snapshot", "receipt_path", "receipt_identity"}:
            raise ValueError("completion_entry_schema_mismatch")
        relative(entry["receipt_path"]); auth.validate_identity(entry["receipt_identity"])
        if expected_receipt_path is not None and entry["receipt_path"] != expected_receipt_path:
            raise ValueError("completion_receipt_path_mismatch")
        binding = dict(controller=self.binding, unit=unit, receipt_identity=entry["receipt_identity"])
        manifest = read_manifest(self.durable, entry["snapshot"], binding)
        if manifest["members"].get(entry["receipt_path"]) != entry["receipt_identity"]:
            raise ValueError("completion_receipt_not_in_checkpoint")
        restore(self.root, self.durable, entry["snapshot"], binding)
        t.verify(self.root / entry["receipt_path"], entry["receipt_identity"])
        return entry["receipt_identity"]
