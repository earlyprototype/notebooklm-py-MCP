"""Exercise local export permissions with real Windows ACLs and a synthetic SDK download."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from notebooklm_mcp_server import export_artifact

if os.name == "nt":
    import win32api
    import win32con
    import win32security as security

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Requires Windows file ACLs")


def set_protected_dacl(path, sddl):
    descriptor = security.ConvertStringSecurityDescriptorToSecurityDescriptor(sddl, 1)
    security.SetNamedSecurityInfo(
        str(path),
        security.SE_FILE_OBJECT,
        security.DACL_SECURITY_INFORMATION | security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        descriptor.GetSecurityDescriptorDacl(),
        None,
    )


def dacl(path):
    info = security.DACL_SECURITY_INFORMATION
    descriptor = security.GetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT, info)
    return security.ConvertSecurityDescriptorToStringSecurityDescriptor(descriptor, 1, info)


@pytest.mark.parametrize("permissions", ["owner-only", "inherited", "moved"])
async def test_export_preserves_dacl(mock_ctx, mock_client, tmp_path, permissions):
    # Make the parent readable by Users even when pytest's temporary directory is private.
    set_protected_dacl(tmp_path, "D:P(A;OICI;FA;;;OW)(A;OICI;FR;;;BU)")
    path = tmp_path / "map.json"
    if permissions == "moved":
        private = tmp_path / "private"
        private.mkdir()
        set_protected_dacl(private, "D:P(A;OICI;FA;;;OW)")
        original = private / "map.json"
        original.write_text("{}", encoding="utf-8")
        original.rename(path)
        private.rmdir()
    else:
        path.write_text("{}", encoding="utf-8")
    if permissions == "owner-only":
        set_protected_dacl(path, "D:P(A;;0x1301bf;;;OW)")  # Modify access for the owner.
    before = dacl(path)
    token = security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    user_sid = security.ConvertSidToStringSid(security.GetTokenInformation(token, security.TokenUser)[0])

    def download(notebook_id, output_path, *, artifact_id):
        staged = Path(output_path)
        staged.write_text(json.dumps({"id": artifact_id}), encoding="utf-8")
        acl = security.GetFileSecurity(str(staged), security.DACL_SECURITY_INFORMATION).GetSecurityDescriptorDacl()
        trustees = {security.ConvertSidToStringSid(acl.GetAce(i)[2]) for i in range(acl.GetAceCount())}
        # Only the current user, owner, SYSTEM and Administrators may have access.
        assert trustees and trustees <= {user_sid, "S-1-3-4", "S-1-5-18", "S-1-5-32-544"}

    mock_client.artifacts.get.return_value = SimpleNamespace(kind="mind_map")
    mock_client.artifacts.download_mind_map.side_effect = download
    result = await export_artifact("nb-1", "replacement", str(path), ctx=mock_ctx)
    assert result == {"success": True, "output_path": str(path)}
    assert json.loads(path.read_text(encoding="utf-8")) == {"id": "replacement"}
    assert dacl(path) == before
    assert list(tmp_path.iterdir()) == [path]


async def test_failed_dacl_copy_preserves_original(mock_ctx, mock_client, tmp_path):
    path = tmp_path / "map.json"
    path.write_bytes(b'{"original": true}')
    before = dacl(path)

    def download(notebook_id, output_path, *, artifact_id):
        Path(output_path).write_bytes(b'{"replacement": true}')
        # Deny WRITE_DAC so Windows cannot apply the original DACL to the download.
        set_protected_dacl(output_path, "D:P(D;;WD;;;OW)(A;;FA;;;OW)")

    mock_client.artifacts.get.return_value = SimpleNamespace(kind="mind_map")
    mock_client.artifacts.download_mind_map.side_effect = download
    result = await export_artifact("nb-1", "art-2", str(path), ctx=mock_ctx)
    assert "error" in result
    assert path.read_bytes() == b'{"original": true}'
    assert dacl(path) == before
    assert list(tmp_path.iterdir()) == [path]
