"""Windows file operations for publishing local artifact exports."""

import shutil
from contextlib import contextmanager
from ctypes import WinError
from pathlib import Path
from uuid import uuid4

import pywintypes
import win32api
import win32con
import win32file
import win32security

_SE_DACL_AUTO_INHERIT_REQ = 0x0100


@contextmanager
def export_directory(dir: Path):
    directory = dir / f".notebooklm-{uuid4().hex}"
    try:
        volume = win32file.GetVolumePathName(str(dir.resolve()))
        if not win32api.GetVolumeInformation(volume)[3] & win32con.FILE_PERSISTENT_ACLS:
            raise OSError("The export volume does not support file ACLs.")
        # Install inheritable permissions at creation, including on Python before 3.13.
        attributes = pywintypes.SECURITY_ATTRIBUTES()
        attributes.SECURITY_DESCRIPTOR = win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
            "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", 1
        )
        win32file.CreateDirectory(str(directory), attributes)
    except pywintypes.error as exc:
        raise WinError(exc.winerror) from exc
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def publish_export(staged: Path, destination: Path) -> None:
    try:
        # On Windows rename refuses to overwrite a file that appeared during download.
        staged.rename(destination)
        return
    except FileExistsError:
        pass

    # ReplaceFileW can recompute inherited ACEs from the parent of a previously moved file.
    # Copy the stored DACL, including its owner for owner-relative access rules.
    info = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    try:
        original = win32security.GetFileSecurity(str(destination), info)
        expected = win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(original, 1, info)
        if original.GetSecurityDescriptorControl()[0] & win32security.SE_DACL_AUTO_INHERITED:
            # Windows retains AUTO_INHERITED when AUTO_INHERIT_REQ accompanies it.
            original.SetSecurityDescriptorControl(_SE_DACL_AUTO_INHERIT_REQ, _SE_DACL_AUTO_INHERIT_REQ)
        win32security.SetFileSecurity(str(staged), info, original)
        applied = win32security.GetFileSecurity(str(staged), info)
        if win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(applied, 1, info) != expected:
            raise OSError("Windows could not preserve the export destination's owner and DACL.")
    except pywintypes.error as exc:
        raise WinError(exc.winerror) from exc
    staged.replace(destination)
