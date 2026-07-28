"""Module 4 — Google Drive upload.

Uploads finished shorts to a Drive folder asynchronously and frees local
disk space once each upload is confirmed.

Uses PyDrive2 with a Service Account: headless, no browser login on every
run — required for a "factory" pipeline meant to run unattended.

One-time setup (see .env.example):
  1. Google Cloud Console -> new project -> enable "Google Drive API".
  2. Create a Service Account -> generate a JSON key -> save it locally.
  3. Create (or pick) a folder in YOUR regular Google Drive.
  4. Share that folder with the service account's email
     (looks like xxx@xxx.iam.gserviceaccount.com), Editor permission.
     Without step 4, uploads fail or the files "don't show up": a service
     account has no Drive storage of its own — it has to write inside a
     folder a real user explicitly shared with it.
  5. Copy that folder's ID (from its Drive URL) into GDRIVE_FOLDER_ID.
"""
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional


def _import_pydrive2():
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError as e:
        raise RuntimeError(
            "PyDrive2 is required for the Google Drive module. Install it with:\n"
            "    pip install PyDrive2"
        ) from e
    return GoogleAuth, GoogleDrive


class DriveUploader:
    """Thin wrapper over PyDrive2 with retries and a non-blocking upload."""

    def __init__(
        self,
        service_account_file: str,
        folder_id: str,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        if not os.path.exists(service_account_file):
            raise FileNotFoundError(
                f"Service account JSON not found: {service_account_file}\n"
                f"Set GDRIVE_SERVICE_ACCOUNT_FILE in your .env."
            )
        if not folder_id:
            raise ValueError("Empty folder_id. Set GDRIVE_FOLDER_ID in your .env.")

        self.service_account_file = service_account_file
        self.folder_id = folder_id
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._drive = self._authenticate()

    def _authenticate(self):
        GoogleAuth, GoogleDrive = _import_pydrive2()
        settings = {
            "client_config_backend": "service",
            "service_config": {"client_json_file_path": self.service_account_file},
        }
        gauth = GoogleAuth(settings=settings)
        gauth.ServiceAuth()
        return GoogleDrive(gauth)

    def _upload_sync(self, local_path: str, drive_filename: Optional[str] = None) -> Dict:
        """The real blocking upload — runs inside a worker thread via upload_async()."""
        gfile = self._drive.CreateFile({
            "title": drive_filename or Path(local_path).name,
            "parents": [{"id": self.folder_id}],
        })
        gfile.SetContentFile(local_path)
        gfile.Upload(param={"supportsAllDrives": True})
        return {
            "id": gfile["id"],
            "title": gfile["title"],
            "webViewLink": gfile.get("alternateLink"),
        }

    async def upload_async(self, local_path: str, drive_filename: Optional[str] = None) -> Dict:
        """Async wrapper via asyncio.to_thread. PyDrive2 itself is sync —
        google-api-python-client has no official async client — so we
        offload the blocking call to a thread instead of hand-rolling the
        resumable-upload protocol. Works from a plain script (asyncio.run)
        today, and from the FastAPI backend in Module 5 (await, non-blocking)
        without changes.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = await asyncio.to_thread(self._upload_sync, local_path, drive_filename)
                print(f"[drive] uploaded: {result['title']} -> {result.get('webViewLink')}", flush=True)
                return result
            except Exception as e:
                last_error = e
                wait = self.retry_backoff_seconds * (2 ** (attempt - 1))
                print(f"[drive] attempt {attempt}/{self.max_retries} failed for {local_path}: {e}", flush=True)
                if attempt < self.max_retries:
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Drive upload failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def cleanup_local_files(paths: List[str]) -> None:
        """Delete local files — call ONLY after the upload is confirmed."""
        for p in paths:
            if not p:
                continue
            try:
                os.remove(p)
                print(f"[drive] deleted local temp file: {p}", flush=True)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[drive] could not delete {p}: {e}", flush=True)


async def upload_shorts_and_cleanup(
    shorts: List[Dict],
    service_account_file: str,
    folder_id: str,
    source_video_path: Optional[str] = None,
    delete_source_video: bool = True,
) -> List[Dict]:
    """Upload every clip in generate_shorts()['shorts'] to Drive and delete
    the local .mp4 files once each upload is confirmed.

    - A clip is only deleted if its own upload succeeded; failed clips stay
      on disk so a retry doesn't require re-rendering from scratch.
    - If delete_source_video=True, the full source download (source_xxx.mp4)
      is deleted once every clip has been attempted — it's the heaviest
      file on disk and is no longer needed after clips are cut from it.
    """
    uploader = DriveUploader(service_account_file, folder_id)
    updated_shorts: List[Dict] = []

    for short in shorts:
        clip_path = short.get("clip_url")
        if not clip_path or not os.path.exists(clip_path):
            updated_shorts.append(short)  # already had no clip (render failed upstream)
            continue
        try:
            drive_result = await uploader.upload_async(clip_path)
            uploader.cleanup_local_files([clip_path])
            updated_shorts.append({
                **short,
                "drive_file_id": drive_result["id"],
                "drive_url": drive_result.get("webViewLink"),
                "clip_url": None,  # no longer lives on local disk
            })
        except Exception as e:
            print(f"[drive] keeping {clip_path} locally, upload failed: {e}", flush=True)
            updated_shorts.append({**short, "drive_error": str(e)})

    if delete_source_video and source_video_path:
        uploader.cleanup_local_files([source_video_path])

    return updated_shorts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m shorts_generator.drive_uploader <file.mp4>")
        sys.exit(1)

    from .config import GDRIVE_FOLDER_ID, GDRIVE_SERVICE_ACCOUNT_FILE

    async def _test():
        uploader = DriveUploader(GDRIVE_SERVICE_ACCOUNT_FILE, GDRIVE_FOLDER_ID)
        result = await uploader.upload_async(sys.argv[1])
        print(result)

    asyncio.run(_test())
