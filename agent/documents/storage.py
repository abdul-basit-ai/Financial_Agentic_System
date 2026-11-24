from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests

from agent.server.auth import valid_service_url


class FileStorage:
    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SECRET_KEY", "") or os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY", ""
        )
        self.bucket = os.getenv("SUPABASE_REPORT_BUCKET", "financial-reports")
        self.root = Path(os.getenv("FINAGENT_FILES_PATH", "data/runtime/reports"))
        if self.url and (not self.key or not valid_service_url(self.url)):
            raise RuntimeError(
                "Configure a secure Supabase endpoint and server storage credential."
            )

    def _headers(self) -> dict[str, str]:
        return {"apikey": self.key, "Authorization": f"Bearer {self.key}"}

    def _validate(self, key: str) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}/[a-zA-Z0-9_-]{1,128}\.pdf", key):
            raise ValueError("Invalid report object key.")

    def _url(self, key: str) -> str:
        self._validate(key)
        return f"{self.url}/storage/v1/object/{quote(self.bucket,safe='')}/{quote(key,safe='/')}"

    @staticmethod
    def _check(response: requests.Response) -> None:
        if not 200 <= response.status_code < 300:
            raise RuntimeError("Private report storage is unavailable.")

    def put(self, owner_id: str, document_id: str, content: bytes) -> str:
        key = f"{owner_id}/{document_id}.pdf"
        self._validate(key)
        if self.url:
            response = requests.post(
                self._url(key),
                headers={
                    **self._headers(),
                    "Content-Type": "application/pdf",
                    "x-upsert": "true",
                },
                data=content,
                timeout=(5, 30),
                allow_redirects=False,
            )
            self._check(response)
        else:
            target = self._local_path(key)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            staged = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
            path = Path(staged.name)
            try:
                with staged:
                    staged.write(content)
                    staged.flush()
                    os.fsync(staged.fileno())
                os.replace(path, target)
            finally:
                path.unlink(missing_ok=True)
        return key

    def read(self, key: str) -> bytes:
        if self.url:
            response = requests.get(
                self._url(key),
                headers=self._headers(),
                timeout=(5, 30),
                allow_redirects=False,
            )
            self._check(response)
            return response.content
        return self._local_path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._validate(key)
        if self.url:
            response = requests.delete(
                f"{self.url}/storage/v1/object/{quote(self.bucket,safe='')}",
                headers=self._headers(),
                json={"prefixes": [key]},
                timeout=(5, 15),
                allow_redirects=False,
            )
            if response.status_code != 404:
                self._check(response)
        else:
            self._local_path(key).unlink(missing_ok=True)

    def _local_path(self, key: str) -> Path:
        self._validate(key)
        root = self.root.resolve()
        target = (root / key).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Invalid report path.")
        return target
