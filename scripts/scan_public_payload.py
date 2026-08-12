"""Scan the candidate public payload for secrets and private identifiers."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = ("pdocs/", "tmp/", ".git/")


def _git_lines(*arguments: str) -> list[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def candidate_files() -> list[str]:
    return sorted(
        path
        for path in _git_lines("ls-files", "--cached", "--others", "--exclude-standard")
        if not path.startswith(EXCLUDED_PREFIXES)
    )


def scan() -> dict:
    credential_patterns = {
        "github_token": re.compile("gh" + r"p_[A-Za-z0-9]{20,}"),
        "github_fine_grained_token": re.compile(
            "github" + r"_pat_[A-Za-z0-9_]{20,}"
        ),
        "openai_key": re.compile("sk" + r"-[A-Za-z0-9]{20,}"),
        "aws_access_key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
        "private_key": re.compile(
            "-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    }
    restricted_patterns = {
        "internal_company_name": re.compile("infi" + "neon", re.IGNORECASE),
        "internal_hostname": re.compile("in" + r"tra\.", re.IGNORECASE),
            "restricted_source_path": re.compile(
            "internal" + r"[-_]reference", re.IGNORECASE
        ),
    }
    local_path_patterns = {
        "windows_user_path": re.compile(
            r"[A-Za-z]:\\Users\\[^\\\s]+\\", re.IGNORECASE
        ),
            "windows_profile_data": re.compile("App" + "Data", re.IGNORECASE),
        "vscode_workspace_storage": re.compile(
            "workspace" + "Storage", re.IGNORECASE
        ),
    }

    findings = []
    binary_files = []
    scanned_files = []
    for relative_path in candidate_files():
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            continue
        content = path.read_bytes()
        if b"\x00" in content[:8192]:
            binary_files.append(relative_path)
            continue
        text = content.decode("utf-8", errors="replace")
        scanned_files.append(relative_path)
        for category, patterns in (
            ("credential", credential_patterns),
            ("restricted_internal", restricted_patterns),
            ("local_path", local_path_patterns),
        ):
            for name, pattern in patterns.items():
                if pattern.search(text):
                    findings.append(
                        {
                            "category": category,
                            "pattern": name,
                            "file": relative_path,
                        }
                    )

    tracked_private = _git_lines("ls-files", "pdocs")
    if tracked_private:
        findings.extend(
            {
                "category": "tracked_private_material",
                "pattern": "pdocs_tracked",
                "file": path,
            }
            for path in tracked_private
        )

    return {
        "schema_version": 1,
        "scope": "candidate public Project 05 files; ignored pdocs and tmp excluded",
        "status": "passed" if not findings else "failed",
        "candidate_file_count": len(candidate_files()),
        "text_file_count": len(scanned_files),
        "binary_file_count": len(binary_files),
        "binary_files": binary_files,
        "tracked_pdocs_count": len(tracked_private),
        "findings": findings,
        "provenance_note": (
            "No internal reference repository was accessed for this scan; "
            "the payload is checked for internal identifiers and local paths only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "evidence" / "public_payload_scan.json",
    )
    args = parser.parse_args()
    report = scan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())