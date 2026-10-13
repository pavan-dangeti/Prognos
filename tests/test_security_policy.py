"""Security policy consistency.

The dependency audit runs with `--ignore-vuln` flags for advisories we have reviewed
and accepted. Those flags are the whole exception, so they have to be checkable:
an exception that cannot be seen is an exception nobody reviews.

These tests read the documented policy and the CI workflow and fail if they drift.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

POLICY = Path("pip-audit.toml")
WORKFLOW = Path(".github/workflows/ci.yml")
DOCKERFILE = Path("docker/Dockerfile")


@pytest.fixture(scope="module")
def documented() -> list[str]:
    """Advisory IDs the project has accepted."""
    if not POLICY.exists():
        pytest.skip(f"{POLICY} not present")
    return list(tomllib.loads(POLICY.read_text())["known_vulnerabilities"])


def _workflow_ignore_flags(text: str) -> list[str]:
    """Advisory IDs passed to `--ignore-vuln` in the CI workflow."""
    return re.findall(r"--ignore-vuln[=\s]+([A-Z]+-\d{4}-\d+)", text)


class TestPolicyIsHonest:
    def test_policy_is_not_empty_and_not_everything(self, documented: list[str]) -> None:
        """An exception list that covers nothing, or everything, is not a policy."""
        assert documented, "no accepted advisories documented"
        assert len(documented) <= 10, "the exception list has grown too large to review"

    def test_every_documented_advisory_is_actually_ignored_in_ci(
        self, documented: list[str]
    ) -> None:
        """The documented reason must correspond to a real CI exception.

        Without this, `pip-audit.toml` could describe a policy that CI does not
        implement, and the audit would fail on a real finding with no one knowing
        why.
        """
        if not WORKFLOW.exists():
            pytest.skip(f"{WORKFLOW} not present")
        flags = _workflow_ignore_flags(WORKFLOW.read_text())
        for advisory in documented:
            assert advisory in flags, f"{advisory} documented but not ignored in CI"

    def test_ci_does_not_ignore_anything_undocumented(self) -> None:
        """No silent exceptions.

        An ID ignored in CI with no written reason is exactly the failure mode this
        file exists to prevent.
        """
        if not WORKFLOW.exists() or not POLICY.exists():
            pytest.skip("workflow or policy not present")
        documented = set(tomllib.loads(POLICY.read_text())["known_vulnerabilities"])
        flags = set(_workflow_ignore_flags(WORKFLOW.read_text()))
        undocumented = flags - documented
        assert not undocumented, (
            f"CI ignores advisories with no documented reason: {sorted(undocumented)}"
        )

    def test_every_advisory_carries_a_reason_and_a_limit(self, documented: list[str]) -> None:
        """An accepted advisory must state why and where the risk stops applying."""
        text = POLICY.read_text()
        for advisory in documented:
            assert advisory in text
        assert "Reason:" in text, "the policy must state a reason"
        assert "Limit:" in text, "the policy must state the limit of that reason"


class TestImageHygiene:
    def test_base_image_is_pinned_by_digest(self) -> None:
        """A floating tag lets the base image change under a passing build."""
        text = DOCKERFILE.read_text()
        from_lines = [line for line in text.splitlines() if line.startswith("FROM ")]
        assert from_lines, "no FROM instruction"
        for line in from_lines:
            assert "@sha256:" in line, f"base image not pinned by digest: {line}"

    def test_container_does_not_run_as_root(self) -> None:
        """A data pipeline reading a 15 GB archive has no business running as root."""
        text = DOCKERFILE.read_text()
        assert "USER prognos" in text
        assert "USER root" not in text.replace("USER prognos", "")

    def test_no_secrets_baked_into_the_image(self) -> None:
        """Keys come from the environment, never from a build argument."""
        text = DOCKERFILE.read_text().upper()
        for marker in ("ARG API_KEY", "ARG TOKEN", "ARG SECRET", "ENV NVIDIA_API_KEY"):
            assert marker not in text, f"{marker} found in the Dockerfile"
