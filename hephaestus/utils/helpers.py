import subprocess


def get_git_commit_hash():
    """Get the current git commit hash.

    Returns:
        str: The current git commit hash, or 'unknown' if an error occurs.
    """
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode()
        )
        return commit_hash
    except Exception:
        return "unknown"
