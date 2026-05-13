#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYPROJECT_FILE="pyproject.toml"

if [[ ! -f "${PYPROJECT_FILE}" ]]; then
    echo "Error: pyproject.toml not found."
    exit 1
fi

CURRENT_VERSION="$(grep '^version = ' "${PYPROJECT_FILE}" | head -n1 | sed 's/version = "\(.*\)"/\1/')"

echo "============================================================"
echo "Current version: ${CURRENT_VERSION}"
echo "============================================================"

IFS='.' read -r MAJOR MINOR PATCH <<< "${CURRENT_VERSION}"

echo
echo "Select version bump:"
echo "1) patch (${MAJOR}.${MINOR}.$((PATCH + 1)))"
echo "2) minor (${MAJOR}.$((MINOR + 1)).0)"
echo "3) major ($((MAJOR + 1)).0.0)"
echo "4) custom"

read -rp "Choice [1-4]: " VERSION_CHOICE

case "${VERSION_CHOICE}" in
    1)
        NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
        ;;
    2)
        NEW_VERSION="${MAJOR}.$((MINOR + 1)).0"
        ;;
    3)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    4)
        read -rp "Enter version: " NEW_VERSION
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

echo
echo "New version: ${NEW_VERSION}"

read -rp "Proceed? [y/N]: " CONFIRM

if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo
echo "============================================================"
echo "Updating version"
echo "============================================================"

sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" "${PYPROJECT_FILE}"

echo
echo "============================================================"
echo "Git status"
echo "============================================================"

git status --short

read -rp "Create git commit and tag? [y/N]: " GIT_CONFIRM

if [[ "${GIT_CONFIRM}" == "y" || "${GIT_CONFIRM}" == "Y" ]]; then
    git add "${PYPROJECT_FILE}"

    git commit -m "Release v${NEW_VERSION}"

    git tag "v${NEW_VERSION}"

    echo
    echo "Git commit and tag created."
fi

echo
echo "============================================================"
echo "Checking GitHub CLI"
echo "============================================================"

if ! command -v gh >/dev/null 2>&1; then
    echo
    echo "GitHub CLI (gh) is not installed."
    echo
    echo "Install instructions for Debian/Ubuntu:"
    echo
    echo "------------------------------------------------------------"
    echo "type -p curl >/dev/null || sudo apt install curl -y"
    echo "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \\"
    echo "  sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
    echo "sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg"
    echo
    echo "echo \"deb [arch=\$(dpkg --print-architecture) \\"
    echo "  signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \\"
    echo "  https://cli.github.com/packages stable main\" | \\"
    echo "  sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null"
    echo
    echo "sudo apt update"
    echo "sudo apt install gh -y"
    echo "------------------------------------------------------------"
    echo
    echo "Then authenticate with:"
    echo
    echo "  gh auth login"
    echo
else
    read -rp "Create GitHub release? [y/N]: " GH_CONFIRM

    if [[ "${GH_CONFIRM}" == "y" || "${GH_CONFIRM}" == "Y" ]]; then
        gh release create \
            "v${NEW_VERSION}" \
            --title "v${NEW_VERSION}" \
            --generate-notes
    fi
fi

echo
read -rp "Upload package to PyPI? [y/N]: " PYPI_CONFIRM

if [[ "${PYPI_CONFIRM}" == "y" || "${PYPI_CONFIRM}" == "Y" ]]; then
    "${SCRIPT_DIR}/publish_pypi.sh"
fi

echo
echo "============================================================"
echo "Release workflow completed"
echo "============================================================"