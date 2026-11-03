# Preparing releases

Run `tools/release.sh` from any directory and choose the stable or prerelease workflow.

## Testing the next release

Choose the PyPI prerelease workflow, enter the target stable version, then select
alpha, beta, or release candidate and its sequence number. For example, the first
release candidate for `0.6.0` is `0.6.0rc1`; subsequent candidates use `rc2`,
`rc3`, and so on. Each version can be uploaded to PyPI only once.

The prerelease path can commit and tag the version as `v0.6.0rc1`, then build,
check, and upload the package to the normal GraxPy PyPI project. It does not
create a GitHub release or pull request. Testers opt in with:

```bash
pip install --pre --upgrade graxpy
```

Without `--pre`, pip continues to choose the newest stable version. After testing,
prepare and publish the stable version through the stable workflow, using the
same target version without the prerelease suffix (for example, `0.6.0`).
