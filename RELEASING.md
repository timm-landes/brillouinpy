# Releasing BrillouinPy

Releases are cut from `develop` and land on `main` as a single tagged merge
commit. `main` is release-only; all development happens on `develop` (see
[`CONTRIBUTING.md`](CONTRIBUTING.md#branching-and-pull-requests)).

Versioning is [SemVer](https://semver.org/): bump the **minor** version for new
features / backwards-compatible changes, the **patch** version for bug fixes
only, the **major** version for breaking changes.

## 1. Prepare on `develop`

```bash
git checkout develop
git pull
ruff check . && pytest tests/ -v          # must be green
```

- Bump `version=` in [`setup.py`](setup.py).
- In [`CHANGELOG.md`](CHANGELOG.md): rename the `## [Unreleased]` heading to
  `## [X.Y.Z] - YYYY-MM-DD`, add a one-line summary under it, and consolidate the
  `### Added` / `### Changed` / `### Fixed` subsections. Leave a fresh, empty
  `## [Unreleased]` above it.
- If `docs/` or docstrings changed, build the docs once locally to check for
  Sphinx warnings (`sphinx-build -b html docs docs/_build/html`).

```bash
git commit -am "chore: release X.Y.Z"
git push github develop
git push origin develop
```

## 2. Merge to `main` and tag

`main` is protected, so the merge goes through a pull request:

1. Open a PR `develop -> main` on GitHub, titled `chore: release X.Y.Z`.
2. Wait for CI (`lint`, `test`) to pass, then **Merge** (a merge commit, not
   squash - keep `--no-ff` history).
3. Tag the merge commit and push the tag to both remotes:

   ```bash
   git checkout main
   git pull github main
   git tag vX.Y.Z
   git push github vX.Y.Z
   git push origin main --tags
   git checkout develop
   git merge main            # bring the merge commit + tag back onto develop
   git push github develop && git push origin develop
   ```

> If branch protection is in guardrail mode (admins may bypass), you can instead
> merge locally: `git checkout main && git merge --no-ff develop -m "chore: release X.Y.Z"`
> then tag and push as above. The PR route is preferred - it runs CI against the
> exact merge result.

## 3. Publish

- **GitHub Release:** github.com/timm-landes/brillouinpy → Releases → *Draft a
  new release* → choose tag `vX.Y.Z`, title `vX.Y.Z`, paste the `[X.Y.Z]`
  section from `CHANGELOG.md` as the description → *Publish release*.
- **Docs:** rebuild automatically from the push to `main`
  ([`.github/workflows/docs.yml`](.github/workflows/docs.yml)) → verify the
  `docs` run is green at
  [Actions](https://github.com/timm-landes/brillouinpy/actions). The
  `deploy-pages` step is occasionally slow (10-20 min); that's normal.
- **PyPI** (optional):

  ```bash
  rm -rf dist build *.egg-info
  python -m build
  twine check dist/*
  twine upload dist/*
  ```

  (`twine` reads the token from `~/.pypirc`.) Then sanity-check:
  `pip install --no-cache-dir BrillouinPy==X.Y.Z`.
- **GitLab wiki:** if `docs/tutorial.md` changed, update the corresponding wiki
  page at
  `gitlab.uni-hannover.de/phytophotonics/brillouinpy/-/wikis` to match.

## 4. Post-release

Nothing extra - keep working on `develop`. New CHANGELOG entries go under the
fresh `## [Unreleased]` heading.
