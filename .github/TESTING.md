# Pull request checks

## Automatic, on every pull request

Every pull request (from anyone) runs, with no command needed:

- **Tests / Lint & Format** (`test_code.yml`): `ruff check` and `ruff format --check`.
- **Docs / build** (`docs.yml`): the strict MkDocs build and the asset check. Deploying
  the site stays `main`-only.

The same **Tests** run also lists **Tests / Model tests**, skipped until it is requested.

## Model, tokenizer and processor tests, on request

The heavy suite (`model_tests.yml`: 9 model shards across Torch, TensorFlow and JAX,
1 tokenizer job, 3 image-processor jobs, 3 processor jobs, and one summary job) runs on
a pull request only when **IMvision12** posts a new comment containing exactly:

```text
/run-tests
```

Other users' comments, edited comments and other text are ignored.

The tests run **inside the pull request's own Tests run**, on the pull request's code,
and every job shows up in the PR's checks as **Tests / Model tests / ...**. Nothing runs
on another branch or in a separate run.

How it works:

1. `run_tests_command.yml` (triggered by the comment, never checking out PR code)
   validates the request and finds the PR's **Tests** run for its current head commit.
2. It marks that commit with a `run-tests` commit status ("Requested by @IMvision12").
3. It re-runs that **Tests** run. The run's **Check for /run-tests** job reads the
   status, finds the request, and lets the **Model tests** job (which calls
   `model_tests.yml`) run.

The comment gets a 🚀 reaction and a reply linking the run.

- A later push is not tested automatically: the request is tied to one commit, so
  comment `/run-tests` again. Commenting it while a run is still going stops that run
  and starts it again with the tests.
- **Pull requests from forks** work the same way. If the fork's run is waiting for
  approval, the reply asks you to approve it first.
- The PR branch must contain the current `test_code.yml` and `model_tests.yml` (re-runs
  use the PR's own workflow files). `run_tests_command.yml` only takes effect once it is
  on `main` (comment events always use the default branch's workflow files).

**Tests / Model tests / Requested model, tokenizer & processor tests** passes only when
every test job passes. To make the suite block merging, require that check in the branch
protection rules or ruleset; the PR then cannot merge until `/run-tests` passes.

## Releases and manual runs

Release events run lint and the whole suite in `model_tests.yml` automatically, with
pip caching and Codecov upload (both disabled for pull request runs). The suite can also
be started by hand on any branch from the Actions tab (**Model tests → Run workflow**).

## Security

The comment workflow has `actions: write` (to re-run the PR's run), `statuses: write`
(to mark the request) and comment permissions, and never checks out or executes PR code.
The test jobs run as an ordinary `pull_request` run: read-only repository access (fork
runs get a read-only token), no secrets, no shared caches, and checkout credentials are
not persisted. Only the comment workflow can create the `run-tests` status.
