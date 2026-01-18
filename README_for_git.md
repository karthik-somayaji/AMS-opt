# Git notes (for this repo): copy a folder from another branch

This README is a practical cheat-sheet for a common task:

> **Bring one directory from another branch into your current branch** (without merging everything else).

It uses examples from this repo:
- Import `netlists/diff_amps/` from Lucas’s branch
- Import `netlists/comparators/` from Heyang’s branch

---

## Mental model (one paragraph)

- A **branch name** (e.g., `karthik-feat-2-dataset_expansion`) is just a pointer to a commit.
- A **remote-tracking branch** (e.g., `origin/heyang-feat-1-dataset-expansion-comps`) is your local record of where a remote branch was the last time you `git fetch`’d.
- `git restore --source <branch> -- <path>` copies *exactly that path* from `<branch>` into your working tree on your current branch. It does **not** merge the branches.

---

## 0) Always start here: check where you are + get latest remote refs

```bash
cd /home/karthik/sim_clean/AMS-opt

git status -sb

git fetch --all --prune
```

What you want:
- `git status -sb` should be clean (or at least you know what’s modified).
- `git fetch --all --prune` updates your knowledge of remote branches under `origin/...`.

If you have uncommitted work you don’t want mixed in:

```bash
git stash -u
```

---

## 1) List branches (local vs remote)

Local branches (on your machine):

```bash
git branch
```

Remote-tracking branches (your local view of the remote):

```bash
git branch -r
```

All branches:

```bash
git branch -a
```

Ask the remote directly (optional):

```bash
git ls-remote --heads origin
```

---

## 2) Safety step (recommended): create a backup branch

Before doing a “copy folder from another branch”, make a quick backup branch.

```bash
# Make sure you're on the target branch first

git switch karthik-feat-2-dataset_expansion

# Create a backup branch from the current commit

git switch -c backup/karthik-feat-2-dataset_expansion-$(date +%Y%m%d)

# Switch back to the working branch

git switch karthik-feat-2-dataset_expansion
```

Why this helps: you can always return to the pre-change state.

---

## 3) Preview what would change in a directory (high-signal learning step)

Before copying anything, compare just that directory:

```bash
# Summary of what differs in that subtree

git diff --stat HEAD..origin/<some-branch> -- <path/to/dir>/

# List the files that differ

git diff --name-only HEAD..origin/<some-branch> -- <path/to/dir>/
```

---

## 4) Copy a folder from another branch (the main trick)

### Option A (recommended): `git restore --source`

```bash
git restore --source origin/<some-branch> -- <path/to/dir>/
```

### Option B (older equivalent): `git checkout <branch> -- path`

```bash
git checkout origin/<some-branch> -- <path/to/dir>/
```

Both options:
- replace/create files under that directory to match the source branch
- do **not** merge other changes

---

## 5) Verify you only changed what you intended

```bash
git status

git diff --stat
```

If anything outside your target folder changed, stop and investigate (common cause: you weren’t clean, or you forgot you had local edits).

---

## 6) Commit + push

```bash
git add <path/to/dir>/

git commit -m "Import <path/to/dir> from <some-branch>"

git push
```

---

## Example 1: import `netlists/diff_amps/` from Lucas

### What happened earlier
You got:
- `fatal: could not resolve lucas-feat2-dataset_expansion`

Reason: that branch name didn’t exist. The actual remote branch was:
- `origin/lucas-feat-2-dataset_expansion`

### Do it (preview + import)

```bash
cd /home/karthik/sim_clean/AMS-opt

# target branch

git switch karthik-feat-2-dataset_expansion

git fetch --all --prune

# confirm branch exists

git branch -r | grep lucas

# preview

git diff --stat HEAD..origin/lucas-feat-2-dataset_expansion -- netlists/diff_amps/

# import only that folder

git restore --source origin/lucas-feat-2-dataset_expansion -- netlists/diff_amps/

# verify

git diff --stat

# commit

git add netlists/diff_amps/

git commit -m "Import netlists/diff_amps from lucas dataset expansion branch"

git push
```

---

## Example 2: import `netlists/comparators/` from Heyang

Source branch:
- `origin/heyang-feat-1-dataset-expansion-comps`

```bash
cd /home/karthik/sim_clean/AMS-opt

# target branch

git switch karthik-feat-2-dataset_expansion

git fetch --all --prune

# confirm branch exists

git branch -r | grep heyang

# preview

git diff --stat HEAD..origin/heyang-feat-1-dataset-expansion-comps -- netlists/comparators/

# import only that folder

git restore --source origin/heyang-feat-1-dataset-expansion-comps -- netlists/comparators/

# verify

git diff --stat

# commit

git add netlists/comparators/

git commit -m "Import netlists/comparators from heyang dataset expansion branch"

git push
```

---

## Undo / recovery (common scenarios)

### Undo the copied folder before committing

```bash
# Discard changes in that folder in both index + working tree

git restore --staged --worktree -- <path/to/dir>/
```

### Undo after committing (safe)

```bash
# Creates a new commit that reverses the last commit

git revert HEAD
```

### Hard reset to your backup branch (dangerous if you have local work)

```bash
git reset --hard backup/karthik-feat-2-dataset_expansion-YYYYMMDD
```

---

## Gotchas (worth memorizing)

- **Branch names must match exactly**: use `git branch -a | grep <name>` to confirm.
- **Remote branches require `git fetch`**: otherwise `origin/<branch>` might not exist locally.
- **This overwrites files in the target directory** if both branches have the same paths with different content.
  - If you want “only add new files, don’t overwrite existing ones”, use a different approach (ask!).
- **Stay clean**: uncommitted changes make it harder to see what your operation did.
