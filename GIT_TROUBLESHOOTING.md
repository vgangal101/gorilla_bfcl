# Git Troubleshooting

Quick reference for common git issues on this repo (`vgangal101/gorilla_bfcl`).

---

## Push fails with 403 or "Authentication failed"

This happens when a fine-grained PAT is embedded in the remote URL or the
cached credentials expire. Fine-grained PATs are scoped to repos you own and
won't work for a collaborator repo you don't own.

**Fix (try this first):**

```bash
# 1. Strip any embedded token — revert to clean HTTPS URL
git remote set-url origin https://github.com/vgangal101/gorilla_bfcl.git

# 2. Push — Windows Credential Manager has the working classic PAT cached
git push -u origin <your-branch>
```

This works because the classic PAT stored in Windows Credential Manager covers
all repos you have collaborator access to.

**If the cached credentials have expired or been revoked:**

Create a new **classic PAT** (not fine-grained) at:
`github.com → Settings → Developer settings → Personal access tokens → Tokens (classic)`
with `repo` scope checked.

Use it once to push, then strip it back out:

```bash
# Embed token temporarily
git remote set-url origin https://mvsg2:<TOKEN>@github.com/vgangal101/gorilla_bfcl.git
git push -u origin <your-branch>

# Strip it back immediately after — never leave tokens in the remote URL
git remote set-url origin https://github.com/vgangal101/gorilla_bfcl.git
```

**Never leave a token embedded in the remote URL** — it shows up in plaintext
via `git remote -v` and in shell history.

---

## Switching branches with uncommitted changes

```
error: Your local changes to the following files would be overwritten by checkout
```

Options:

```bash
# Option A — stash changes, switch, restore later
git stash
git switch <branch>
git stash pop   # when you want them back

# Option B — discard changes (irreversible)
git checkout -- <file>
git switch <branch>

# Option C — commit first
git add <file>
git commit -m "wip"
git switch <branch>
```
