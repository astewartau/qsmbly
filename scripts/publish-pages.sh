#!/bin/bash
# Publish a built site into one subtree of the GitHub Pages branch.
#
# The Pages branch holds the deployed site as independent subtrees: the released site at the root
# and a staging build under staging/. Each deploy replaces its own subtree and leaves the others
# exactly as they were, so a push to the staging branch cannot disturb what is in production and a
# release cannot roll back staging.
#
# The branch carries a single commit, rebuilt and force-pushed on every deploy. The site is ~25 MB
# of wasm per subtree, and an append-only history would grow by that much every time. Git is
# content-addressed, so a rebuild only stores files that actually changed: a deploy that touches
# only JavaScript adds kilobytes, not megabytes.
#
# Usage:
#   publish-pages.sh --source <dir> --remote <url> [options]
#
#   --source DIR        built site to publish (required)
#   --remote URL        push target, e.g. https://x-access-token:$TOKEN@github.com/owner/repo.git
#   --subtree PATH      subtree to replace; empty or omitted means the root of the site
#   --keep NAME         when publishing the root, a top-level entry to preserve (repeatable)
#   --prune-unknown     allow a root deploy to delete top-level entries it neither builds nor keeps
#   --branch NAME       Pages branch (default: gh-pages)
#   --domain HOST       written to CNAME at the branch root; Pages reads the custom domain there
#   --noindex PATH      written to robots.txt at the branch root as a Disallow rule (repeatable)
#   --message TEXT      commit message
#   --dry-run           do everything except the push, and print the resulting tree

set -euo pipefail

SOURCE=""
REMOTE=""
SUBTREE=""
BRANCH="gh-pages"
DOMAIN=""
MESSAGE="Deploy"
DRY_RUN=0
PRUNE_UNKNOWN=0
KEEP=()
NOINDEX=()

while [ $# -gt 0 ]; do
    case "$1" in
        --source)   SOURCE="$2"; shift 2 ;;
        --remote)   REMOTE="$2"; shift 2 ;;
        --subtree)  SUBTREE="${2#/}"; SUBTREE="${SUBTREE%/}"; shift 2 ;;
        --keep)     KEEP+=("$2"); shift 2 ;;
        --branch)   BRANCH="$2"; shift 2 ;;
        --domain)   DOMAIN="$2"; shift 2 ;;
        --noindex)  NOINDEX+=("${2#/}"); shift 2 ;;
        --message)  MESSAGE="$2"; shift 2 ;;
        --prune-unknown) PRUNE_UNKNOWN=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *) echo "publish-pages.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

[ -n "$SOURCE" ] || { echo "publish-pages.sh: --source is required" >&2; exit 2; }
[ -d "$SOURCE" ] || { echo "publish-pages.sh: --source '$SOURCE' is not a directory" >&2; exit 2; }
[ -n "$REMOTE" ] || { echo "publish-pages.sh: --remote is required" >&2; exit 2; }

SOURCE="$(cd "$SOURCE" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
PAGES="${WORK}/pages"

# A shallow clone is all that is needed: the previous commit is never a parent of the next one.
# The branch may not exist yet on a first deploy, which is not an error.
if git clone --quiet --depth 1 --branch "$BRANCH" "$REMOTE" "$PAGES" 2>/dev/null; then
    echo "Updating ${BRANCH}: $(git -C "$PAGES" log -1 --format='%h %s')"
else
    echo "Creating ${BRANCH} (no existing branch on the remote)"
    git init --quiet "$PAGES"
fi

cd "$PAGES"

if [ -n "$SUBTREE" ]; then
    rm -rf "./${SUBTREE:?}"
    mkdir -p "./${SUBTREE}"
    cp -a "${SOURCE}/." "./${SUBTREE}/"
    echo "Replaced /${SUBTREE}/"
else
    # Anything at the top level that this build does not provide and that was not named with
    # --keep is about to be deleted. That is exactly how publishing a release would silently take
    # the staging site down, so make it a deliberate choice rather than a side effect of a forgotten
    # flag. Nothing is at risk on a first deploy, where the branch is empty.
    ORPHANS=()
    while IFS= read -r entry; do
        name="$(basename "$entry")"
        case "$name" in
            .git) continue ;;
            # Branch-level files this script writes itself, below.
            CNAME|robots.txt) continue ;;
        esac
        for kept in "${KEEP[@]+"${KEEP[@]}"}"; do
            [ "$name" = "$kept" ] && continue 2
        done
        [ -e "${SOURCE}/${name}" ] && continue
        ORPHANS+=("$name")
    done < <(find . -mindepth 1 -maxdepth 1)

    if [ ${#ORPHANS[@]} -gt 0 ] && [ "$PRUNE_UNKNOWN" != "1" ]; then
        echo "publish-pages.sh: refusing to publish the site root." >&2
        echo "  These are deployed on ${BRANCH} but are neither in --source nor in --keep:" >&2
        printf '    %s\n' "${ORPHANS[@]}" >&2
        echo "  Add --keep <name> to preserve them, or --prune-unknown to delete them." >&2
        exit 1
    fi

    # Clear the root but keep .git and every subtree this deploy does not own, so the released
    # site and the staging build stay independent of each other.
    PRUNE=(find . -mindepth 1 -maxdepth 1 ! -name .git)
    for name in "${KEEP[@]+"${KEEP[@]}"}"; do
        PRUNE+=(! -name "$name")
    done
    "${PRUNE[@]}" -exec rm -rf {} +
    cp -a "${SOURCE}/." ./
    echo "Replaced the site root, keeping: ${KEEP[*]:-nothing}"
fi

# Branch-level files, rewritten on every deploy so they survive a root replacement and exist even
# if the first deploy to a fresh branch is a staging one.
if [ -n "$DOMAIN" ]; then
    echo "$DOMAIN" > CNAME
fi
if [ ${#NOINDEX[@]} -gt 0 ]; then
    # robots.txt is only honoured at the site root, so it cannot live inside the staging subtree.
    {
        echo "User-agent: *"
        for path in "${NOINDEX[@]}"; do
            echo "Disallow: /${path}/"
        done
    } > robots.txt
fi

# --orphan rather than a normal commit: it keeps the index and working tree but drops the parent,
# so the branch stays one commit deep. `git switch --orphan` would clear the tree instead.
git checkout --quiet --orphan deploy
git add -A
git -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit --quiet -m "$MESSAGE"

if [ "$DRY_RUN" = "1" ]; then
    echo "--- dry run, not pushing. Resulting tree: ---"
    git ls-tree -r --name-only HEAD | sed 's/^/  /'
    exit 0
fi

git push --quiet --force "$REMOTE" "deploy:${BRANCH}"
echo "Pushed $(git rev-parse --short HEAD) to ${BRANCH}"
