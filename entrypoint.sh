#!/bin/bash -e

set -Eeuo pipefail

bump_type=$1
ref_branch=$2
github_token=$3
force_push=$4
tag_commit=$5

runner () {
    echo "🟡 starting $@"
    $@ && echo "🟢 $@ passed" || (echo "🔴 $@ failed" && exit 1)
}

if [[ -n "$GITHUB_WORKSPACE" ]]; then
    git config --global --add safe.directory $GITHUB_WORKSPACE
fi

git config --global user.name $GITHUB_ACTOR
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

if [ -n "$(git status --porcelain)" ]; then
    # Working directory clean
    # Uncommitted changes
    echo "🔴 There are uncommitted changes, exiting"
    exit 1
fi
git reset --hard

# Bump version
VERSION=`pyverto version`
case $bump_type in
    'release')
        pyverto release --commit;;
    'major')
        pyverto major --commit;;
    'minor')
        pyverto minor --commit;;
    'micro' | 'patch' | 'fix')
        pyverto micro --commit;;
    'alpha')
        pyverto alpha --commit;;
    'beta')
        pyverto beta --commit;;
    'c' | 'rc' | 'pre' | 'preview')
        pyverto pre --commit;;
    'r' | 'rev' | 'post')
        pyverto rev --commit;;
    'dev')
        pyverto dev --commit;;
    *)
        echo "🔵 Skipped Version Bump";;
esac
NEW_VERSION=`pyverto version`


if [ "$VERSION" != "$NEW_VERSION" ]; then
    echo "🟢 Success: bump version: $VERSION → $NEW_VERSION"

    # Push the new version
    git remote -v

    # Authenticate
    if [ -n "$github_token" ]; then
	git remote set-url origin "https://${GITHUB_ACTOR}:${github_token}@github.com/${GITHUB_REPOSITORY}.git"
    fi

    # Force push?
    if [ "$force_push" = true ]; then
        PUSH_FLAGS="--force-with-lease"
    else
        PUSH_FLAGS=""
    fi

    # Push
    if [ -n "$ref_branch" ]; then
        git push origin HEAD:$ref_branch $PUSH_FLAGS
    elif [ -n "$GITHUB_HEAD_REF" ]; then
        git push origin HEAD:$GITHUB_HEAD_REF $PUSH_FLAGS
    else
        git push $PUSH_FLAGS
    fi
    if [ "$tag_commit" = true ]; then
        git push --tags
    fi
    echo "🟢 Success version push"
fi

exit 0
