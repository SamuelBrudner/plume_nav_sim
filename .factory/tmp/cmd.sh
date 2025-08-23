#!/usr/bin/env bash
set -e
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "droid/utils-tests" ]; then
  git checkout -b droid/utils-tests
fi
git rm -f pytest_benchmark.py || true
git add .factory
git commit -F .factory/commit-message.txt || true
