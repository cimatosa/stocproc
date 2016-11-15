#!/bin/bash

pip install sphinx
pip install sphinx_rtd_theme
cd doc
openssl aes-256-cbc -K $encrypted_b45fa9d06fcb_key -iv $encrypted_b45fa9d06fcb_iv -in deploy_key.enc -out deploy_key -d
chmod 600 deploy_key
eval `ssh-agent -s`
ssh-add deploy_key
mkdir build
cd build
git clone https://github.com/cimatosa/stocproc.git --branch gh-pages html
cd ..
make html
cd build/html
git config user.name "travis gh-pages"
git config user.email "$COMMIT_AUTHOR_EMAIL"
git add .
SHA=`git rev-parse --verify HEAD`
REPO=`git config remote.origin.url`
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
git commit -m "Deploy to GitHub Pages $SHA"
git push $SSH_REPO gh-pages
