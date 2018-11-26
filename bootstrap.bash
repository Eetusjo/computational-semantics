#!/bin/bash

cd /home/jovyan/

rm -r work
rm bootstrap.bash

export GIT_COMMITTER_NAME=anonymous
export GIT_COMMITTER_EMAIL=anon@localhost
git clone https://github.com/Eetusjo/computational-semantics.git
