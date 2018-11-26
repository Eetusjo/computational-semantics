#!/bin/bash

cd /home/jovyan/

rm -r work
rm bootstrap.bash

export GIT_COMMITTER_NAME=anonymous
export GIT_COMMITTER_EMAIL=anon@localhost
git clone https://github.com/Eetusjo/computational-semantics.git

# cd computational-semantics/assignment5/data/quora/
# gunzip quora_duplicate_questions.tsv.gz 

# cd /home/jovyan/

pip install spacy
python -m spacy download en
