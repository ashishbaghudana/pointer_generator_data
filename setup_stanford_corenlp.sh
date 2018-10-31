#!/bin/bash
DOWNLOAD_URL="http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip"
DIRECTORY="stanford-corenlp-full-2018-10-05"
ZIP_FILE="stanford-corenlp-full-2018-10-05.zip"
VERSION="3.9.2"
CORENLP_JAR="stanford-corenlp-$VERSION.jar"

if [ ! -d "$DIRECTORY" ]; then
  curl -L $DOWNLOAD_URL -o $ZIP_FILE
  unzip $ZIP_FILE
fi

export CLASSPATH=$(pwd)/$DIRECTORY/$CORENLP_JAR
