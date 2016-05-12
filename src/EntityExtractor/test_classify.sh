#!/usr/bin/env bash

java -cp $1 edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier $2 -testFile $3
