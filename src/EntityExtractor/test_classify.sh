#!/usr/bin/env bash
#echo "COMMAND: java -cp $1 edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier $2 -testFile $3 1> "$3.out" 2> "$3.res""
java -cp $1 edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier $2 -testFile $3 1> "$3.out" 2> "$3.res"
