CRFClassifier invoked on Wed May 18 15:24:24 PDT 2016 with arguments:
   -loadClassifier EntityExtractor/Models/model-Amount.ser.gz -testFile EntityExtractor/Test-Files/test-Amount.tsv
testFile=EntityExtractor/Test-Files/test-Amount.tsv
loadClassifier=EntityExtractor/Models/model-Amount.ser.gz
Loading classifier from /home/wlane/compling/biomed-nlp-575/substance_abuse_extractor/src/EntityExtractor/Models/model-Amount.ser.gz ... done [0.2 sec].
CRFClassifier tagged 920 words in 1 documents at 2562.67 words per second.
         Entity	P	R	F1	TP	FP	FN
              0	0.4091	0.3333	0.3673	9	13	18
         Amount	0.6667	0.5385	0.5957	14	7	12
         Totals	0.5349	0.4340	0.4792	23	20	30
