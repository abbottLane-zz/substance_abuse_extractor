CRFClassifier invoked on Mon May 16 12:24:29 PDT 2016 with arguments:
   -loadClassifier EntityExtractor/Models/model-Amount.ser.gz -testFile EntityExtractor/Test-Files/test-Amount.tsv
testFile=EntityExtractor/Test-Files/test-Amount.tsv
loadClassifier=EntityExtractor/Models/model-Amount.ser.gz
Loading classifier from /home/wlane/compling/biomed-nlp-575/substance_abuse_extractor/src/EntityExtractor/Models/model-Amount.ser.gz ... done [0.2 sec].
CRFClassifier tagged 804 words in 1 documents at 3621.62 words per second.
         Entity	P	R	F1	TP	FP	FN
              0	0.3913	0.3462	0.3673	9	14	17
         Amount	0.6364	0.5600	0.5957	14	8	11
         Totals	0.5111	0.4510	0.4792	23	22	28
