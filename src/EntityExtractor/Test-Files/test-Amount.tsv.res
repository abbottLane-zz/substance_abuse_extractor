CRFClassifier invoked on Mon May 16 12:51:56 PDT 2016 with arguments:
   -loadClassifier EntityExtractor/Models/model-Amount.ser.gz -testFile EntityExtractor/Test-Files/test-Amount.tsv
testFile=EntityExtractor/Test-Files/test-Amount.tsv
loadClassifier=EntityExtractor/Models/model-Amount.ser.gz
Loading classifier from /home/wlane/compling/biomed-nlp-575/substance_abuse_extractor/src/EntityExtractor/Models/model-Amount.ser.gz ... done [0.2 sec].
CRFClassifier tagged 804 words in 1 documents at 3295.08 words per second.
         Entity	P	R	F1	TP	FP	FN
              0	0.4545	0.3846	0.4167	10	12	16
         Amount	0.6190	0.5200	0.5652	13	8	12
         Totals	0.5349	0.4510	0.4894	23	20	28
