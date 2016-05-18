CRFClassifier invoked on Wed May 18 14:29:52 PDT 2016 with arguments:
   -loadClassifier EntityExtractor/Models/model-Amount.ser.gz -testFile EntityExtractor/Test-Files/test-Amount.tsv
testFile=EntityExtractor/Test-Files/test-Amount.tsv
loadClassifier=EntityExtractor/Models/model-Amount.ser.gz
Loading classifier from /home/wlane/compling/biomed-nlp-575/substance_abuse_extractor/src/EntityExtractor/Models/model-Amount.ser.gz ... done [0.2 sec].
CRFClassifier tagged 920 words in 1 documents at 3680.00 words per second.
         Entity	P	R	F1	TP	FP	FN
              0	0.5455	0.4444	0.4898	12	10	15
         Amount	0.7143	0.5769	0.6383	15	6	11
         Totals	0.6279	0.5094	0.5625	27	16	26
