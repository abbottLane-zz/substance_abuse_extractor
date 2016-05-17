CRFClassifier invoked on Mon May 16 12:51:53 PDT 2016 with arguments:
   -loadClassifier EntityExtractor/Models/model-Status.ser.gz -testFile EntityExtractor/Test-Files/test-Status.tsv
testFile=EntityExtractor/Test-Files/test-Status.tsv
loadClassifier=EntityExtractor/Models/model-Status.ser.gz
Loading classifier from /home/wlane/compling/biomed-nlp-575/substance_abuse_extractor/src/EntityExtractor/Models/model-Status.ser.gz ... done [0.3 sec].
CRFClassifier tagged 804 words in 1 documents at 2429.00 words per second.
         Entity	P	R	F1	TP	FP	FN
              0	0.5375	0.4624	0.4971	43	37	50
         Status	0.6962	0.5978	0.6433	55	24	37
         Totals	0.6164	0.5297	0.5698	98	61	87
