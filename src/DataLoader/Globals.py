# Files
CLASSF_EVAL_FILE = "classifier_results.txt"

# Classifier Types
SUBSTANCE = "SUBSTANCE"
ALCOHOL = "Alcohol"
DRUGS = "Drug"
TOBACCO = "Tobacco"
SPECIFIC_CLASSIFIER_TYPES = [ALCOHOL, DRUGS, TOBACCO]

# Classification Labels
HAS_SUBSTANCE = "has_subs_info"
NO_SUBSTANCE = "no_subs_info"

# Word/Gram Classes
NUMBER = "NUMBER"
DECIMAL = "DECIMAL"
MONEY = "MONEY"
PERCENT = "PERCENT"

# Status values
NONE = "none"
UNKNOWN = "unknown"
PAST = "past"
CURRENT = "current"

# Attribute types
STATUS = "Status"
TEMPORAL = "Temporal"
METHOD = "Method"
TYPE = "Type"
AMOUNT = "Amount"
FREQ = "Frequency"
HISTORY = "History"
ATTRIBUTE_TYPES = {STATUS, TEMPORAL, METHOD, TYPE, AMOUNT, FREQ, HISTORY}

# Event Filler Features
HAS_GRAM = "HAS_GRAM_"
ATTRIB_TYPE = "ATTRIB_TYPE_"
EVENT_TYPE = "EVENT_TYPE_"
SURR_WORD = "SURR_WORD_"

# POI types represent the points of information important for evaluation of TP/FP/etc
POI_TYPES = {ALCOHOL, DRUGS, TOBACCO}.union(ATTRIBUTE_TYPES)
