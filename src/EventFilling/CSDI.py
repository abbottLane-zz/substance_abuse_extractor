from DataLoader import Globals as g


def evaluate_csdi(predicted_events, gold_events, correct_counts, sub_counts, del_counts, insert_counts,
                  events_file, status_file, attrib_file):
    # Identify correct, substitutions, insertions; Identify deletions from matched events
    predictions_csdi(predicted_events, gold_events, correct_counts, sub_counts, del_counts, insert_counts,
                     events_file, status_file, attrib_file)

    # Identify deletions - missing pred events, duplicate gold events
    gold_deletions(predicted_events, gold_events, del_counts, events_file, status_file, attrib_file)


def predictions_csdi(predicted_events, gold_events, correct_counts, sub_counts, del_counts, insert_counts,
                     events_file, status_file, attrib_file):
    for pred_event in predicted_events:
        found_type = False
        seen_types = set()

        # Search gold events for match to predicted event
        for gold_event in gold_events:
            # Make sure gold type is substance, don't give credit for matching duplicate event types
            if gold_event.type in g.SPECIFIC_CLASSIFIER_TYPES and gold_event.type not in seen_types:
                seen_types.add(gold_event.type)
                if pred_event.type == gold_event.type:
                    found_type = True
                    matching_type_csdi(pred_event, gold_event, correct_counts, sub_counts, del_counts, insert_counts,
                                       status_file, attrib_file)

        # Predicted event not in gold
        if not found_type:
            inserted_pred_event(pred_event, insert_counts, events_file, status_file, attrib_file)


def gold_deletions(predicted_events, gold_events, del_counts, events_file, status_file, attrib_file):
    seen_types = set()
    for gold_event in gold_events:
        if gold_event not in seen_types:
            if gold_event.type in g.SPECIFIC_CLASSIFIER_TYPES:
                seen_types.add(gold_event.type)

                # Check if there is a corresponding pred event
                found_match = False
                for predicted_event in predicted_events:
                    if predicted_event.type == gold_event.type:
                        found_match = True

                # No match: deletions
                if not found_match:
                    deleted_gold_event(gold_event, del_counts, events_file, status_file, attrib_file)

        # Duplicate event type: deletions
        else:
            deleted_gold_event(gold_event, del_counts, events_file, status_file, attrib_file)


def matching_type_csdi(predicted_event, gold_event, correct_counts, sub_counts, del_counts, insert_counts,
                       status_file, attrib_file):
    # Type
    correct_counts[predicted_event.type] += 1

    # Status
    if predicted_event.status == gold_event.get_status():
        correct_counts[g.STATUS] += 1
    else:
        sub_counts[g.STATUS] += 1
        status_file.write("Sub in " + predicted_event.type + ", should be [" + gold_event.get_status() +
                          "] but is [" + predicted_event.status + "]\n")

    # Attributes
    pred_attribs = predicted_event.attributes_list
    gold_attribs = gold_event.dict_of_attribs

    attrib_csi(pred_attribs, gold_attribs, correct_counts, sub_counts, insert_counts, attrib_file)
    attrib_deletions(pred_attribs, gold_attribs, del_counts, attrib_file)


def inserted_pred_event(predicted_event, insert_counts, events_file, status_file, attrib_file):
    # Type
    insert_counts[predicted_event.type] += 1
    events_file.write("Inserted " + predicted_event.type + "\n")

    # Status
    insert_counts[g.STATUS] += 1
    status_file.write("Inserted due to event " + predicted_event.type + "\n")

    # Attributes
    for attrib in predicted_event.attributes_list:
        insert_counts[attrib.type] += 1
        attrib_file.write("Inserted due to event " + predicted_event.type + "\n")


def deleted_gold_event(gold_event, del_counts, events_file, status_file, attrib_file):
    # Type
    del_counts[gold_event.type] += 1
    events_file.write("Deleted " + gold_event.type + "\n")

    # Status
    del_counts[g.STATUS] += 1
    status_file.write("Deleted due to deleted event " + gold_event.type + "\n")

    # Attributes
    for attrib in gold_event.dict_of_attribs.values():
        del_counts[attrib.type] += 1
        attrib_file.write("Deleted due to deleted event " + gold_event.type + "\n")


def attrib_csi(pred_attribs, gold_attribs, correct_counts, sub_counts, insert_counts, attrib_file):
    seen_types = set()
    for pred_attrib in pred_attribs:
        if pred_attrib.type not in seen_types:
            found_match = False

            for gold_attrib in gold_attribs.values():
                if pred_attrib.type == gold_attrib.type:
                    found_match = True

                    exact, partial = compare_spans(int(pred_attrib.span_begin), int(pred_attrib.span_end),
                                                   int(gold_attrib.span_begin), int(gold_attrib.span_end))
                    if exact:
                        correct_counts[pred_attrib.type] += 1
                    else:
                        sub_counts[pred_attrib.type] += 1
                        attrib_file.write("Sub, should be [" + gold_attrib.text + "] " +
                                          str(gold_attrib.span_begin) + " " + str(gold_attrib.span_end) +
                                          "but is [" + pred_attrib.text + "] " +
                                          str(pred_attrib.span_begin) + " " + str(pred_attrib.span_end) + "\n")

            # Inserted attribute
            if not found_match:
                insert_counts[pred_attrib.type] += 1
                attrib_file.write("Inserted " + pred_attrib.type + " [" + pred_attrib.text + "] " +
                                  str(pred_attrib.span_begin) + " " + str(pred_attrib.span_end) + "\n")

        # Duplicate type - gold only has one
        else:
            insert_counts[pred_attrib.type] += 1
            attrib_file.write("Insert - Duplicate " + pred_attrib.type + "\n")


def attrib_deletions(pred_attribs, gold_attribs, del_counts, attrib_file):
    for gold_attrib in gold_attribs.values():
        if gold_attrib.type in g.ATTRIBUTE_TYPES:
            found_match = False
            for pred_attrib in pred_attribs:
                if gold_attrib.type == pred_attrib.type:
                    found_match = True

            if not found_match:
                del_counts[gold_attrib.type] += 1
                attrib_file.write("Deleted " + gold_attrib.type + " [" + gold_attrib.text + "] " +
                                  str(gold_attrib.span_begin) + " " + str(gold_attrib.span_end) + "\n")


def compare_spans(gold_begin, gold_end, pred_begin, pred_end):
    exact = False
    partial = False

    if int(gold_begin) == pred_begin and int(gold_end) == pred_end:
        exact = True
        partial = True
    elif int(gold_end) >= pred_begin and int(gold_begin) <= pred_end:
        partial = True

    return exact, partial


def calculate_precision_recall(correct_counts, sub_counts, del_counts, insert_counts):
    outfile = open("System_Performace.txt", "a")

    # Event Detection
    outfile.write("<<< Event Detection >>>\n")
    correct = sum([correct_counts[t] for t in correct_counts if (t in g.SPECIFIC_CLASSIFIER_TYPES)])
    subs = sum([sub_counts[t] for t in sub_counts if (t in g.SPECIFIC_CLASSIFIER_TYPES)])
    deletions = sum([del_counts[t] for t in del_counts if (t in g.SPECIFIC_CLASSIFIER_TYPES)])
    insertions = sum([insert_counts[t] for t in insert_counts if (t in g.SPECIFIC_CLASSIFIER_TYPES)])

    csdi_precision_recall(correct, subs, deletions, insertions, outfile)

    # Status Classification
    outfile.write("<<< Status Classification >>>\n")
    correct = sum([correct_counts[t] for t in correct_counts if (t == g.STATUS)])
    subs = sum([sub_counts[t] for t in sub_counts if (t == g.STATUS)])
    deletions = sum([del_counts[t] for t in del_counts if (t == g.STATUS)])
    insertions = sum([insert_counts[t] for t in insert_counts if (t == g.STATUS)])

    csdi_precision_recall(correct, subs, deletions, insertions, outfile)

    # Attribute Extraction
    outfile.write("<<< Attribute Extraction >>>\n")
    correct = sum([correct_counts[t] for t in correct_counts if (t in g.ATTRIBUTE_TYPES)])
    subs = sum([sub_counts[t] for t in sub_counts if (t in g.ATTRIBUTE_TYPES)])
    deletions = sum([del_counts[t] for t in del_counts if (t in g.ATTRIBUTE_TYPES)])
    insertions = sum([insert_counts[t] for t in insert_counts if (t in g.ATTRIBUTE_TYPES)])

    csdi_precision_recall(correct, subs, deletions, insertions, outfile)

    # System
    outfile.write("<<< Full System >>>\n")
    correct = sum([correct_counts[t] for t in correct_counts])
    subs = sum([sub_counts[t] for t in sub_counts])
    deletions = sum([del_counts[t] for t in del_counts])
    insertions = sum([insert_counts[t] for t in insert_counts])

    csdi_precision_recall(correct, subs, deletions, insertions, outfile)

    outfile.write("----------------------------------------------\n----------------------------------------------\n")
    outfile.close()


def csdi_precision_recall(correct, subs, deletions, insertions, outfile):
    precision = 0
    recall = 0
    f_score = 0

    if correct:
        precision = correct / (correct + subs + insertions)
        recall = correct / (correct + subs + deletions)
        f_score = (2 * precision * recall) / (precision + recall)

    outfile.write("Correct: " + str(correct) + "\n")
    outfile.write("Sub:     " + str(subs) + "\n")
    outfile.write("Del:     " + str(deletions) + "\n")
    outfile.write("Insert:  " + str(insertions) + "\n\n")

    outfile.write("Precision: " + str(precision) + "\n")
    outfile.write("Recall: " + str(recall) + "\n")
    outfile.write("F1: " + str(f_score) + "\n\n")
