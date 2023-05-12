import os
import json


# Code taken from
# https://gitlab.com/hwu-ilab/hermit-nlu/blob/master/data/nlu_benchmark/
# to ensure same evaluation metrics as https://arxiv.org/abs/1910.00912
# Additional methods were added to be able to call desired evaluation methods.


arg_format_pattern = r"\[\s*(?P<label>[\w']*)\s*:(?P<filler>[\s\w'\.@\-&+]+)\]"
arg_annotation_pattern = r"\[\s*[\w']*\s*:[\s\w'\.@\-&+]+\]"


def load_json_prediction_file(predictions_file):
    _, filename = os.path.split(predictions_file)
    with open(predictions_file, "r") as f:
        json_prediction = json.load(f)
        f.close()
    return json_prediction


def squeeze_prediction_span(json_prediction):
    squeezed_predictions = []
    for example in json_prediction:
        frame_pred_set = set()
        dialogue_act_pred_set = set()
        frame_gold_set = set()
        dialogue_act_gold_set = set()
        entities_gold = []
        entities_pred = []
        current_frame_element_gold = ""
        current_frame_element_pred = ""
        intent_gold_set = set(example["intent_gold"])
        intent_pred_set = set(example["intent_pred"])
        for frame_element_token, token in zip(
            example["frame_element_gold"], example["tokens"]
        ):
            if frame_element_token == "O":
                continue
            if frame_element_token.startswith("B-"):
                entity_gold = {}
                entities_gold.append(entity_gold)
                current_frame_element_gold = frame_element_token[2:]
                entity_gold[current_frame_element_gold] = [token]
            elif frame_element_token[2:] == current_frame_element_gold:
                entity_gold[current_frame_element_gold].append(token)
            else:
                entity_gold = {}
                entities_gold.append(entity_gold)
                current_frame_element_gold = frame_element_token[2:]
                entity_gold[current_frame_element_gold] = [token]
        for frame_element_token, token in zip(
            example["frame_element_pred"], example["tokens"]
        ):
            if frame_element_token == "O":
                continue
            if frame_element_token.startswith("B-"):
                entity_pred = {}
                entities_pred.append(entity_pred)
                entity_pred[frame_element_token[2:]] = [token]
            elif frame_element_token[2:] == current_frame_element_pred:
                entity_pred[current_frame_element_pred].append(token)
            else:
                entity_pred = {}
                entities_pred.append(entity_pred)
                current_frame_element_pred = frame_element_token[2:]
                entity_pred[current_frame_element_pred] = [token]

        new_example = {
            "tokens": example["tokens"],
            "dialogue_act_gold": list(dialogue_act_gold_set),
            "dialogue_act_pred": list(dialogue_act_pred_set),
            "frame_gold": list(frame_gold_set),
            "frame_pred": list(frame_pred_set),
            "intent_gold": list(intent_gold_set),
            "intent_pred": list(intent_pred_set),
            "entities_gold": entities_gold,
            "entities_pred": entities_pred,
        }
        squeezed_predictions.append(new_example)
    return squeezed_predictions


