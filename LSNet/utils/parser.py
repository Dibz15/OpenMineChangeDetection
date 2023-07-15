import argparse as ag
import json
from collections import OrderedDict

def get_parser_with_args(metadata_json='metadata.json'):
    json_str = ''
    with open(metadata_json, 'r') as f:
        for line in f:
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    return opt
