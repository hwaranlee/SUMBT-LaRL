
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import sqlite3
import numpy as np
import copy
import os
import random
import re
import math
import pprint

from nltk.stem.porter import *

stemmer = PorterStemmer()

requestable_slots = {
    'restaurant':   ['name', 'food', 'area', 'pricerange', 'phone', 'postcode', 'address'],
    'hotel':        ['name', 'type', 'area', 'pricerange', 'stars', 'internet', 'parking', 'phone', 'postcode', 'address'],
    'attraction':   ['name', 'type', 'area', 'phone', 'postcode', 'address', 'entrance fee'],
    'hospital':     ['department', 'phone'],
    'taxi':         ['departure', 'destination', 'leaveAt', 'arriveBy'],
    'train':        ['trainID', 'departure', 'destination', 'day', 'leaveAt', 'arriveBy', 'price', 'duration']
}
constraint_slots = {
    'restaurant':   ['food', 'area', 'pricerange'],
    'hotel':        ['type', 'area', 'pricerange', 'stars', 'internet', 'parking'],
    'attraction':   ['type', 'area'],
    'hospital':     ['department'],
    'taxi':         ['departure', 'destination', 'leaveAt', 'arriveBy'],
    'train':        ['departure', 'destination', 'day', 'leaveAt', 'arriveBy']
}

booking_slots = {
    'restaurant':   ['day', 'time', 'people'],
    'hotel':        ['day', 'stay', 'people'],
    'attraction':   [],
    'hospital':     [],
    'taxi':         [],
    'train':        ['people'],
    'police':       []
}

# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
dbs = {}
for domain in domains:
    dbs[domain] = json.load(open('.//convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/{}_db.json'.format(domain)))

# loading value_dict.json
value_dict = json.load(open('.//convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/value_dict.json'))

# slot value augmentation
restaurant_s2v = json.load(open('.//convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/restaurant_s2v.json'))
value_dict['restaurant']['food'].extend(restaurant_s2v['food'])

def fuzzy_string_distance(s, t, costs=(1, 1, 1)):
    """
    iterative_levenshtein(s, t) -> ldist
    ldist is the Levenshtein distance between the strings s and t.
    For all i and j, dist[i,j] will contain the Levenshtein distance
    between the first i characters of s and the first j characters of t
    """
    rows = len(s) + 1
    cols = len(t) + 1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    col = row = 0
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(
                dist[row - 1][col] + deletes, dist[row][col - 1] + inserts,
                dist[row - 1][col - 1] + cost)  # substitution

    return dist[row][col]

def value_grounding(domain, slot, value):
    FUZZY_MATCH_THRESHOLD = 0.2

    if value in ['', 'not mentioned', 'dontcare', 'dont care', "don't care", "do n't care", "do nt care", 'none']:
        return ''

    if value == 'el shaddia guesthouse':
        return 'el shaddai'

    if value == 'meze bar restaurant':
        return 'meze bar'

    if value == 'pizza hut':
        return ['pizza hut city centre', 'pizza hut cherry hinton', 'pizza hut fen ditton']

    if slot == 'type' and value == 'special':
        return ''

    domain = domain.lower()
    slot = slot.lower()
    highest_ratio = -1
    best_match = value
    for candidate in value_dict[domain][slot]:
        distance = fuzzy_string_distance(value.replace(' ', '').lower(),
                                         candidate.replace(' ', '').lower(),
                                         (1, 0.1, 1))
        similarity = len(value) - distance
        ratio = math.exp(-distance / 5.0) * (similarity / max(float(len(value)), 1e-5))

        if highest_ratio < ratio:
            highest_ratio = ratio
            best_match = candidate

    if highest_ratio >= FUZZY_MATCH_THRESHOLD:
        return best_match
    else:
        return value

def query(domain, belief, ignore_open=True):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""

    nonactive_domain_flag = all(val == '' for slot, val in belief.items())

    constraints = []
    for slot in belief:
        if belief[slot] != "":
            val = belief[slot]
            if '|' in val:
                val_list = val.split('|')
                constraints.append([slot, val_list])
            elif '>' in val:
                val_list = val.split('>')
                constraints.append([slot, val_list])
            elif '<' in val:
                val_list = val.split('<')
                constraints.append([slot, val_list])
            else:
                constraints.append([slot, val])

    # query the db
    if domain == 'taxi':
        return [{'type': ' '.join([random.choice(dbs[domain]['taxi_colors']), random.choice(dbs[domain]['taxi_types'])]), 'phone': ''.join([str(random.randint(1, 9)) for _ in range(11)])}], nonactive_domain_flag
    if domain == 'police':
        return dbs['police'], nonactive_domain_flag

    # Perform value grounding
    _constraints = []
    for index, (slot, value) in enumerate(constraints):
        if slot not in ['leaveAt', 'arriveBy']:
            if type(value) == list:
                _elem = []
                for i, val in enumerate(value):
                    _elem.append(value_grounding(domain, slot, val))
                _constraints.append([slot, _elem])
            elif type(value) == str:
                _constraints.append([slot, value_grounding(domain, slot, value)])
            else:
                raise NotImplementedError()
        else:
            _constraints.append([slot, value])
    constraints = _constraints

    found = []
    for i, record in enumerate(dbs[domain]):
        for key, val in constraints:
            if type(val) == list:
                try:
                    record_keys = [key.lower() for key in record]
                    if key.lower() not in record_keys and stemmer.stem(key) not in record_keys:
                        continue
                    if key == 'leaveAt':
                        val1_list = [int(val_temp.split(':')[0]) * 100 + int(val_temp.split(':')[1]) for val_temp in val if val_temp not in ['', 'not mentioned', 'dontcare', 'dont care', "don't care", "do n't care", "do nt care", 'none']]
                        val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                        #if val1_0 > val2 and val1_1 > val2:
                        if all(val_temp > val2 for val_temp in val1_list):
                            break
                    elif key == 'arriveBy':
                        val1_list = [int(val_temp.split(':')[0]) * 100 + int(val_temp.split(':')[1]) for val_temp in val if val_temp not in ['', 'not mentioned', 'dontcare', 'dont care', "don't care", "do n't care", "do nt care", 'none']]
                        val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                        if all(val_temp < val2 for val_temp in val1_list):
                            break
                    elif ignore_open and key in ['destination', 'departure']:
                        continue
                    else:
                        if all(val_temp.strip() != record[key].strip() for val_temp in val if val_temp not in ['', 'not mentioned', 'dontcare', 'dont care', "don't care", "do n't care", "do nt care", 'none']):
                            break
                except:
                    continue
            else:
                if val == "" or val == "dont care" or val == "not mentioned" or val == "don't care" or val == "dontcare" or val == "do n't care":
                    pass
                else:
                    try:
                        record_keys = [key.lower() for key in record]
                        if key.lower() not in record_keys and stemmer.stem(key) not in record_keys:
                            continue
                        if key == 'leaveAt':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        elif key == 'arriveBy':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        elif ignore_open and key in ['destination', 'departure']:
                            continue
                        else:
                            if val.strip() != record[key].strip():
                                break
                    except:
                        continue
        else:
            record['reference'] = f'{i:08d}'
            found.append(record)

    return found, nonactive_domain_flag

def _query_length_feature(num, domain):
    """Return number of available entities for particular domain."""
    if domain == 'train':
        if num == 0:
            vector = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector = np.array([0, 0, 0, 0, 0, 1])
    elif domain == 'hospital':
        if num == 0:
            vector = np.array([1, 0])
        elif num >= 1:
            vector = np.array([0, 1])
    else:
        if num == 0:
            vector = np.array([1, 0, 0, 0, 0, 0])
        elif num == 1:
            vector = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector = np.array([0, 0, 0, 0, 0, 1])

    return vector

def _booking_feature(state, pointer_vector, mode=None):
    """Add information about availability of the booking option."""
    # Booking pointer

    if mode == 'predict':
        rest_vec = np.array([1, 0])
        if "book" in state['restaurant']:
            if state['restaurant']['book']['people'] not in ['', 'none', 'not mentioned', 'dontcare'] and state['restaurant']['book']['day'] not in ['', 'none', 'not mentioned', 'dontcare'] and state['restaurant']['book']['time'] not in ['', 'none', 'not mentioned', 'dontcare']:
                rest_vec = np.array([0, 1])

        hotel_vec = np.array([1, 0])
        if "book" in state['hotel']:
            if state['hotel']['book']['people'] not in ['', 'none', 'not mentioned', 'dontcare'] and state['hotel']['book']['day'] not in ['', 'none', 'not mentioned', 'dontcare'] and state['hotel']['book']['stay'] not in ['', 'none', 'not mentioned', 'dontcare']:
                hotel_vec = np.array([0, 1])

        train_vec = np.array([1, 0])
        if "book" in state['train']:
            if state['train']['book']['people'] not in ['', 'none', 'not mentioned', 'dontcare']:
                train_vec = np.array([0, 1])

    else:
        rest_vec = np.array([0])
        if "book" in state['restaurant']:
            if "booked" in state['restaurant']['book']:
                if state['restaurant']['book']["booked"]:
                    if "reference" in state['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([1])

        hotel_vec = np.array([0])
        if "book" in state['hotel']:
            if "booked" in state['hotel']['book']:
                if state['hotel']['book']["booked"]:
                    if "reference" in state['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([1])

        train_vec = np.array([0])
        if "book" in state['train']:
            if "booked" in  state['train']['book']:
                if state['train']['book']["booked"]:
                    if "reference" in state['train']['book']["booked"][0]:
                        train_vec = np.array([1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector

def make_db_feature(state, mode=None):
    db_feature_domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital']
    kb_result = {'restaurant': [], 'hotel': [], 'attraction': [], 'train': [], 'hospital': []}

    db_feature_vec = np.array([], dtype=np.float32)
    domain_active_feature_vec = np.array([], dtype=np.float32)
    for domain in db_feature_domains:

        if domain != 'train':
            kb_result[domain], nonactive_domain_flag = query(domain, state[domain]['semi'])
        else:
            kb_result[domain], nonactive_domain_flag = query(domain, state[domain]['semi'], ignore_open=False)

        if nonactive_domain_flag:
            db_feature_vec = np.hstack((db_feature_vec, np.zeros(1)))
            domain_active_feature_vec = np.hstack((domain_active_feature_vec, np.zeros(1)))
        else:
            db_feature_vec = np.hstack((db_feature_vec, np.ones(1)))
            domain_active_feature_vec = np.hstack((domain_active_feature_vec, np.ones(1)))

        db_feature_vec = np.hstack((db_feature_vec, _query_length_feature(len(kb_result[domain]), domain)))

        if domain != 'hospital':
            entropy_feature = np.zeros(len(constraint_slots[domain]))
            if not nonactive_domain_flag and len(kb_result[domain]) != 0:
                for i, slot in enumerate(constraint_slots[domain]):
                    count_val_dict = {}
                    for res in kb_result[domain]:
                        if res[slot] in count_val_dict:
                            count_val_dict[res[slot]] += 1
                        else:
                            count_val_dict[res[slot]] = 1

                    if len(kb_result[domain]) == 1 or len(count_val_dict) == 1:
                        entropy_feature[i] = 0.0
                    else:
                        entropy = 0.0
                        for val in count_val_dict:
                            entropy += count_val_dict[val] / len(kb_result[domain]) * np.log2(count_val_dict[val] / len(kb_result[domain]))
                        entropy_feature[i] = -entropy / np.log2(len(count_val_dict))

            db_feature_vec = np.hstack((db_feature_vec, entropy_feature))

        if domain == 'hotel':
            hotel_type_feature = np.zeros(2)
            hotel_parking_feature = np.zeros(2)
            hotel_internet_feature = np.zeros(2)
            if not nonactive_domain_flag:
                for res in kb_result[domain]:
                    if res['type'] == 'hotel':
                        hotel_type_feature[0] = 1.0
                    elif res['type'] == 'guesthouse':
                        hotel_type_feature[1] = 1.0
                    else:
                        pass

                    if res['parking'] == 'yes':
                        hotel_parking_feature[0] = 1.0
                    elif res['parking'] == 'no':
                        hotel_parking_feature[1] = 1.0
                    else:
                        pass

                    if res['internet'] == 'yes':
                        hotel_internet_feature[0] = 1.0
                    elif res['internet'] == 'no':
                        hotel_internet_feature[1] = 1.0
                    else:
                        pass
            db_feature_vec = np.hstack((db_feature_vec, hotel_type_feature))
            db_feature_vec = np.hstack((db_feature_vec, hotel_parking_feature))
            db_feature_vec = np.hstack((db_feature_vec, hotel_internet_feature))

        if domain == 'attraction':
            entrance_fee_feature = np.zeros(3)
            if not nonactive_domain_flag:
                for res in kb_result[domain]:
                    if res['entrance fee'] == 'free':
                        entrance_fee_feature[1] = 1.0
                    elif res['entrance fee'] == 'unknown':
                        entrance_fee_feature[2] = 1.0
                    else:
                        entrance_fee_feature[0] = 1.0
            db_feature_vec = np.hstack((db_feature_vec, entrance_fee_feature))

    db_feature_vec = _booking_feature(state, db_feature_vec, mode)

    return db_feature_vec, domain_active_feature_vec, kb_result


value_count_0_template = \
[
    "i am sorry but i have not found any matches . would you like to try something else ?",
    "i am sorry there are no matches that meet your criteria .",
    "i do not have anything meeting that criteria . can i try something else ?",
    "unfortunately there are no matches for that criteria .",
    "i am sorry , there are no [active_domain] -s that meet your criteria . would you like to try something else ?",
]


def populate_template(template, top_results, num_results, state, active_domain_input):
    #active_domain = active_domain_input if len(top_results.keys()) == 0 else list(top_results.keys())[0]
    active_domain = None if active_domain_input is None else active_domain_input
    template = template.replace('book [value_count] of', 'book one of')
    template = template.replace('book [value_count] for you', 'book one for you')
    template = template.replace('would you like to book [value_count]', 'would you like to book one')
    template = template.replace('[cambridge_towninfo_centre]', 'Cambridge Towninfo Centre')
    template = template.replace('the table will be reserved for [value_count] minutes', 'the table will be reserved for 15 minutes')

    if active_domain is not None:
        if num_results[active_domain] == 0:
            if '[value_count]' in template and 'that meet your criteria' in template:
                template = random.choice(value_count_0_template)
        template = template.replace('[active_domain]', active_domain)

    tokens = template.split()
    response = []

    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            domain = token[1:-1].split('_')[0]
            slot = token[1:-1].split('_')[1]
            if domain == 'train':
                if slot == 'trainid':
                    slot = 'trainID'
                elif slot == 'leaveat':
                    slot = 'leaveAt'
                elif slot == 'arriveby':
                    slot = 'arriveBy'
            if domain == 'attraction' and slot == 'entrance':
                slot = 'entrance fee'
            if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain]:
                response.append(top_results[domain][slot])
            elif domain == 'value':
                if slot == 'count':
                    if active_domain is not None:
                        response.append(str(num_results[active_domain]))
                    else:
                        response.append(token)
                elif slot == 'place':
                    if 'arrive' in response:
                        for d in state:
                            if d == 'history':
                                continue
                            if 'destination' in state[d]['semi'] and state[d]['semi']['destination'] not in ['', 'none']:
                                response.append(state[d]['semi']['destination'])
                                break
                    elif 'leave' in response:
                        for d in state:
                            if d == 'history':
                                continue
                            if 'departure' in state[d]['semi'] and state[d]['semi']['departure'] not in ['', 'none']:
                                response.append(state[d]['semi']['departure'])
                                break
                    else:
                        try:
                            for d in state:
                                if d == 'history':
                                    continue
                                for s in ['destination', 'departure']:
                                    if s in state[d]['semi'] and s in state[d]['semi'][s] not in ['', 'none']:
                                        response.append(state[d]['semi'][s])
                                        raise
                        except:
                            pass
                        else:
                            if active_domain not in ['train', 'taxi']:
                                response.append('cambridge')
                            else:
                                response.append(token)
                elif slot == 'time':
                    if 'arrive' in ' '.join(response[-3:]):
                        if active_domain is not None and 'arriveBy' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                            response.append(top_results[active_domain]['arriveBy'])
                            continue
                        for d in state:
                            if d == 'history':
                                continue
                            if 'arriveBy' in state[d]['semi'] and state[d]['semi']['arriveBy'] not in ['', 'none']:
                                response.append(state[d]['semi']['arriveBy'])
                                break
                    elif 'leave' in ' '.join(response[-3:]):
                        if active_domain is not None and 'leaveAt' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                            response.append(top_results[active_domain]['leaveAt'])
                            continue
                        for d in state:
                            if d == 'history':
                                continue
                            if 'leaveAt' in state[d]['semi'] and state[d]['semi']['leaveAt'] not in ['', 'none']:
                                response.append(state[d]['semi']['leaveAt'])
                                break
                    elif 'book' in template:
                        if state['restaurant']['book']['time'] not in ['', 'none']:
                            response.append(state['restaurant']['book']['time'])
                    else:
                        try:
                            for d in state:
                                if d == 'history':
                                    continue
                                for s in ['arriveBy', 'leaveAt']:
                                    if s in state[d]['semi'] and state[d]['semi'][s] not in ['', 'none']:
                                        response.append(state[d]['semi'][s])
                                        raise
                        except:
                            pass
                        else:
                            response.append(token)
                elif slot == 'people':
                    if slot in state[active_domain]['book']:
                        if state[active_domain]['book'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['book'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['book'] and state[d]['book'][slot] not in ['', 'none']:
                                response.append(state[d]['book'][slot])
                                break
                        else:
                            response.append(token)
                elif slot == 'day':
                    if slot in state[active_domain]['book']:
                        if state[active_domain]['book'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['book'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['book'] and state[d]['book'][slot] not in ['', 'none']:
                                response.append(state[d]['book'][slot])
                                break
                        else:
                            response.append(token)
                elif slot == 'stay':
                    if slot in state[active_domain]['book']:
                        if state[active_domain]['book'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['book'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['book'] and state[d]['book'][slot] not in ['', 'none']:
                                response.append(state[d]['book'][slot])
                                break
                        else:
                            response.append(token)
                else:
                    # slot-filling based on query results
                    for d in top_results:
                        if slot in top_results[d]:
                            response.append(top_results[d][slot])
                            break
                    else:
                        # slot-filling based on belief state
                        try:
                            if slot in state[active_domain]['semi']:
                                if state[active_domain]['semi'][slot] not in ['', 'none']:
                                    response.append(state[active_domain]['semi'][slot])
                            else:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                                        response.append(state[d]['semi'][slot])
                                        break
                                else:
                                    response.append(token)
                        except:
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                                    response.append(state[d]['semi'][slot])
                                    break
                            else:
                                response.append(token)
            else:
                # slot-filling based on belief state
                try:
                    if slot in state[active_domain]['semi']:
                        if state[active_domain]['semi'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['semi'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                                response.append(state[d]['semi'][slot])
                                break
                        else:
                            response.append(token)
                except:
                    for d in state:
                        if d == 'history':
                            continue
                        if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                            response.append(state[d]['semi'][slot])
                            break
                    else:
                        response.append(token)

                # print(token)
                #response.append(token)
        else:
            response.append(token)

    try:
        response = ' '.join(response)
    except Exception as e:
        print(e)
        import pprint
        pprint.pprint(response)
        raise
    response = response.replace(' -s', 's')
    response = response.replace(' -ly', 'ly')
    response = response.replace(' .', '.')
    response = response.replace(' ?', '?')
    response = response.replace('minutes minutes', 'minutes')
    response = response.replace('free pounds', 'free')
    response = response.replace('pounds pounds', 'pounds')

    return response

def populate_template_v2(template, top_results, num_results, state, active_domain_input):
    # active_domain_input is a list
    # return one active domain among them

    active_domain = None if active_domain_input is None else active_domain_input[0]
    template = template.replace('book [value_count] of', 'book one of')
    template = template.replace('book [value_count] for you', 'book one for you')
    template = template.replace('[cambridge_towninfo_centre]', 'Cambridge Towninfo Centre')
    tokens = template.split()
    response = []

    def response_from_state(domain, slot):
        if slot in state[domain]['semi'] and state[domain]['semi'][slot] not in ['', 'none']:
            return state[domain]['semi'][slot]
        else:
            return None

    # if all db results are null, return template
    n = 0
    for num_result in num_results.values():
        n += num_result
    if n == 0:
        if '[' in template: # which cannot fill the delex slot
            response = 'i am sorry but i have not found any matches for you .'
        return response, active_domain


    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            domain = token[1:-1].split('_')[0]
            slot = token[1:-1].split('_')[1]
            # renaming
            if domain == 'train':
                if slot == 'trainid':
                    slot = 'trainID'
                elif slot == 'leaveat':
                    slot = 'leaveAt'
                elif slot == 'arriveby':
                    slot = 'arriveBy'
            if domain == 'attraction' and slot == 'entrance':
                slot = 'entrance fee'

            # filling from top_results
            if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain]:
                if domain != 'taxi':
                    response.append(top_results[domain][slot])
                    active_domain = domain
                else: # for taxi domain
                    if slot == 'type':
                        response.append(" ".join(top_results['taxi_colors'], top_results['taxi_types']))
                    else:
                        response.append(top_results[domain][slot])

            # filling values_
            elif domain == 'value':

                if slot == 'count':
                    response.append(str(num_results[active_domain]))
                    # hard coding here
                    if num_results[active_domain] == 0:
                        response = 'i am sorry but i have not found any matches for you .'.split()
                        break

                elif slot == 'place':
                    if 'arrive' in response:
                        for d in state:
                            if d != 'history':
                                value = response_from_state(d, 'destination')
                                if value is not None:
                                    response.append(value)
                                    break

                    elif 'leave' in response:
                        for d in state:
                            if d != 'history':
                                value = response_from_state(d, 'departure')
                                if value is not None:
                                    response.append(value)
                                    break
                    else:
                        try:
                            for d in state:
                                if d != 'history':
                                    for s in ['destination', 'departure']:
                                        if s in state[d]['semi'] and s in state[d]['semi'][s] not in ['', 'none']:
                                            response.append(state[d]['semi'][s])
                                            raise
                        except:
                            pass
                        else:
                            response.append(token)

                elif slot == 'time':
                    if 'arrive' in ' '.join(response[-3:]):
                        if active_domain is not None and 'arriveBy' in top_results[active_domain]:
                            response.append(top_results[active_domain]['arriveBy'])
                            continue
                        for d in state:
                            if d == 'history':
                                continue
                            if 'arriveBy' in state[d]['semi'] and state[d]['semi']['arriveBy'] not in ['', 'none']:
                                response.append(state[d]['semi']['arriveBy'])
                                break
                    elif 'leave' in ' '.join(response[-3:]):
                        if active_domain is not None and 'leaveAt' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                            response.append(top_results[active_domain]['leaveAt'])
                            continue
                        for d in state:
                            if d == 'history':
                                continue
                            if 'leaveAt' in state[d]['semi'] and state[d]['semi']['leaveAt'] not in ['', 'none']:
                                response.append(state[d]['semi']['leaveAt'])
                                break
                    elif 'book' in response:
                        if state['restaurant']['book']['time'] not in ['', 'none']:
                            response.append(state['restaurant']['book']['time'])
                    else:
                        try:
                            for d in state:
                                if d == 'history':
                                    continue
                                for s in ['arriveBy', 'leaveAt']:
                                    if s in state[d]['semi'] and state[d]['semi'][s] not in ['', 'none']:
                                        response.append(state[d]['semi'][s])
                                        raise
                        except:
                            pass
                        else:
                            response.append(token)
                elif slot == 'people':
                    if slot in state[active_domain]['book']:
                        if state[active_domain]['book'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['book'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['book'] and state[d]['book'][slot] not in ['', 'none']:
                                response.append(state[d]['book'][slot])
                                break
                        else:
                            response.append(token)
                elif slot == 'day':
                    if slot in state[active_domain]['book']:
                        if state[active_domain]['book'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['book'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['book'] and state[d]['book'][slot] not in ['', 'none']:
                                response.append(state[d]['book'][slot])
                                break
                        else:
                            response.append(token)
                elif slot == 'stay':
                    if slot in state[active_domain]['book']:
                        if state[active_domain]['book'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['book'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['book'] and state[d]['book'][slot] not in ['', 'none']:
                                response.append(state[d]['book'][slot])
                                break
                        else:
                            response.append(token)
                else:
                    # slot-filling based on query results
                    for d in top_results:
                        if slot in top_results[d]:
                            response.append(top_results[d][slot])
                            break
                    else:
                        # slot-filling based on belief state
                        try:
                            if slot in state[active_domain]['semi']:
                                if state[active_domain]['semi'][slot] not in ['', 'none']:
                                    response.append(state[active_domain]['semi'][slot])
                            else:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                                        response.append(state[d]['semi'][slot])
                                        break
                                else:
                                    response.append(token)
                        except:
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                                    response.append(state[d]['semi'][slot])
                                    break
                            else:
                                response.append(token)
            else:
                # slot-filling based on belief state
                try:
                    if slot in state[active_domain]['semi']:
                        if state[active_domain]['semi'][slot] not in ['', 'none']:
                            response.append(state[active_domain]['semi'][slot])
                    else:
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                                response.append(state[d]['semi'][slot])
                                break
                        else:
                            response.append(token)
                except:
                    for d in state:
                        if d == 'history':
                            continue
                        if slot in state[d]['semi'] and state[d]['semi'][slot] not in ['', 'none']:
                            response.append(state[d]['semi'][slot])
                            break
                    else:
                        response.append(token)
        else:
            response.append(token)

    try:
        response = ' '.join(response)
    except Exception as e:
        print(e)
        pprint.pprint(response)
        raise
    response = response.replace(' -s', 's')
    response = response.replace(' -ly', 'ly')
    response = response.replace(' .', '.')
    response = response.replace(' ?', '?')

    return response, active_domain


def get_active_domain(prev_active_domain, prev_state, state):
    domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'hospital', 'police']
    active_domain = None

    for domain in domains:
        if prev_state[domain] != state[domain]:
            active_domain = domain

    if active_domain is None:
        active_domain = prev_active_domain

    return active_domain

def get_active_domain_v2(prev_active_domain, prev_state, state):
    domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'hospital', 'police']
    active_domain = []

    for domain in domains:
        if prev_state[domain] != state[domain]:
            active_domain.append(domain)

    if active_domain is None:
        active_domain = [prev_active_domain]

    return active_domain

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat  = re.compile("\d{1,3}[.]\d{1,2}")

replacements = []
with open('.//convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/mapping.pair') as fin:
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

replacements_number = []
with open('.//convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/mapping_number.pair') as fin:
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements_number.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

with open('.//convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/delex_da_slot_dict.json', 'r') as f:
    delex_da_slot_dict = json.load(f)

def delexicalise_using_da(turn):
    text_delex = copy.deepcopy(turn['text'])

    if turn['span_info']:
        span_info_confidence = True
        text_delex = re.sub('\t', '', text_delex)
        text_delex = re.sub('\n', '', text_delex)
        text_delex = re.sub(' +', ' ', text_delex)
        text_delex_list = text_delex.split(' ')
        for item in turn['span_info']:
            if item[1] == 'Fee':
                if item[2] in delex_da_slot_dict['Fee'].keys():
                    if (int(item[4]) - int(item[3])) == 0:
                        text_delex_list[int(item[3])] = '[attraction_entrance]-pounds'
                    else:
                        if item[2] in ["$ 20", "10 lb", "1 GBP", "5 GBP"]:
                            text_delex_list[int(item[3]):int(item[4])+1] = ['[attraction_entrance]', 'pounds']
                        else:
                            text_delex_list[int(item[3])] = '[attraction_entrance]'
            elif item[1] == 'People':
                for fromx, tox in replacements_number:
                    temp = ' ' + item[2] + ' '
                    item[2] = (temp.lower()).replace(fromx, tox)[1:-1]
                    temp = ' ' + text_delex_list[int(item[3])] + ' '
                    text_delex_list[int(item[3])] = (temp.lower()).replace(fromx, tox)[1:-1]
                if item[2] in delex_da_slot_dict['People'].keys():
                    if text_delex_list[int(item[3])] in item[2]:
                        text_delex_list[int(item[3])] = '[value_people]'
                    else:
                        span_info_confidence = False
            elif item[1] == 'Stars':
                for fromx, tox in replacements_number:
                    temp = ' ' + item[2] + ' '
                    item[2] = (temp.lower()).replace(fromx, tox)[1:-1]
                    temp = ' ' + text_delex_list[int(item[3])] + ' '
                    text_delex_list[int(item[3])] = (temp.lower()).replace(fromx, tox)[1:-1]
                if item[2] in delex_da_slot_dict['Stars'].keys():
                    if (text_delex_list[int(item[3])] in item[2] or '-star' in text_delex_list[int(item[3])]):
                        if '-star' in item[2] or '-star' in text_delex_list[int(item[3])]:
                            text_delex_list[int(item[3])] = '[hotel_stars]-star'
                        else:
                            text_delex_list[int(item[3])] = '[hotel_stars]'
                    else:
                        span_info_confidence = False
            elif item[1] == 'Stay':
                for fromx, tox in replacements_number:
                    temp = ' ' + item[2] + ' '
                    item[2] = (temp.lower()).replace(fromx, tox)[1:-1]
                    temp = ' ' + text_delex_list[int(item[3])] + ' '
                    text_delex_list[int(item[3])] = (temp.lower()).replace(fromx, tox)[1:-1]
                if item[2] in delex_da_slot_dict['Stay'].keys():
                    if text_delex_list[int(item[3])] in item[2]:
                        text_delex_list[int(item[3])] = '[hotel_stay]'
                    else:
                        span_info_confidence = False
            elif item[0].split('-')[0] == 'Train' and item[1] == 'Time':
                if item[2] in delex_da_slot_dict['Train_Time'].keys():
                    if (int(item[4]) - int(item[3])) == 0:
                        text_delex_list[int(item[3])] = '[train_duration]-minute'
                    else:
                        if any(x in (text_delex_list[int(item[3])+1]).lower() for x in ['min', 'hour']):
                            text_delex_list[int(item[3]):int(item[3])+2] = ['[train_duration]', 'minutes']
                        elif any(x in (text_delex_list[int(item[3])+2]).lower() for x in ['min', 'hour']):
                            text_delex_list[int(item[3])+1:int(item[3])+3] = ['[train_duration]', 'minutes']

        if span_info_confidence:
            text_delex = ' '.join(text_delex_list)
        else:
            text_delex = copy.deepcopy(turn['text'])

    return text_delex

def insert_space(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text, sub=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    if sub:
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [train_price] ', text)
        #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insert_space(token, text)

    # insert white space for 's
    text = insert_space('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    for fromx, tox in replacements_number:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def prepare_slot_values_independent():
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'police']
    dic = set()

    # read databases
    for domain in domains:
        fin = open('convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/' + domain + '_db_orig.json')
        db_json = json.load(fin)
        fin.close()

        for ent in db_json:
            for key, val in ent.items():
                if val == '?' or val == 'free':
                    pass
                elif key == 'address':
                    dic.add((normalize(val), '[{}_address]'.format(domain)))
                    if "road" in val:
                        val = val.replace("road", "rd")
                        dic.add((normalize(val),  '[{}_address]'.format(domain)))
                    elif "rd" in val:
                        val = val.replace("rd", "road")
                        dic.add((normalize(val),  '[{}_address]'.format(domain)))
                    elif "st" in val:
                        val = val.replace("st", "street")
                        dic.add((normalize(val),  '[{}_address]'.format(domain)))
                    elif "street" in val:
                        val = val.replace("street", "st")
                        dic.add((normalize(val),  '[{}_address]'.format(domain)))
                elif key == 'name':
                    dic.add((normalize(val),  '[{}_name]'.format(domain)))
                    if "b & b" in val:
                        val = val.replace("b & b", "bed and breakfast")
                        dic.add((normalize(val), '[{}_name]'.format(domain)))
                    elif "bed and breakfast" in val:
                        val = val.replace("bed and breakfast", "b & b")
                        dic.add((normalize(val), '[{}_name]'.format(domain)))
                    elif "hotel" in val and 'gonville' not in val:
                        val = val.replace("hotel", "")
                        dic.add((normalize(val), '[{}_name]'.format(domain)))
                    elif "restaurant" in val:
                        val = val.replace("restaurant", "")
                        dic.add((normalize(val), '[{}_name]'.format(domain)))
                elif key == 'postcode':
                    dic.add((normalize(val), '[{}_postcode]'.format(domain)))
                elif key == 'phone':
                    dic.add((val, '[{}_phone]'.format(domain)))
                elif key == 'trainID':
                    dic.add((normalize(val), '[{}_trainid]'.format(domain)))
                elif key == 'department':
                    dic.add((normalize(val), '[{}_department]'.format(domain)))
                elif key == 'duration':
                    dic.add((normalize(val), '[{}_duration]'.format(domain)))

                # NORMAL DELEX
                elif key == 'area':
                    #dic.add((normalize(val), '[{}_area]'.format(domain)))
                    dic.add((normalize(val), '[value_area]'))
                elif key == 'food':
                    dic.add((normalize(val), '[{}_food]'.format(domain)))
                elif key == 'pricerange':
                    #dic.add((normalize(val), '[{}_pricerange]'.format(domain)))
                    dic.add((normalize(val), '[value_pricerange]'))
                elif key == 'type' and domain == 'attraction':
                    dic.add((normalize(val), '[{}_type]'.format(domain)))
                else:
                    pass

        if domain == 'hospital':
            dic.add((normalize('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.add((normalize('Hills Road'), '[' + domain + '_' + 'address' + ']'))
            dic.add((normalize('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.add(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.add(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.add(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.add((normalize('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

        if domain == 'police':
            dic.add((normalize('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.add((normalize('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.add(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.add(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.add((normalize('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))
            dic.add((normalize('Parkside Police'), '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    fin = open('convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/' + 'train' + '_db_orig.json')
    db_json = json.load(fin)
    fin.close()

    for ent in db_json:
        for key, val in ent.items():
            if key == 'departure':
                dic.add((normalize(val), '[value_place]'))

    fin = open('convlab/modules/e2e/multiwoz/SUMBT_LaRL/utils/db_ours/' + 'taxi' + '_db_orig.json')
    db_json = json.load(fin)
    fin.close()

    for color_item in db_json['taxi_colors']:
        for types_item in db_json['taxi_types']:
            temp = '{} {}'.format(color_item, types_item)
            dic.add((temp, '[taxi_type]'))

    # add specific values:
    for val in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        dic.add((normalize(val), '[value_day]'))

    for val in restaurant_s2v['food']:
        dic.add((normalize(val), '[restaurant_food]'))

    for val in ["boating", "boats", "cinemas", "colleges", "concerthalls", "concert hall", "concert halls", "museums", "nightclubs", "night club", "night clubs", "parks", "swimmingpools", "swimming pool", "swimming pools", "theaters"]:
        dic.add((normalize(val), '[attraction_type]'))

    dic.add((normalize('Cambridge Towninfo Centre'), '[cambridge_towninfo_centre]'))

    # more general values add at the end
    dic = sorted(dic, key=lambda x:len(x[0].split()), reverse=True)
    return dic

def delexicalise_reference_number(sent, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    if turn['metadata']:
        for domain in turn['metadata']:
            if turn['metadata'][domain]['book']['booked']:
                for slot in turn['metadata'][domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'
                    key = normalize(turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent

def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt

def in_list(key, lis):
    for l in lis:
        if key in l:
            return True
    return False

def fix_delex(sent, dialog_act, bs):
    """Given system dialogue acts fix automatic delexicalization."""
    back_sent = copy.copy(sent)

    if bs is not None:
        keys = bs.keys()
        done = False
        if "attraction" in keys:
            if 'restaurant_' in sent and "restaurant" not in keys:
                sent = sent.replace("restaurant_", "attraction_")
                sent = sent.replace("attraction_reference", "restaurant_reference")
                done = True
            if 'hotel_' in sent and "hotel" not in keys:
                sent = sent.replace("hotel_", "attraction_")
                sent = sent.replace("attraction_reference", "hotel_reference")
                done = True
        if "hotel" in keys:
            if 'attraction_' in sent and "attraction" not in keys:
                sent = sent.replace("attraction_", "hotel_")
                done = True
            if 'restaurant_' in sent and "restaurant" not in keys:
                sent = sent.replace("restaurant_", "hotel_")
                done = True
        if 'restaurant' in keys:
            if 'attraction_' in sent and "attraction" not in keys:
                sent = sent.replace("attraction_", "restaurant_")
                done = True
            if 'hotel_' in sent and "hotel" not in keys:
                sent = sent.replace("hotel_", "restaurant_")
                done = True

    if dialog_act:
        keys = dialog_act.keys()
        done = False
        if in_list("Attraction", keys):
            if 'restaurant_' in sent and not in_list("Restaurant", keys):
                sent = sent.replace("restaurant_", "attraction_")
                done = True
            if 'hotel_' in sent and not in_list("Hotel", keys):
                sent = sent.replace("hotel_", "attraction_")
                done = True
        if in_list("Hotel", keys):
            if 'attraction_' in sent and not in_list("Attraction", keys):
                sent = sent.replace("attraction_", "hotel_")
                done = True
            if 'restaurant_' in sent and not in_list("Restaurant", keys):
                sent = sent.replace("restaurant_", "hotel_")
                done = True
        if in_list('Restaurant', keys):
            if 'attraction_' in sent and not in_list("Attraction", keys):
                sent = sent.replace("attraction_", "restaurant_")
                done = True
            if 'hotel_' in sent and not in_list("Hotel", keys):
                sent = sent.replace("hotel_", "restaurant_")
                done = True

        if in_list("Train", keys):
            words = sent.split(' ')
            tmp_time, tmp_place = None, None
            for i, word in enumerate(words):
                if "leav" in word or "depart" in word or "from" in word:
                    tmp_time = "[train_leaveat]"
                    tmp_place = "[train_departure]"
                if "arriv" in word or "get" in word or "go" in word or "to" in word or "desti" in word:
                    tmp_time = "[train_arriveby]"
                    tmp_place = "[train_destination]"
                if word == "[value_time]":
                    if tmp_time is not None:
                        words[i] = tmp_time
                    else:
                        words[i] = "[train_leaveat]"
                if word == "[value_place]":
                    if tmp_place is not None:
                        words[i] = tmp_place
                    else:
                        words[i] = "[train_departure]"
                if word == "[value_day]":
                    words[i] = "[train_day]"
            sent = " ".join(words)

    sent = sent.replace("hotel_food", "restaurant_food")
    sent = sent.replace("attraction_food", "restaurant_food")
    sent = sent.replace("hotel_type", "attraction_type")
    sent = sent.replace("restaurant_type", "attraction_type")

    return sent

def post_delex(sent):
    sent = sent.replace('the table will be reserved for [value_count] minutes', 'the table will be reserved for 15 minutes')
    sent = sent.replace(' [value_count] person ', ' [value_people] person ')
    sent = sent.replace(' [value_count] people ', ' [value_people] people ')
    sent = sent.replace(' [value_count] day ', ' [hotel_stay] day ')
    sent = sent.replace(' [value_count] days ', ' [hotel_stay] days ')
    sent = sent.replace(' [value_count] night ', ' [hotel_stay] night ')
    sent = sent.replace(' [value_count] nights ', ' [hotel_stay] nights ')
    sent = sent.replace(' [value_count],[value_count] ', ' [value_count] ')
    sent = sent.replace(' [hotel_stay] nights [hotel_stay] days ', ' [hotel_stay] days ')
    sent = sent.replace(' [hotel_stay] days [hotel_stay] nights ', ' [hotel_stay] days ')

    return sent

def create_delex_data_sys(turn, sent_act, bs, dic):
    # normalization, split and delexicalization of the sentence
    sent = delexicalise_using_da(turn)
    sent = sent.replace('I \'d ', 'i would ')
    sent = sent.replace(' i \'d ', ' i would ')
    sent = normalize(sent)
    words = sent.split()
    sent = delexicalise(' '.join(words), dic)
    # parsing reference number GIVEN belief state
    sent = delexicalise_reference_number(sent, turn)
    # changes to numbers only here
    digitpat = re.compile(" \d+ ")
    sent = re.sub(digitpat, ' [value_count] ', sent)
    sent = sent.replace(' # ', ' ')
    sent = fix_delex(sent, sent_act, None)
    sent = post_delex(sent)

    return sent.strip()

def create_delex_data_usr(turn, sent_act, bs, dic):
    # normalization, split and delexicalization of the sentence
    sent = sent.replace('I \'d ', 'i would ')
    sent = sent.replace(' i \'d ', ' i would ')
    sent = normalize(sent)
    words = sent.split()
    sent = delexicalise(' '.join(words), dic)
    # parsing reference number GIVEN belief state
    sent = delexicalise_reference_number(sent, turn)
    # changes to numbers only here
    digitpat = re.compile(" \d+ ")
    sent = re.sub(digitpat, ' [value_count] ', sent)
    sent = sent.replace(' # ', ' ')
    sent = fix_delex(sent, sent_act, bs)

    return sent.strip()

def lower_dict(dictionary):
    new_dictionary = {}
    for k in dictionary:
        for key, val in dictionary[k]:
            if key != "none":
                if k.lower().split('-')[0] in domains:
                    new_dictionary["domain-{}-{}".format(k.lower(), key.lower())] = val.lower().strip()
                else:
                    new_dictionary["{}-{}".format(k.lower(), key.lower())] = val.lower().strip()
            else:
                if k.lower().split('-')[0] in domains:
                    new_dictionary["domain-{}".format(k.lower())] = val.lower().strip()
                else:
                    new_dictionary["{}".format(k.lower())] = val.lower().strip()
    return new_dictionary


if __name__ == '__main__':
    state = {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "1",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "abbey pool",
                "area": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "barbeque>modern european",
                "price range": "",
                "name": "",
                "area": "",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": []
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "departure": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": "3"
            },
            "semi": {
                "leaveAt": "12:00",
                "destination": "kings lynn",
                "day": "sunday",
                "arriveBy": "",
                "departure": "cambridge"
            }
        }
    }

    db_feature, _, kb_results = make_db_feature(state, mode='predict')
    print(len(db_feature))
    db_feature = db_feature.tolist()
    print(db_feature)