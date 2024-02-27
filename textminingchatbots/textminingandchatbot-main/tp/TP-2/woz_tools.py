import re
import json
import os

class MultiWoZTools:
    ''' The different tools to process dataset information
    '''
    @staticmethod
    def to_minutes(mydata):
        hour_re = re.compile('\s*([0-9]+?).*')
        complete_re = re.compile('\s*([0-9]+):([0-9]).*')
        add_hour = 0
        if 'pm' in mydata:
            add_hour += 12
        res = complete_re.findall(mydata)
        if(len(res) > 0):
            hour = int(res[0][0]) + add_hour
            minutes = int(res[0][1])
        else: 
            res2 = hour_re.findall(mydata)
            if(len(res2) > 0):

                hour = int(res2[0][0]) + add_hour
                minutes = 0
            else: 
                return None
        return hour * 60 + minutes

    @staticmethod
    def encode_dialogue_act(dialogue_act):
        act = dialogue_act['span_info']
        return '[dialogue_act]' + ','.join([st.split('-')[0].lower()+"_"+asn.split('_')[-1].lower()+" "+asv
                        for st, asn, asv in zip(act['act_type'], act['act_slot_name'], act['act_slot_value'])]) + '[end_dialogue_act]'

    @staticmethod
    def delexicalize_answer(utterance, dialogue_act):
        act = dialogue_act['span_info']
        act_names = ['['+st.split('-')[0].lower()+"_"+asn.split('_')[-1].lower()+']'
                        for st, asn in zip(act['act_type'], act['act_slot_name'])]
        str_res = ''
        prev = 0 
        for i, (s, e) in enumerate(zip(act['span_start'], act['span_end'])):
            str_res += utterance[prev:s] + act_names[i]
            if(i+1 >= len(act_names)):
                str_res += utterance[e:]
            prev = e
        return str_res
        
    @staticmethod
    def encode_frame_state(in_state):
        state = []
        for frame in in_state:
            intent = frame['active_intent']
            slots = []
            for name, value in zip(frame['slots_values']['slots_values_name'], frame['slots_values']['slots_values_list']):
                slots.append(f'[{name}:{value}]')
            state.append(f"[intent:{intent}]"+("".join(slots))+"[end_intent]")
        return '[state]'+"".join(state) + '[end_state]'

    @staticmethod
    def state_to_db(in_state):
        db_queries = []
        for frame in in_state:
            intent = frame['active_intent']
            domain = intent.split('_')[1]
            slots = []
            for name, value in zip(frame['slots_values']['slots_values_name'], frame['slots_values']['slots_values_list']):
                minutes = MultiWoZTools.to_minutes(value[0])
                if minutes is not None:
                    value = minutes
                    slots.append((name.split('-')[1], '<', [value+120]))
                    slots.append((name.split('-')[1], '>', [value-120]))
                else:
                    slots.append((name.split('-')[1], '=', value))
                
            db_queries.append((domain, slots))
        return db_queries

    @staticmethod
    def decode_frame_state(str_state):
        find_state_re = re.compile('\[state](.*?)\[end_state]')
        find_intent_re = re.compile('\[intent:(.+?)\](.+?)\[end_intent\]')
        find_slot_re = re.compile('\[(.+?):\[(.+?)\]')
        find_values = re.compile("\'(.+?)\'")
        my_state = find_state_re.findall(str_state)
        if(len(my_state) == 0):
            return
        res = []
        for domain_find in find_intent_re.findall(my_state[0]):
            active_intent = domain_find[0]
            str_slots = domain_find[1]
            slot_values = []
            slot_names  = []
            for slot_find in find_slot_re.findall(str_slots):
                slot_names.append(slot_find[0])
                slot_values.append(find_values.findall(slot_find[1]))
            res.append({'active_intent': active_intent, 'slots_values':{'slots_values_name': slot_names, 'slots_values_list':slot_values}})
        return res

class MultiWoZDatabase:
    def __init__(self, folder = 'data/db'):
        self.database = {}
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r') as db_file:
                name = filename.split('_')[0]
                self.database[name] = [{k.lower(): v for k, v in d.items()}for d in json.load(db_file) ]

    def search_(self, table_name, *kwargs):
        '''
            (column_name, {=, <, > }, [value1, value2]
        '''
        items = []
        for item in self.database[table_name]:
            local_selection = [True]
            for arg in kwargs :
                column, operator, values = arg[0], arg[1], arg[2]
                is_locally_selected = False
                if not(column in item):
                    is_locally_selected = True
                    continue
                for value in values:
                    try:
                        if value == 'dontcare':
                            is_locally_selected = True
                            continue
                        if operator == "=":
                            is_locally_selected = item[column] == value
                        if operator == ">":
                            minutes = MultiWoZTools.to_minutes(item[column])
                            if(minutes is not None and isinstance(value, int)):
                                is_locally_selected = minutes > value
                            else:
                                is_locally_selected = item[column] > value
                        if operator == "<":
                            minutes = MultiWoZTools.to_minutes(item[column])
                            if(minutes is not None and isinstance(value, int)):
                                is_locally_selected = minutes < value
                            else:
                                is_locally_selected = item[column] < value
                    except: 
                        pass
                local_selection.append(is_locally_selected)
            if all(local_selection):
                items.append(item)
        return len(items), items

    def search(self, query):
        res_db = []
        for table, constraint in query:
            res_db.append((table, self.search_(table, *constraint)))
        return res_db

    def table_def(self, table_name):
        return self.database[table_name][0]

    def encode_db_results(self, db_results):
        encoded_domain = []
        for domain, results in db_results:
            nb_item, results = results[0], results[1]
            #list_results = []
            # if len(results) > 2 :
            #    results = [results[0], results[-1]]
            
            # for result in results:
            #     entry = []
            #     for k, v in result.items():
            #         if k  == 'location' or k == 'introduction':
            #             continue
            #         if domain == 'hotel' and k == 'price':
            #             for room_type, room_price in  v.items():
            #                 entry.append(f"[{k}-{room_type}:{room_price}]")
            #         else:
            #             entry.append(f"[{k}:{v}]")
            #     rs =''.join(entry)
            #     list_results.append(f'[result_item]{rs}[end_result_item]')
            # ed = ''.join(list_results)
            encoded_domain.append(f'[result:{domain}:{nb_item}]')#{ed}[end_result:{domain}]')
        all = ''.join(encoded_domain)
        return f'[db]{all}[end_db]'