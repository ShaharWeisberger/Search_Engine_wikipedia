import math
from pathlib import Path
import pickle



'''
this class is an hash class that will help us to separate all our data
into many sorted pkl for fast data access.
'''

def index_hash(id_num):
    '''
    calculating the bin we want to find
    '''
    # id = int(round(id_num/round(math.sqrt(6342910))))
    id = int(round(id_num/6838024))
    return id

#68380237
def write_as_Counter(id_val , base_dir , name):
    our_current_index = index_hash(id_val[0][0])
    dic = {}
    for id, val in id_val:
        if index_hash(id) > our_current_index:
            # we will dump the file into
            with open(Path(base_dir + name + str(our_current_index)) ,'wb') as f:
                pickle.dump(dic, f)

            our_current_index = index_hash(id)
            dic = {}
            dic[id] = val
        else:
            dic[id] = val
            save_id = id

    #save the last dic
    if(dic[save_id] != None):
        with open(Path(base_dir + name + str(our_current_index)), 'wb') as f:
            pickle.dump(dic, f)


def get_dic(path , name, id):
    index = str(index_hash(id))
    with open(Path(path + name + index), 'rb') as f:
        wid2pv = pickle.loads(f.read())
    return wid2pv

#############################################################################################