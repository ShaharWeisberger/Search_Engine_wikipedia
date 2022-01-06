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
    id = int(round(id_num/683803))
    return id


# def write_as_Counter(id_val , base_dir='' , name ):
#     our_current_index = index_hash(id_val[0][0])


