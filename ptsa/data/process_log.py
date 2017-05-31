
import numpy as np
import numpy.core.numerictypes as nt
_typestr = nt._typestr

import yaml

defaults = {str:'',
            np.string_:'',
            np.float64:np.float64(0),
            np.int64:np.int64(0),
            int:int(0),
            int:int(0),
            float:float(0.0),
            bool:False,
            type(None):None}

def _addrow(i,d,cols,tofill,to_ignore=[],prefix=''):
    for k in d:
           # add prefix
        key = prefix+k
        if key in to_ignore:
            # skip it
            continue

        # see if dict
        if isinstance(d[k],dict):
            _addrow(i,d[k],cols,tofill,to_ignore=to_ignore,prefix=key+'_')
            continue

        # see if tuple
        if isinstance(d[k],tuple):
            # turn into indexed dict
            tdict = {}
            for j in range(len(d[k])):
                tdict[str(j)] = d[k][j]
            _addrow(i,tdict,cols,tofill,to_ignore=to_ignore,prefix=key+'_')
            continue

        # see if already there
        if not key in cols:
            # add and fill it with default vals
            cols[key] = [defaults[type(d[k])]]*i

        # append it
        cols[key].append(d[k])

        # remove from list to fill
        if key in tofill:
            del tofill[key]

def load_yaml(yaml_file, **append_cols):
    # load the dictlist
    dictlist = yaml.load(open(yaml_file,'r'))
    for i in range(len(dictlist)):
        dictlist[i].update(append_cols)
    return dictlist

def log2rec(dictlist, to_ignore=[], force_format = None, **append_cols):
    """
    Convert a list of dicts (possibly in a yaml file) to a recarray.

    You can use the append_cols kwargs to add fields with the value
    replicated across all rows.  This is a good way to add subject ids
    and personality variables.
    """

    # see if dictlist is a yaml file
    if isinstance(dictlist,str):
        # assume it's a file and read it in
        dictlist = yaml.load(open(dictlist,'r'))

    # make a dict of columns
    cols = {}

    for i,d in enumerate(dictlist):
        # get list to fill
        tofill = dict([(k,defaults[type(cols[k][-1])]) for k in list(cols.keys())])

        # loop over keys and fill cols
        _addrow(i,d,cols,tofill, to_ignore=to_ignore)

        # anything not filled add the default val
        for k in tofill:
            cols[k].append(tofill[k])

    # add in the additional columns
    for k in append_cols:
        cols[k] = [append_cols[k]]*len(cols[list(cols.keys())[0]])

    # get the formats
    formats = {}
    for k in list(cols.keys()):
        obj = np.asarray(cols[k])
        if not isinstance(obj, np.ndarray):
            raise ValueError("item in the array list must be an ndarray.")
        oformat = _typestr[obj.dtype.type]
        if issubclass(obj.dtype.type, nt.flexible):
            oformat += repr(obj.itemsize)
        formats[k] = oformat

    # enforce formats
    if isinstance(force_format,dict):
        for k in force_format:
            if k in formats:
                formats[k] = force_format[k]

    # make the rec from arrays
    rec = np.rec.fromarrays([cols[k] for k in list(cols.keys())],
                            names = ','.join(list(cols.keys())),
                            formats = ','.join([formats[k] for k in list(cols.keys())]))
    return rec
