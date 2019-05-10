
# taken from https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py
def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = '<P>'
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_) and prev_type != '<P>':
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_: #and type_ != '<P>':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_: #and type_ != '<P>':
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, score



# taken from https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py
def get_entities_p(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = 'O'
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]
            
        if type_ == '<P>':
            if prev_type != 'O': type_ = prev_type

        if end_of_chunk_p(prev_tag, tag, prev_type, type_) and prev_type != '<P>':
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk_p(prev_tag, tag, prev_type, type_):
            begin_offset = i
        if type_ != '<P>' or prev_tag == 'O':
            prev_type = type_ 
        #elif prev_type == 'O':
        #    prev_type = type_
        prev_tag = tag
        #prev_type = type_
        

    return chunks


def end_of_chunk_p(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    #if prev_tag == 'E': chunk_end = True
    #if prev_tag == 'S': chunk_end = True

    #if prev_tag == 'B' and tag == 'B': chunk_end = True
    #if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    #if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_: #and type_ != '<P>':
        chunk_end = True

    return chunk_end


def start_of_chunk_p(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    #if tag == 'S': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I' and type_ != '<P>': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_ and type_ != '<P>':
        chunk_start = True

    return chunk_start

def pair(*arg):
    l = len(arg) 
    pairs= []
    for i in range(l):
        for j in range(l-i-1):
            pairs.append([arg[i], arg[i+j+1]])
    return pairs

def sc(p, c):
    if c != 1:
        ct = c -3 if c > 4 else c
    else:
        ct = p
    return (c > 1 and c < 5) or (ct != p and ct > 0) # (c is 2, 3, 4, B-*, must new start) or c_type != prev type, and ct is not 'o'
def ec(p, c):
    if c != 1:
        ct = c -3 if c > 4 else c
    else:
        ct = p
    if p > 1: return (c >1 and c < 5) or (ct != p) # pre is type (c is 2, 3, 4, B-*,) or ctype != ptype
def get_e(label):
    p, c = 0,0
    s = 0
    chk = []
    for i, c in enumerate(label):
        if ec(p, c): chk.append((s,i))
        if sc(p, c): s = i
        if c != 1: p = c -3 if c > 4 else c
    if ec(p, 0): chk.append((s, i+1)) # count final chunk
    return chk