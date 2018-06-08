
class FilterDatum(object):
    """Represents a filter for training"""
    def __init__(self, filter_np, tp_score):
        self.filter_np = filter_np
        self.tp_score = tp_score
