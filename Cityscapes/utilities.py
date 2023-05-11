def multi_mean_tuple(lst):
    sglob, lglob = 0, 0
    for slst in lst:
        sglob += sum(slst)
        lglob += len(slst)
    return sglob / lglob