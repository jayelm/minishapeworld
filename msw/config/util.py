def a_or_an(s):
    """
    Return `a` or `an` depending on the first voewl sound of s. Dumb heuristic
    """
    if s[0].lower() in 'aeiou':
        return 'an'
    return 'a'
