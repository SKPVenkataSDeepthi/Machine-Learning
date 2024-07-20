#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# RPS.py

def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    guess = "R"
    
    if len(opponent_history) > 1:
        # Pattern recognition strategy
        patterns = {
            "RR": "P", "RP": "S", "RS": "R",
            "PR": "S", "PP": "R", "PS": "P",
            "SR": "R", "SP": "P", "SS": "S"
        }
        last_move = opponent_history[-1]
        second_last_move = opponent_history[-2]
        pattern = second_last_move + last_move

        if pattern in patterns:
            guess = patterns[pattern]
        else:
            guess = "R"

    if len(opponent_history) > 3:
        # Frequency analysis strategy
        from collections import Counter
        last_three = opponent_history[-3:]
        counter = Counter(last_three)
        most_common = counter.most_common(1)[0][0]

        if most_common == "R":
            guess = "P"
        elif most_common == "P":
            guess = "S"
        elif most_common == "S":
            guess = "R"
    
    return guess

