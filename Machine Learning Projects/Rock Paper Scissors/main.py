#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This entrypoint file to be used in development. Start by reading README.md
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player
from unittest import main

play(player, quincy, 1000)
play(player, abbey, 1000)
play(player, kris, 1000)
play(player, mrugesh, 1000)

play(player, quincy, 1000)

play(player, abbey, 1000)

play(player, kris, 1000)

play(player, mrugesh, 1000)

#main(module='test_module', exit=False)

