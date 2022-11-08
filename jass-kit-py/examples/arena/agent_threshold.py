import logging
from jass.game.const import *
from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.agents.agent import Agent
from jass.game.rule_schieber import RuleSchieber
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 # import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold
from pathlib import Path

import os



# score if the color is trump
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0,]
# score if uneufe is selected (all colors)
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

def calculate_trump_selection_score(cards, trump: int) -> int:
    result = 0
    for card in cards:
        color = color_of_card[card]
        offset = offset_of_card[card]
        if color == trump:
            result += trump_score[offset]
        elif trump == OBE_ABE:
            result += obenabe_score[offset]
        elif trump == UNE_UFE:
            result += uneufe_score[offset]
        else:
            result += no_trump_score[offset]
    return result

class AgentThreshold(Agent):
    """
    Agent to act as a player in a game of jass.
    """

    counter = 0

    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

    def partner_played_cards(self, obs: GameObservation):
        all_tricks = obs.tricks
        partner_id = (obs.player + 2)%4
        return [trick[partner_id] for trick in all_tricks]

    def get_valid_cards_int(self, obs: GameObservation):
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards)
        return card_list

    def action_trump(self, obs: GameObservation) -> int:
            """
            Determine trump action for the given observation
            Args:
                obs: the game observation, it must be in a state for trump selection

            Returns:
                selected trump as encoded in jass.game.const or jass.game.const.PUSH
            """

            valid_cards = self._rule.get_valid_cards_from_obs(obs)


            card_list = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards)
            best_score = 0
            best_color = -1
            for color in trump_ints:
                score = calculate_trump_selection_score(card_list, color)
                if best_score < score:
                    best_score = score
                    best_color = color

            if best_score > 77:
                return best_color

            path_to_data = Path('.')
            #print(path_to_data)

            cards = [
            # Diamonds
            'DA','DK','DQ','DJ','D10','D9','D8','D7','D6',
            # Hearts
            'HA','HK','HQ','HJ','H10','H9','H8','H7','H6',
            # Spades
            'SA','SK','SQ','SJ','S10','S9','S8','S7','S6',
            # Clubs
            'CA','CK','CQ','CJ','C10','C9','C8','C7','C6'
            ]
            forehand = ['FH']

            feature_columns = cards + forehand

            a = np.append(valid_cards,[obs.forehand+1])
        
            df = pd.DataFrame(a.reshape(-1, len(a)),columns=feature_columns)
            for color in 'DHSC':
            
                # Jack and nine combination
                new_col = '{}_J9'.format(color)
                df[new_col]  = df['{}J'.format(color)] & df['{}9'.format(color)]
                feature_columns.append(new_col)
                
                # Exercise: Add other features here such as the combination of Ace-King-Queen (Dreiblatt).
                # todo

                akq_col = '{}_AKQ'.format(color)
                df[akq_col]  = df['{}A'.format(color)] & df['{}K'.format(color)] & df['{}Q'.format(color)]
                feature_columns.append(akq_col)

            # logisticSelector = pickle.load(open("D:/hslul/DL4G/DL4G-HSLU/jass-kit-py/examples/arena/rf", 'rb'))
            # logisticResult = logisticSelector.predict(df)

            # treeSelector = pickle.load(open("D:/hslul/DL4G/DL4G-HSLU/jass-kit-py/examples/arena/dt", 'rb'))
            # treeResult = treeSelector.predict(df)

            # knnSelector = pickle.load(open("D:/hslul/DL4G/DL4G-HSLU/jass-kit-py/examples/arena/knn", 'rb'))
            # knnResult = knnSelector.predict(df)

            script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
            rel_path = "svm"
            abs_file_path = os.path.join(script_dir, rel_path)

            svcSelector = pickle.load(open(abs_file_path, 'rb'))
            svcResult = svcSelector.predict(df)


            # print("prediciton time")
            # print(f"logistic: {logisticResult} tree: {treeResult} knn:{knnResult} svc:{svcResult}")


            helper = {0: 'DIAMONDS', 1: 'HEARTS', 2: 'SPADES', 3:'CLUBS',
                                    4: 'OBE_ABE', 5: 'UNE_UFE', 10: 'PUSH'}
            trumpResultId = (list(helper.keys())[list(helper.values()).index(svcResult[0])])

            if obs.forehand == -1 and trumpResultId == 10:
                return best_color

            # if obs.forehand == -1 and trumpResultId == 10:
            #     #print("'I must deicde and can't schiebe")
            #     card_list = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards)
            #     best_score = 0
            #     best_color = -1
            #     for color in trump_ints:
            #         score = calculate_trump_selection_score(card_list, color)
            #         if best_score < score:
            #             best_score = score
            #             best_color = color
            #     return best_color


            return trumpResultId

            # card_list = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards)
            # best_score = 0
            # best_color = -1
            # for color in trump_ints:
            #     score = calculate_trump_selection_score(card_list, color)
            #     if best_score < score:
            #         best_score = score
            #         best_color = color

            # if best_score < 68 and obs.forehand != -1:
            #     return PUSH
            # return best_color

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        self.counter += 1

        all_played_cards = np.ravel(obs.tricks)
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        card_list = self.get_valid_cards_int(obs)

        def cards_of_color(color):
            start = color*9
            end = color*9+9
            return list(range(start, end))

        def bock_of_color(color):
            # What would be the bock?
            for card in reversed(cards_of_color(color)):
                if card not in all_played_cards:
                    return card

        def got_bock_of_color(color):
            return bock_of_color(color) in card_list

        trump_cards = cards_of_color(obs.trump)
        current_color = color_of_card[obs.current_trick[0]]

        partner_cards = self.partner_played_cards(obs=obs)
        partner_current = partner_cards[obs.nr_tricks]
        #print(partner_current)

        if partner_current != -1:
            # partner has played

            partner_strong_card = False
            color_of_partner_card = color_of_card[partner_current]
            #print(color_of_partner_card)
            if obs.trump == UNE_UFE:
                #print("uneufe")
                x = partner_current / (color_of_partner_card + 1);
                if (x > D8 and got_bock_of_color(current_color)) or x > D7:
                    partner_strong_card = True
                    for card in card_list:
                        #print(card)
                        c_order = card / (color_of_partner_card + 1)
                        #print(card / (color_of_partner_card + 1))
                        if c_order == DQ or c_order == DK:
                            return card # get rid of mediocre cards in this round
            if obs.trump == OBE_ABE:
                #print("OBE_ABE")
                x = partner_current / (color_of_partner_card + 1);
                if (x < DJ and got_bock_of_color(current_color)) or x < DQ:
                    partner_strong_card = True
                    for card in card_list:
                        #print(card)
                        c_order = card / (color_of_partner_card + 1)
                        #print(card / (color_of_partner_card + 1))
                        if c_order == D6 or c_order == D7:
                            return card # get rid of mediocre cards in this round
            if partner_strong_card:
                for card in card_list:
                    #print(card)
                    c_order = card / (color_of_partner_card + 1)
                    #print(card / (color_of_partner_card + 1))
                    if c_order == D9 or c_order == D8:
                        return card # get rid of mediocre cards in this round
                    
            

        
        
        # Wenn ich de Trumpf agseid han, ich als ersts dra ben
        # ond de Buur devo han, spill ich de
        if (
            obs.player == obs.declared_trump # Ich ha de Trumpf agseid
            and obs.trump < 4 # Nome Farbe
            and obs.trump*9+3 in card_list # Ich ha de Buur
        ):
            # print(f'+{self.counter} {obs.trump*9+3}')
            return(obs.trump*9+3)

        # Wenn ich weiss, dasses kei Trümpf me hed ond ich ha en Bock, denn
        # spiel ich de. 
        

        # Alli Trümpf sind gspillt
        elif all(card in all_played_cards for card in trump_cards):
            # Wenn ich als ersts spile ond vo ergendere Farb en Bock ha
            if obs.current_trick[0] == -1:
                for color in list(range(6)):
                    if got_bock_of_color(color):
                        # Denn spill ich de Bock
                        # print(f'+{self.counter} {bock_of_color(color)}')
                        return bock_of_color(color)
            # Wenn ich vo de aktuelle Farb en Bock han
            if got_bock_of_color(current_color):
                # Denn spill ich de Bock
                # print(f'+{self.counter} {bock_of_color(current_color)}')
                return bock_of_color(current_color)

        # (för spöter) Wenn ich kei Bock me han, spil ich das, wo mer min
        # Partner (met Jassmagie) azeigt hed. 

        # we use the global random number generator here
        # print(f'-{self.counter} {np.random.choice(np.flatnonzero(valid_cards))}')
        return np.random.choice(np.flatnonzero(valid_cards))
