from jass.game.const import *
from jass.game.game_util import *
from jass.game.game_observation import GameObservation

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
        else:
            result += no_trump_score[offset]
    return result

class Agent:
    """
    Agent to act as a player in a game of jass.
    """

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

        if obs.forehand == -1 and best_score < 68:
            return PUSH
        return best_color
        

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # we use the global random number generator here
        return np.random.choice(np.flatnonzero(valid_cards))
