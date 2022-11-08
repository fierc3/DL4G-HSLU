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
        card_list = self.get_valid_cards_int(obs)
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
