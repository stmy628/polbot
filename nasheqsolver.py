import numpy as np
import pygambit as gbt

class GTSolver: 
    def __init__(self, players, actions, payoffs, repeated=False, discount_factor=0.9):
        self.players = players
        self.actions = actions
        self.payoffs = payoffs
        self.repeated = repeated
        self.discount_factor = discount_factor
        self.nash_eq = None
        self.nash_eq_payoffs = None
        self.num_players = len(players)
        self.build_matrices()

    # Matrix builder
    def build_matrices(self):
        action_sizes = [len(self.actions[player]) for player in self.players]

        # Create empty matrices for each player's payoffs
        self.payoff_matrices = [np.zeros(action_sizes) for _ in range(self.num_players)]

        # Fill matrices with payoffs
        for action_tuple, payoffs in self.payoffs.items():
            indices = tuple(self.actions[player].index(action) for player, action in zip(self.players, action_tuple))
            for i in range(self.num_players):
                self.payoff_matrices[i][indices] = payoffs[i]

    def nash_eq_solver(self):
        game = gbt.Game.new_table(self.num_players, *[len(self.actions[player]) for player in self.players])

        # Fill the game with payoffs
        for action_tuple, payoffs in self.payoffs.items():
            indices = tuple(self.actions[player].index(action) for player, action in zip(self.players, action_tuple))
            for i in range(self.num_players):
                game[i][indices].payoff = payoffs[i]

        solver = gbt.nash.ExternalEnumMixedSolver()
        self.nash_eq = solver.solve(game)
        return self.nash_eq
    
    def payoff_repeated(self):
        if self.repeated:
            return [(1 / (1 - self.discount_factor)) * p for p in self.payoffs]
        return self.payoffs

    def recommend_action(self):
        equilibria = self.nash_eq_solver()
        if not equilibria:
            return "No Nash Equilibrium found."

        best_eq = next(equilibria)  # First Nash Equilibrium
        suggested_actions = {self.players[i]: self.actions[self.players[i]][np.argmax(best_eq[i])] for i in range(self.num_players)}

        return suggested_actions
    
if __name__ == "__main__":
    players = ["Offender A", "Offender B", "Police"]
    actions = {
        "Offender A": ["Fight", "De-escalate"],
        "Offender B": ["Fight", "De-escalate"],
        "Police": ["Arrest", "Warn"]
    }
    
    # Payoff matrix format: (ActionA, ActionB, ActionC): (PayoffA, PayoffB, PayoffC)
    payoff_matrix = {
        ("Fight", "Fight", "Arrest"): (-10, -10, 5),
        ("Fight", "Fight", "Warn"): (-5, -5, 3),
        ("Fight", "De-escalate", "Arrest"): (3, -5, 4),
        ("Fight", "De-escalate", "Warn"): (1, 0, 2),
        ("De-escalate", "Fight", "Arrest"): (-5, 3, 4),
        ("De-escalate", "Fight", "Warn"): (0, 1, 2),
        ("De-escalate", "De-escalate", "Arrest"): (5, 5, 2),
        ("De-escalate", "De-escalate", "Warn"): (6, 6, 1)
    }

    game_solver = GTSolver(players, actions, payoff_matrix, discount_factor=0.9, repeated=True)
    print("Nash Equilibria:", list(game_solver.nash_eq_solver()))
    print("Recommended Actions:", game_solver.recommend_action())