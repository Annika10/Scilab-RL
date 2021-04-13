from enum import Enum


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """

    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # randomly after the current step, in the same episode, but samples are drawn more towards the end of the episode
    RNDEND = 3

    FUTURE2 = 4

    RNDEND2 = 5

    FUTURE3 = 6

# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future": GoalSelectionStrategy.FUTURE,
    "final": GoalSelectionStrategy.FINAL,
    "episode": GoalSelectionStrategy.EPISODE,
    "rndend": GoalSelectionStrategy.RNDEND,
    "future2": GoalSelectionStrategy.FUTURE2,
    "rndend2": GoalSelectionStrategy.RNDEND2,
    "future3": GoalSelectionStrategy.FUTURE3,
}
