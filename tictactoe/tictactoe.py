"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = 0
    o = 0

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == X:
                x += 1
            elif board[i][j] == O:
                o += 1

    if x > o:
        return O
    else:
        return X

    raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                temporary = (i, j)
                state = tuple(temporary)
                moves.add(state)
    return moves

    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    nextboard = deepcopy(board)
    if nextboard[action[0]][action[1]] != EMPTY:
        raise ValueError
    nextboard[action[0]][action[1]] = player(board)
    return nextboard

    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[2][0] == board[1][1] == board[0][2] != EMPTY:
        return board[2][0]
    if board[0][0] == board[1][0] == board[2][0] != EMPTY:
        return board[0][0]
    if board[0][1] == board[1][1] == board[2][1] != EMPTY:
        return board[0][1]
    if board[0][2] == board[1][2] == board[2][2] != EMPTY:
        return board[0][2]
    if board[0][0] == board[0][1] == board[0][2] != EMPTY:
        return board[0][0]
    if board[1][0] == board[1][1] == board[1][2] != EMPTY:
        return board[1][0]
    if board[2][0] == board[2][1] == board[2][2] != EMPTY:
        return board[2][0]
    return

    raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is None:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == EMPTY:
                    return False
    return True

    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

    raise NotImplementedError


def inner(board):
    chosen = ()
    if player(board) == O:
        value = math.inf
    else:
        value = -math.inf
    for choice in actions(board):
        current = result(board, choice)
        if terminal(current) is not True:
            temp = inner(current)
            if player(board) == X:
                if temp[1] > value:
                    value = temp[1]
                    chosen = choice
            else:
                if temp[1] < value:
                    value = temp[1]
                    chosen = choice
        if terminal(current) is True:
            if player(board) == X:
                if utility(current) > value:
                    value = utility(current)
                    chosen = choice
            else:
                if utility(current) < value:
                    value = utility(current)
                    chosen = choice
    return chosen, value


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    return inner(board)[0]

    raise NotImplementedError
