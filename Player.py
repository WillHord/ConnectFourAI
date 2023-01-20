import numpy as np

class Node:
    def __init__(self, player_number, move, board):
        self.player_number = player_number
        self.board = board
        self.move = move
        self.children = []
        self.value = None
        
    def add_child(self, child):
        self.children.append(child)
        
    def get_children(self):
        return self.children
    
    def remove_child(self, child):
        self.children.remove(child)

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.depth = 4
        
    def check_arr(self, row, player_number):
        def maxSequence():
            maxSequence = 0
            sequence = 0
            for i in range(len(row)):
                if row[i] != player_number:
                        sequence = 0
                else:
                    sequence += 1
                    if sequence > maxSequence:
                        maxSequence = sequence
            return maxSequence

        def emptySequence():
            maxSequence = 0
            sequence = 0
            for i in range(len(row)):
                if row[i] == player_number:
                    sequence += 1
                elif row[i] == 0 and (i == 0 or row[i-1] == player_number):
                    sequence += 1
                elif row[i] == 0 and (i == len(row)-1 or row[i+1] == player_number):
                    sequence += 1
                else:
                    sequence = 0
                if sequence > maxSequence:
                    maxSequence = sequence
            return maxSequence
        return maxSequence() + (emptySequence() * 0.6)
    
    def check_horizontal(self, board, player_number):
        return max([self.check_arr(row,player_number) for row in board])
    
    def check_vertical(self, board, player_number):
        return self.check_horizontal(board.T, player_number)
    
    def check_diagonal(self, board, player_number):
        maxSequence = 0
        for op in [None, np.fliplr]:
            op_board = op(board) if op else board
            
            root_diag = np.diagonal(op_board, offset=0).astype(np.int)
            maxSequence = max(maxSequence, self.check_arr(root_diag, player_number))

            for i in range(1, board.shape[1]-3):
                for offset in [i, -i]:
                    diag = np.diagonal(op_board, offset=offset)
                    maxSequence = max(maxSequence, self.check_arr(diag.astype(np.int), player_number))

        return maxSequence
    
    def check_win(self, board, player_number):
        if max(self.check_diagonal(board, player_number),
               self.check_vertical(board, player_number),
               self.check_horizontal(board, player_number)) >= 4:
            return True
        
        
    def get_possible_moves(self, board):
        possible_moves = []
        for col in range(7):
            for row in range(5,-1,-1):
                if board[row][col] == 0:
                    possible_moves.append((col, np.copy(board)))
                    possible_moves[-1][1][row][col] = self.player_number
                    break
        return possible_moves
    
    def close_to_center(self, board, player):
        return (np.count_nonzero(board == player) - np.count_nonzero(np.transpose(board)[4] == player)) * -0.1
        
        
    def make_tree(self, currentPlayer, depth, board, move = -1):
        root = Node(currentPlayer, move, board)
        
        if depth > 0:
            if currentPlayer == 0:
                currentPlayer = self.player_number
            elif currentPlayer == 1:
                currentPlayer = 2
            else:
                currentPlayer = 1
            for move in self.get_possible_moves(board):
                child = self.make_tree(currentPlayer, depth-1, move[1], move[0])
                root.add_child(child)
        return root
        

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """        
        tree = self.make_tree(0, self.depth, board)
        
        moves = [None for i in range(self.depth)]
        def alpha_beta_helper(node: Node, depth: int, alpha: int, beta: int, player: int):
            if depth == 0 or node.children == []:
                return self.evaluation_function(node.board, player)
            if player == self.player_number:
                value = -1000000000
                for i in node.children:
                    player = 2 if self.player_number == 1 else 1
                    tempValue = alpha_beta_helper(i, depth-1, alpha, beta, player)
                    if value < tempValue:
                        value = tempValue
                        moves[depth - 1] = i.move
                    if value >= beta:
                        break
                    alpha = max(alpha, value)
                return value
            else:
                value = 1000000000
                for i in node.children:
                    player = self.player_number
                    tempValue = alpha_beta_helper(i, depth-1, alpha, beta, player)
                    if value > tempValue:
                        value = tempValue
                        moves[depth -1] = i.move
                    if value <= alpha:
                        break
                    beta = min(beta, value)
                return value
        best_score = alpha_beta_helper(tree, self.depth, -1000000000, 1000000000, self.player_number)
        
        return moves[-1]

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        moves = [None for i in range(self.depth)]
        tree = self.make_tree(0, self.depth, board)
        def expectimax_helper(node, depth, player):
            if depth == 0 or node.children == []:
                return self.evaluation_function(node.board, self.player_number)
            elif player == self.player_number:
                value = -1000000000
                for i in node.children:
                    tempValue = expectimax_helper(i, depth-1, 2 if self.player_number == 1 else 1)
                    if tempValue > value:
                        value = tempValue
                        moves[depth - 1] = i.move
            else:
                value = 0
                for i in node.children:
                    value += ((1 / len(node.children)) * expectimax_helper(i, depth-1, self.player_number))
            return value

        best_score = expectimax_helper(tree, self.depth, self.player_number)
        # print(moves)
        return moves[-1]



    def evaluation_function(self, board, player):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        
        player1_diag = self.check_diagonal(board, player)
        player1_horiz = self.check_horizontal(board, player)
        player1_vert = self.check_vertical(board, player)
        
        player1Max = max(player1_diag, player1_horiz, player1_vert)
        
        player2 = 2 if player == 1 else 1
        player2Max = max(self.check_diagonal(board, player2),
                         self.check_vertical(board, player2),
                         self.check_horizontal(board, player2))
        
        return player1Max - (2 * player2Max) #+ self.close_to_center(board, player)
         


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

