import pygame, sys, random
from pygame.locals import *

# Create the constants (go ahead and experiment with different values)
BOARDSIZE = 4
TILESIZE = 80
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
FPS = 45
#                 R    G    B
BLACK =         (  0,   0,   0)
WHITE =         (255, 255, 255)
GREEN =    (  0,  128, 0 )
DARKTURQUOISE = (  3,  54,  73)
BLUE =         (  25, 25,  112)

BGCOLOR = DARKTURQUOISE
TILECOLOR = BLUE
TEXTCOLOR = WHITE
BORDERCOLOR = GREEN
BASICFONTSIZE = 20

BUTTONCOLOR = WHITE
BUTTONTEXTCOLOR = BLACK
MESSAGECOLOR = WHITE

XMARGIN = int((WINDOWWIDTH - (TILESIZE * BOARDSIZE + (BOARDSIZE - 1))) / 2)
YMARGIN = int((WINDOWHEIGHT - (TILESIZE * BOARDSIZE + (BOARDSIZE - 1))) / 2)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

def main():
    global FPSCLOCK, DISPLAYSURF

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    pygame.display.set_caption('Slide Puzzle')

    DISPLAYSURF.fill(BGCOLOR)
    menu = Menu()
    run_game(menu)

def run_game(menu):
    board = Board()
    board.shuffle(80, menu)

    while True:
        if board.is_solved():
            menu.show_msg('solve')
        else:
            menu.show_msg('running')

        check_for_quit()
        for e in pygame.event.get():
            if e.type == MOUSEBUTTONUP:
                tile = board.get_tile(e.pos)
                if tile:
                    board.find_move(tile)
                else:
                    menu.check_rect(e.pos, board)
            elif e.type == KEYUP:
                    board.check_key(e.key)

        pygame.display.update()
        FPSCLOCK.tick(FPS)

def terminate():
    pygame.quit()
    sys.exit()

def check_for_quit():
    for event in pygame.event.get(QUIT): # get all the QUIT events
        terminate() # terminate if any QUIT events are present
    for event in pygame.event.get(KEYUP): # get all the KEYUP events
        if event.key == K_ESCAPE:
            terminate() # terminate if the KEYUP event was for the Esc key
        pygame.event.post(event) # put the other KEYUP event objects back

class Tile(object):
    def __init__(self, id, coord):
        self.id = id
        self.coord = coord

    def move(self, direction, speed):
        x, y = self.coord
        for s in range(0, TILESIZE, speed):
            self.move_tile(x, y, s, direction)
            pygame.display.update()
            FPSCLOCK.tick(FPS)
        self.move_tile(x, y, TILESIZE + 1, direction)

    def move_tile(self, x, y, distance, direction):
        dx, dy = direction
        self.draw_tile(x + dx, y + dy, BGCOLOR)
        self.coord = (x + distance * dx, y + distance * dy)
        self.draw()

    def draw(self):
        self.draw_tile(self.coord[0], self.coord[1], TILECOLOR)
        self.draw_text()

    def draw_tile(self, x, y, color):
        pygame.draw.rect(DISPLAYSURF, color, (x, y, TILESIZE, TILESIZE))

    def draw_text(self):
        font = pygame.font.Font('freesansbold.ttf', 35)
        surf = font.render(str(self.id), True, TEXTCOLOR)
        rect = surf.get_rect()
        x, y = self.coord
        rect.center = (x + TILESIZE // 2, y + TILESIZE // 2)
        DISPLAYSURF.blit(surf, rect)

    def is_me(self, pos):
        x, y = self.coord
        rect = pygame.Rect(x, y, TILESIZE, TILESIZE)
        return rect.collidepoint(pos)

class Board(object):
    def __init__(self):
        self.tiles = []
        self.game_moves = []
        self.shuffle_moves = []
        self.make_tiles()
        self.board = self.make_board()
        self.solved = self.make_board()
        self.draw_board()
        self.blank = self.board[BOARDSIZE - 1][BOARDSIZE - 1]

    def is_solved(self):
        return self.board == self.solved

    def make_tiles(self):
        for i in range(BOARDSIZE * BOARDSIZE - 1):
            x = XMARGIN + (i % BOARDSIZE) * (TILESIZE + 1)
            y = YMARGIN + (i // BOARDSIZE) * (TILESIZE + 1)
            coord = (x, y)
            self.tiles.append(Tile(i + 1, coord))
        coord = (x + TILESIZE + 1, y)
        self.tiles.append(Tile(0, coord))

    def make_board(self):
        board = []
        for y in range(BOARDSIZE):
            row = []
            for x in range(BOARDSIZE):
                tile = self.tiles[y * BOARDSIZE + x]
                row.append(tile)
            board.append(row)
        return board
 
    def draw_board(self):
        for boards in self.board:
            for tile in boards:
                if tile.id:
                    tile.draw()
        self.draw_border()
        pygame.display.update()
        pygame.time.wait(500)

    def draw_border(self):
        x = XMARGIN - 5
        y = YMARGIN - 5
        size = TILESIZE * BOARDSIZE + 11
        pygame.draw.rect(DISPLAYSURF, BORDERCOLOR, (x, y, size, size), 4)

    def get_tile(self, pos):
        for boards in self.board:
            for tile in boards:
                if tile.is_me(pos):
                    return tile
        return None

    def get_board_index(self, coord):
        x, y = coord
        x = (x - XMARGIN) // (TILESIZE + 1)
        y = (y - YMARGIN) // (TILESIZE + 1)

        return (x, y)

    def is_tile(self, tile, direction):
        x, y = self.get_board_index(tile.coord)
        x += direction[0]
        y += direction[1]

        return ((x >= 0 and x < BOARDSIZE) and (y >= 0 and y < BOARDSIZE))

    def get_movable_tile(self, direction):
        x, y = self.get_board_index(self.blank.coord)
        x += direction[0]
        y += direction[1]

        return self.board[y][x]

    def get_valid_move(self, tile):
        moves = [UP, DOWN, LEFT, RIGHT]

        for i in range(len(moves) - 1, -1, -1):
            if not self.is_tile(tile, moves[i]):
                moves.remove(moves[i])

        return moves

    def get_random_move(self, last_move, tile):
        moves = self.get_valid_move(tile)
        if last_move:
            moves.remove(last_move)

        return random.choice(moves)

    def move_tile(self, tile, direction, speed):
        x, y = self.get_board_index(tile.coord)
        dx, dy = direction
        coord = tile.coord
        tile.move(direction, speed)
        self.board[y][x], self.board[y + dy][x + dx] = self.board[y + dy][x + dx], self.board[y][x]
        self.blank = self.board[y][x]
        self.blank.coord = coord

    def shuffle(self, nums, msg):
        msg.show_msg('shuffle')
        last_move = None
        for i in range(nums):
            check_for_quit()
            valid_move = self.get_random_move(last_move, self.blank)
            tile = self.get_movable_tile(valid_move)
            last_move = (valid_move[0] * -1, valid_move[1] * -1)
            self.call_move(tile, last_move, self.shuffle_moves, TILESIZE // 3)

    def find_move(self, tile):
        moves = self.get_valid_move(tile)
        x, y = self.get_board_index(tile.coord)
        for move in moves:
            if self.board[y + move[1]][x + move[0]] == self.blank:
                self.call_move(self.board[y][x], move, self.game_moves, TILESIZE // 5)

    def check_key(self, key):
        event = {K_UP: UP, K_DOWN: DOWN, K_LEFT: LEFT, K_RIGHT: RIGHT, K_w: UP, K_s: DOWN, K_a: LEFT, K_d: RIGHT}
        if key in event.keys():
            move = event[key]
            r_move = (move[0] * -1, move[1] * -1)
            if self.is_tile(self.blank, r_move):
                tile = self.get_movable_tile(r_move)
                self.call_move(tile, move, self.game_moves, TILESIZE // 5)

    def call_move(self, tile, move, move_list, speed):
        move_list.append(move)
        self.move_tile(tile, move, speed)

    def reset_board(self):
        self.game_moves.reverse()

        for move in self.game_moves:
            check_for_quit()
            tile = self.get_movable_tile(move)
            r_move = (move[0] * -1, move[1] * -1)
            self.move_tile(tile, r_move, TILESIZE // 2)
        self.game_moves.clear()

    def slove(self):
        self.game_moves = self.shuffle_moves + self.game_moves
        self.reset_board()
        self.shuffle_moves.clear()

class Menu(object):
    def __init__(self):
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.draw_menu()

    def draw_menu(self):
        self.reset_rect = self.make_text('Reset', TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 90)
        self.new_rect = self.make_text('New Game', TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 60)
        self.solve_rect = self.make_text('Solve', TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 30)

    def show_msg(self, msg_id):
        msg = {
            'shuffle': 'Generating new puzzle...                                ',
            'running': 'Click tile or press arrow keys to slide.                ',
            'solve': 'Solved!                                                   ',
            'solving': 'Solving.........                                        ',
            'reset': 'Resetting.........                                        '
        }
        self.make_text(msg[msg_id], MESSAGECOLOR, BGCOLOR, 5, 5)

    def make_text(self, text, color, bgcolor, top, left):
        surf = self.font.render(text, True, color, bgcolor)
        rect = surf.get_rect()
        rect.topleft = (top, left)
        DISPLAYSURF.blit(surf, rect)
        return rect

    def check_rect(self, pos, board):
        if self.reset_rect.collidepoint(pos):
            self.show_msg('reset')
            board.reset_board()
        elif self.new_rect.collidepoint(pos):
            run_game(self)
        elif self.solve_rect.collidepoint(pos):
            self.show_msg('solving')
            board.slove()


if __name__ == '__main__':
    main()

