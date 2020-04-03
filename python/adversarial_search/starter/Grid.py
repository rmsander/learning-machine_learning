from copy import deepcopy

directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)

class Grid:
    def __init__(self, size: int=4):
        self.size = size
        self.map  = [[0] * self.size for i in range(self.size)]

    def clone(self):
        """ Returns a new Grid with a cloned map """
        gridCopy = Grid(self.size)
        gridCopy.map = deepcopy(self.map)

        return gridCopy

    def canInsert(self, pos: tuple) -> bool:
        return self.getCellValue(pos) == 0

    def insertTile(self, pos: tuple, value: int) -> None:
        if self.canInsert(pos):
            self.setCellValue(pos, value)

    def crossBound(self, pos: tuple) -> bool:
        """ Returns True if position is within the board"""
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def setCellValue(self, pos: tuple, value: int) -> None:
        """ Set the value of cell at position pos to value """
        if self.crossBound(pos):
            self.map[pos[0]][pos[1]] = value

    def getCellValue(self, pos: tuple):
        """ Return the value at pos if valid """
        return self.map[pos[0]][pos[1]] if self.crossBound(pos) else None

    def getAvailableCells(self) -> list:
        """ Returns a list of empty cells """
        return [(x,y)
                for x in range(self.size)
                for y in range(self.size)
                if self.map[x][y] == 0]

    def getMaxTile(self) -> int:
        """ Returns the tile with maximum value """
        return max(max(row) for row in self.map)

    def move(self, direction: int):
        """ Moves the grid in a specified direction """
        if direction == UP:
            return self.moveUD(False)
        if direction == DOWN:
            return self.moveUD(True)
        if direction == LEFT:
            return self.moveLR(False)
        if direction == RIGHT:
            return self.moveLR(True)

    def moveUD(self, down:bool=False)->bool:
        """ Move up or down """
        r = range(self.size -1, -1, -1) if down else range(self.size)

        moved = False

        for j in range(self.size):
            cells = []

            for i in r:
                cell = self.map[i][j]

                if cell != 0:
                    cells.append(cell)

            self.merge(cells)

            for i in r:
                value = cells.pop(0) if cells else 0

                if self.map[i][j] != value:
                    moved = True

                self.map[i][j] = value

        return moved

    def moveLR(self, right:bool=False)->bool:
        """ Move left or right """
        r = range(self.size - 1, -1, -1) if right else range(self.size)

        moved = False

        for i in range(self.size):
            cells = []

            for j in r:
                cell = self.map[i][j]

                if cell != 0:
                    cells.append(cell)

            self.merge(cells)

            for j in r:
                value = cells.pop(0) if cells else 0

                if self.map[i][j] != value:
                    moved = True

                self.map[i][j] = value

        return moved

    def merge(self, cells:list) -> None:
        """ Merge tiles """
        if len(cells) <= 1: return cells

        i = 0
        while i < len(cells) - 1:
            if cells[i] == cells[i+1]:
                cells[i] *= 2

                del cells[i+1]

            i += 1

    def canMove(self, dirs=vecIndex):
        # Init Moves to be Checked
        checkingMoves = set(dirs)

        for x in range(self.size):
            for y in range(self.size):

                # If Current Cell is Filled
                if self.map[x][y]:

                    # Look Ajacent Cell Value
                    for i in checkingMoves:
                        move = directionVectors[i]

                        adjCellValue = self.getCellValue((x + move[0], y + move[1]))

                        # If Value is the Same or Adjacent Cell is Empty
                        if adjCellValue == self.map[x][y] or adjCellValue == 0:
                            return True

                # Else if Current Cell is Empty
                elif self.map[x][y] == 0:
                    return True

        return False

    def getAvailableMoves(self, dirs=vecIndex): # -> List[(int, Grid)]
        """ Returns a list of available moves, along with moved grids """
        availableMoves = []

        for x in dirs:
            gridCopy = self.clone()

            if gridCopy.move(x):
                availableMoves.append((x, gridCopy))

        return availableMoves

if __name__ == '__main__':
    g = Grid()
    g.map[0][0] = 2
    g.map[1][0] = 2
    g.map[3][0] = 4

    while True:
        for i in g.map:
            print(i)

        print(g.getAvailableMoves())

        v = input()

        g.move(v)
