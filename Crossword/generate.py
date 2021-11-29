import sys

from crossword import *
import copy


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        domains = copy.deepcopy(self.domains)
        for variables in domains:
            for words in domains[variables]:
                if len(words) != variables.length:
                    self.domains[variables].remove(words)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        domains = copy.deepcopy(self.domains)
        revision = False
        if self.crossword.overlaps[x, y] is None:
            return revision
        i, j = self.crossword.overlaps[x, y]
        for words_in_x in domains[x]:
            correspond = False
            for words_in_y in domains[y]:
                if words_in_x[i] == words_in_y[j]:
                    correspond = True
                    break
            if correspond is False:
                self.domains[x].remove(words_in_x)
                revision = True
        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = []
            for x in self.domains:
                for y in self.domains:
                    if x != y and tuple([y, x]) not in arcs and self.crossword.overlaps[x, y] is not None:
                        arcs.append(tuple([x, y]))
        for arc in arcs:
            x, y = arc
            if self.revise(x, y) is True:
                for var in self.domains:
                    if x != var and tuple([x, var]) not in arcs and self.crossword.overlaps[x, var] is not None:
                        arcs.append(tuple([var, y]))
        for var in self.domains:
            if len(self.domains[var]) == 0:
                return False
            return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if assignment is False:
            return False
        if len(assignment) != len(self.crossword.variables):
            return False
        for var in assignment:
            if assignment[var] is None or assignment[var] not in self.crossword.words:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for var1 in assignment:
            for var2 in assignment:
                if var1 != var2 and assignment[var1] == assignment[var2]:
                    return False
        for var in assignment:
            if len(assignment[var]) != var.length:
                return False
        for var3 in assignment:
            for var4 in self.crossword.neighbors(var3):
                if var4 in assignment:
                    i, j = self.crossword.overlaps[var3, var4]
                    if assignment[var3][i] != assignment[var4][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        ordering = {}
        values = self.domains[var]
        neighbors = self.crossword.neighbors(var)
        for val in values:
            count = 0
            for neighbor in neighbors:
                if neighbor not in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    for val0 in self.domains[neighbor]:
                        if val[i] == val0[j]:
                            count += 1
            ordering[val] = count
        sorted_ord = sorted(ordering, key=lambda x: ordering[x])
        # print(sorted_ord)
        return sorted_ord
        # return self.domains[var]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        lowests = []
        lowest = -1
        for var in self.crossword.variables:
            if var not in assignment:
                if lowest == -1:
                    lowest = len(self.domains[var])
                elif len(self.domains[var]) < lowest:
                    lowest = len(self.domains[var])
        for var in self.crossword.variables:
            if var not in assignment:
                if len(self.domains[var]) == lowest:
                    lowests.append(var)
        if len(lowests) == 1:
            return lowests[0]
        else:
            most = -1
            for var in lowests:
                if most == -1:
                    most = len(self.crossword.neighbors(var))
                elif len(self.crossword.neighbors(var)) > most:
                    most = len(self.crossword.neighbors(var))
        for var in lowests:
            if len(self.crossword.neighbors(var)) == most:
                return var

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment) is True:
            return assignment
        var = self.select_unassigned_variable(assignment)
        # print(var)
        for val in self.order_domain_values(var, assignment):
            assignment[var] = val
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                assignment.pop(var)
            else:
                assignment.pop(var)
        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
