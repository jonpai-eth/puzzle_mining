import argparse
import hashlib
import json
import math
import pathlib

import numpy as np
import tqdm

THIS_FILE = pathlib.Path(__file__)
PUZZLE_EXPLANATION = """\
Sudoku is a puzzle where you have to fill a nxn grid with numbers from 1 to n such that
 each of the n rows, columns, and contiguous, non-overlapping sqrt(n)xsqrt(n) blocks
 contain all the numbers from 1 to n. This implies that each row, column, and block must
 contain each number only once.

Some of the grid is already filled with numbers. The goal is to fill the rest of the
 grid such that the rules of Sudoku are satisfied.

An example instance of the puzzle may look like this (for n=4):

3 2 | 1 4
1 0 | 3 2
----+----
2 1 | 4 3
4 3 | 2 0

where unfilled values are represented with a '0'. A solution to this puzzle instance
 would be:

3 2 | 1 4
1 4 | 3 2
----+----
2 1 | 4 3
4 3 | 2 1

"""


def generate_trivial_solution(n: int) -> np.ndarray:
    """
    Generate a trivial Sudoku solution for a given grid size.

    Args:
        n (int): The grid size (e.g., 4 for a 4x4 Sudoku, 9 for a 9x9 Sudoku).

    Returns:
        np.ndarray: A trivial Sudoku solution array of shape (n, n).
    """
    m = int(math.sqrt(n))
    if m * m != n:
        raise ValueError("Grid size must be a perfect square (e.g., 4, 9, 16).")

    def pattern(r, c):
        return (m * (r % m) + r // m + c) % n

    numbers = np.arange(1, n + 1)
    puzzle = np.array([[numbers[pattern(r, c)] for c in range(n)] for r in range(n)])
    return puzzle.astype(np.uint8)


def shuffle_rows(puzzle: np.ndarray) -> np.ndarray:
    """
    Shuffle the rows within each block.

    Args:
        puzzle (np.ndarray): The Sudoku puzzle array.
        n (int): The grid size.

    Returns:
        np.ndarray: The puzzle with shuffled rows within each block.
    """
    n = puzzle.shape[0]
    m = int(math.sqrt(n))
    shuffled = puzzle.copy()
    for block_row in range(m):
        # Determine the row indices for the current block row
        start = block_row * m
        end = start + m
        block = shuffled[start:end]
        # Shuffle the rows within this block
        np.random.shuffle(block)
        shuffled[start:end] = block
    return shuffled


def shuffle_cols(puzzle: np.ndarray) -> np.ndarray:
    n = puzzle.shape[0]
    m = int(math.sqrt(n))
    shuffled = puzzle.copy()
    for block_col in range(m):
        # Determine the column indices for the current block column
        start = block_col * m
        end = start + m
        block = shuffled[:, start:end]
        # Shuffle the columns within this block
        shuffled[:, start:end] = block[:, np.random.permutation(m)]
    return shuffled


def shuffle_block_rows(puzzle: np.ndarray) -> np.ndarray:
    n = puzzle.shape[0]
    m = int(math.sqrt(n))
    shuffled = puzzle.copy()
    # Split the puzzle into block rows
    blocks = np.split(shuffled, m, axis=0)
    # Shuffle the block rows
    np.random.shuffle(blocks)
    # Concatenate back the shuffled blocks
    shuffled = np.vstack(blocks)
    return shuffled


def shuffle_block_cols(puzzle: np.ndarray) -> np.ndarray:
    n = puzzle.shape[0]
    m = int(math.sqrt(n))
    shuffled = puzzle.copy()
    # Split the puzzle into block columns
    blocks = np.split(shuffled, m, axis=1)
    # Shuffle the block columns
    np.random.shuffle(blocks)
    # Concatenate back the shuffled blocks
    shuffled = np.hstack(blocks)
    return shuffled


def shuffle_numbers(puzzle: np.ndarray) -> np.ndarray:
    n = puzzle.shape[0]
    numbers = np.arange(1, n + 1)
    shuffled_numbers = np.random.permutation(numbers)
    mapping = {
        original: shuffled for original, shuffled in zip(numbers, shuffled_numbers)
    }
    vectorized = np.vectorize(lambda x: mapping[x] if x != 0 else 0)
    return vectorized(puzzle)


def generate_full_board(n: int) -> np.ndarray:
    """
    Generate a new filled Sudoku board of size n x n.

    Args:
        n (int): The grid size (e.g., 4 for a 4x4 Sudoku, 9 for a 9x9 Sudoku).

    Returns:
        np.ndarray: A filled Sudoku board.
    """
    puzzle = generate_trivial_solution(n)

    puzzle = shuffle_block_rows(puzzle)
    puzzle = shuffle_block_cols(puzzle)

    puzzle = shuffle_rows(puzzle)
    puzzle = shuffle_cols(puzzle)

    puzzle = shuffle_numbers(puzzle)

    return puzzle


def remove_numbers(puzzle: np.ndarray, num_remove: int) -> np.ndarray:
    """
    Remove a specified number of numbers from the puzzle.

    0 is used to represent empty cells.

    Args:
        puzzle (np.ndarray): The Sudoku puzzle array.
        num_remove (int): The number of numbers to remove.

    Returns:
        np.ndarray: The puzzle with numbers removed.
    """
    # Flatten the puzzle to make it easier to work with.
    flat_puzzle = puzzle.flatten()

    # Get the shuffled indices of all the numbers in the puzzle.
    indices = np.random.permutation(len(flat_puzzle))

    # Remove the specified number of numbers.
    flat_puzzle[indices[:num_remove]] = 0

    # Reshape the puzzle back to its original shape.
    return flat_puzzle.reshape(puzzle.shape)


def stringify_puzzle(puzzle: np.ndarray) -> str:
    """
    Convert the Sudoku puzzle array into a formatted string with appropriate separators.

    Args:
        puzzle (np.ndarray): The Sudoku puzzle array.

    Returns:
        str: Formatted Sudoku puzzle string.
    """
    n = puzzle.shape[0]
    m = int(math.sqrt(n))
    if m * m != n:
        raise ValueError("Grid size must be a perfect square (e.g., 4, 9, 16).")

    # Determine the width of each cell based on the largest number
    cell_width = len(str(n))

    # Build the horizontal separator line
    # Inner blocks have more dashes to account for spaces on both sides
    # Outer separators have fewer dashes
    separator_segments = []
    for block in range(m):
        if block == 0 or block == m - 1:
            # Outer blocks have fewer dashes
            segment = "-" * (m * (cell_width + 1))
        else:
            # Inner blocks have more dashes
            segment = "-" * (m * (cell_width + 1) + 1)
        separator_segments.append(segment)
    block_separator = "+".join(separator_segments)

    lines = []
    for i in range(n):
        if i % m == 0 and i != 0:
            lines.append(block_separator)

        row = []
        for j in range(n):
            if j % m == 0 and j != 0:
                row.append("|")
            cell = str(puzzle[i, j]).rjust(cell_width)
            row.append(cell)
        # Join the cells in the row with spaces for readability
        lines.append(" ".join(row))

    return "\n".join(lines)


def stringify_index(row: int, col: int) -> str:
    """
    Convert the (row, col) index to a human-readable string.

    Args:
        row (int): The row index (0-based).
        col (int): The column index (0-based).

    Returns:
        str: The formatted index string.
    """
    return f"({row + 1}, {col + 1})"


def is_solved(puzzle: np.ndarray) -> bool:
    """
    Check if the puzzle is solved.

    Args:
        puzzle (np.ndarray): A 2D array representing the Sudoku puzzle.

    Returns:
        bool: True if the puzzle is solved, False otherwise.
    """
    n = puzzle.shape[0]
    m = int(math.sqrt(n))

    # Check rows and columns
    for i in range(n):
        row = puzzle[i, :]
        if not np.array_equal(np.sort(row[row > 0]), np.arange(1, n + 1)):
            return False

        col = puzzle[:, i]
        if not np.array_equal(np.sort(col[col > 0]), np.arange(1, n + 1)):
            return False

    # Check blocks
    for box_r in range(m):
        for box_c in range(m):
            block = puzzle[box_r * m : (box_r + 1) * m, box_c * m : (box_c + 1) * m]
            block = block.flatten()
            if not np.array_equal(np.sort(block[block > 0]), np.arange(1, n + 1)):
                return False

    return True


def generate_prompt(puzzle: np.ndarray) -> str:
    """
    Generate a prompt for the puzzle.

    Args:
        puzzle (np.ndarray): The Sudoku puzzle array.

    Returns:
        A string representing the prompt.
    """
    n = puzzle.shape[0]
    puzzle = stringify_puzzle(puzzle)
    # return (
    #     f"{PUZZLE_EXPLANATION}\nNow let's solve the following {n}x{n} Sudoku puzzle:"
    #     f"\n\n{puzzle}\n\nLet's think step by step to solve this problem.\n"
    # )
    return (
        f"Let's solve the following {n}x{n} Sudoku puzzle:\n\n{puzzle}\n\n"
        "Let's think step by step to solve this problem.\n"
    )


def check_value(puzzle: np.ndarray, row: int, col: int, value: int) -> bool:
    """
    Check if the value can be placed at the specified (row, col) in the puzzle.

    Args:
        puzzle (np.ndarray): A 2D array representing the Sudoku puzzle.
        row (int): The row index.
        col (int): The column index.
        value (int): The value to check.

    Returns:
        bool: True if the value can be placed, False otherwise.
    """
    n = puzzle.shape[0]
    m = int(math.sqrt(n))

    # Check row
    if value in puzzle[row, :]:
        return False

    # Check column
    if value in puzzle[:, col]:
        return False

    # Check block
    start_row = (row // m) * m
    start_col = (col // m) * m
    block = puzzle[start_row : start_row + m, start_col : start_col + m]
    if value in block:
        return False

    return True


def backtrack(
    puzzle: np.ndarray, missing_positions: list[tuple[int, int]], cot: str
) -> tuple[str, bool]:
    """
    Solve the Sudoku puzzle using backtracking and generate a CoT solution.

    Args:
        puzzle (np.ndarray): A 2D array representing the Sudoku puzzle.
        missing_positions (list of tuples): List of (row, col) tuples for missing cells.
        cot (str): Current chain-of-thought string.

    Returns:
        tuple: Updated CoT string and a boolean indicating if the puzzle is solved.
    """
    if not len(missing_positions):
        cot += (
            "\nAlright that's it! We filled in every missing value and there are no"
            " conflicts.\n"
        )
        return cot, True

    row, col = missing_positions[0]

    cot += f"\nAttempting to solve cell {stringify_index(row, col)}.\n"

    for value in range(1, puzzle.shape[0] + 1):
        if check_value(puzzle, row, col, value):
            puzzle[row, col] = value
            cot += (
                f"For now, putting in value {value} at {stringify_index(row, col)}.\n"
                "This would make our board look like this:\n"
                f"{stringify_puzzle(puzzle)}\n"
            )
            new_cot, solved = backtrack(puzzle, missing_positions[1:], cot)
            if solved:
                return new_cot, True
            puzzle[row, col] = 0
            cot += (
                f"Removing value {value} from {stringify_index(row, col)} because"
                " it didn't work.\nSo the board is back to:\n"
                f"{stringify_puzzle(puzzle)}\n"
            )
        else:
            cot += (
                f"Value {value} cannot be placed at {stringify_index(row, col)} due to"
                " a conflict.\n"
            )

    cot += (
        f"No valid values found for cell {stringify_index(row, col)}."
        " Initiating backtracking.\n"
    )
    return cot, False


def generate_cot_solution(puzzle: np.ndarray) -> str:
    """
    Generate a CoT solution for the puzzle.

    Args:
        masked_puzzle: A 3x3x3x3 array representing the puzzle.
        solution: A 3x3x3x3 array representing the solution.

    Returns:
        A string representing the CoT solution.
    """
    missing_indices = np.argwhere(puzzle == 0)
    simplified_missing_indices = [stringify_index(*idx) for idx in missing_indices]

    cot = (
        f"First, let's see which values are missing in the puzzle. There are"
        f" {len(missing_indices)} missing numbers at indices"
        f" [{', '.join(simplified_missing_indices)}] Let's go through them one by one"
        " and search for a solution using backtracking.\n"
        "Let's look at the first missing number.\n"
    )

    cot, _ = backtrack(puzzle, missing_indices, cot)
    return cot


def generate_dataset_sudoku(
    puzzle_specs: list[tuple[int, int, int]], out_path: pathlib.Path
) -> None:
    """Generate a dataset of Sudoku puzzles in JSONL format."""

    def get_puzzle():
        while True:
            full_board = generate_full_board(n)
            solution = stringify_puzzle(full_board)
            if (
                puzzle_id := hashlib.sha224(solution.encode()).hexdigest()
                not in generated_puzzles
            ):
                generated_puzzles.add(puzzle_id)
                return full_board, solution

    with (
        out_path.open("w") as f_train,
        (out_path.with_suffix(".val.jsonl")).open("w") as f_val,
    ):
        for n, num_puzzles, num_remove in puzzle_specs:
            generated_puzzles = set()
            for _ in tqdm.tqdm(range(num_puzzles), f"train sudoku ({n}, {num_remove})"):
                full_board, solution = get_puzzle()
                masked_board = remove_numbers(full_board, num_remove)

                cot_solution = generate_cot_solution(masked_board.copy())
                text = (
                    generate_prompt(masked_board)
                    + cot_solution
                    + f"So the solved board is\n{solution}\n"
                )

                entry = {
                    "complexity": f"{n}, {num_remove}",
                    "text": text,
                }
                f_train.write(json.dumps(entry) + "\n")

            for _ in tqdm.tqdm(range(10), "val sudoku ({n}, {num_remove})"):
                full_board, solution = get_puzzle()
                masked_board = remove_numbers(full_board, num_remove)

                entry = {
                    "complexity": f"{n}, {num_remove}",
                    "problem": generate_prompt(masked_board),
                    "solution": solution,
                }
                f_val.write(json.dumps(entry) + "\n")


def parse_puzzle_spec(spec_str):
    try:
        parts = spec_str.split(",")
        if len(parts) != 3:
            raise ValueError("Exactly three values required.")
        grid_size, num_puzzles, num_remove = (int(part.strip()) for part in parts)

        if math.isqrt(grid_size) ** 2 != grid_size:
            raise ValueError("GRID_SIZE must be a perfect square (e.g., 4, 9, 16).")

        if num_puzzles <= 0:
            raise ValueError("NUM_PUZZLES must be a positive integer.")

        if not (0 <= num_remove < grid_size**2):
            raise ValueError(f"NUM_REMOVE must be between 0 and {grid_size**2 - 1}.")

        return (grid_size, num_puzzles, num_remove)

    except ValueError as ve:
        raise argparse.ArgumentTypeError(
            f"Invalid puzzle specification '{spec_str}': {ve}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Sudoku puzzles.",
        epilog=(
            "Example usage:\n"
            "  python sudoku.py \\\n"
            "      --puzzle-spec 9,50,40 \\\n"
            "      --puzzle-spec 16,20,60 \\\n"
            "      --out-path datasets/custom_sudoku.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--puzzle-spec",
        action="append",
        type=parse_puzzle_spec,
        metavar="GRID_SIZE,NUM_PUZZLES,NUM_REMOVE",
        help="Specify puzzle specifications as three integers separated by commas: GRID_SIZE,NUM_PUZZLES,NUM_REMOVE. This argument can be used multiple times.",
    )

    parser.add_argument(
        "--out-path",
        default=THIS_FILE.parent / "data" / "raw" / "sudoku.jsonl",
        help="Output path for the dataset.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    generate_dataset_sudoku(args.puzzle_spec, args.out_path)


if __name__ == "__main__":
    main()
