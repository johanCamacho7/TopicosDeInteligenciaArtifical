import random
import time


def print_array_view(array, current_pos, neighbors, chosen, target, window=40):
    start = max(0, current_pos - window // 2)
    end = min(len(array), current_pos + window // 2)

    view = []
    for i in range(start, end):
        if i == target:
            view.append("T")  # target
        elif i == current_pos:
            view.append("C")  # current
        elif i == chosen:
            view.append("S")  # selected next move
        elif i in neighbors:
            view.append(".")  # neighbor
        else:
            view.append("-")  # background
    print("".join(view))


def tabu_search_visual(array, target, max_iters=50, tabu_size=5, neighborhood_size=5):
    n = len(array)
    current_pos = random.randint(0, n - 1)
    best_pos = current_pos
    tabu_list = []

    print(f"Start at position {current_pos}, target = {target}\n")

    for step in range(max_iters):
        neighbors = [(current_pos + i) % n for i in range(-neighborhood_size, neighborhood_size + 1) if i != 0]
        candidates = [pos for pos in neighbors if pos not in tabu_list]

        scores = {pos: abs(pos - target) for pos in candidates}
        next_pos = min(scores, key=scores.get)

        print(f"Step {step + 1}")
        print_array_view(array, current_pos, candidates, next_pos, target)
        print(f"Current = {current_pos}, Chosen = {next_pos}, Distance to target = {abs(next_pos - target)}\n")

        tabu_list.append(current_pos)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        current_pos = next_pos

        if current_pos == target:
            print(f"Target found at position {current_pos}!")
            return current_pos

        time.sleep(0.3)  # small delay for readability

    print("Finished without exact match. Best found =", best_pos)
    return best_pos


# Example usage
array = list(range(100))
target_pos = 4
found_pos = tabu_search_visual(array, target_pos)
