import math
from collections import Counter


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""

    if not labels:
        return 0.0

    counts = Counter(labels)
    total = len(labels)

    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_information_gain(
    examples: list[dict],
    attr: str,
    target_attr: str
) -> float:
    """Calculate the information gain of splitting on attr."""

    if not examples:
        return 0.0

    # Entropy before splitting
    labels = [example[target_attr] for example in examples]
    parent_entropy = calculate_entropy(labels)

    # Group examples by the attribute value
    groups = {}

    for example in examples:
        value = example[attr]

        if value not in groups:
            groups[value] = []

        groups[value].append(example)

    # Calculate weighted entropy after splitting
    weighted_entropy = 0.0
    total = len(examples)

    for group in groups.values():
        group_labels = [
            example[target_attr]
            for example in group
        ]

        group_entropy = calculate_entropy(group_labels)

        weighted_entropy += (
            len(group) / total
        ) * group_entropy

    return parent_entropy - weighted_entropy


def majority_class(
    examples: list[dict],
    target_attr: str
) -> str:
    """Return the majority class. Break ties alphabetically."""

    labels = [
        example[target_attr]
        for example in examples
    ]

    counts = Counter(labels)

    max_count = max(counts.values())

    # Get all classes tied for the highest count
    candidates = [
        label
        for label, count in counts.items()
        if count == max_count
    ]

    # Alphabetically first label wins
    return min(candidates)


def learn_decision_tree(
    examples: list[dict],
    attributes: list[str],
    target_attr: str
) -> dict:
    """Build a decision tree using the ID3 algorithm."""

    labels = [
        example[target_attr]
        for example in examples
    ]

    # Base case 1:
    # All examples belong to the same class
    if len(set(labels)) == 1:
        return labels[0]

    # Base case 2:
    # No attributes left to split on
    if not attributes:
        return majority_class(examples, target_attr)

    # Find the attribute with the highest information gain
    gains = {
        attr: calculate_information_gain(
            examples,
            attr,
            target_attr
        )
        for attr in attributes
    }

    # max() will choose the first one in case of ties
    best_attr = max(gains, key=gains.get)

    # Create the tree
    tree = {
        best_attr: {}
    }

    # Remaining attributes for recursive calls
    remaining_attributes = [
        attr
        for attr in attributes
        if attr != best_attr
    ]

    # Find all possible values of the best attribute
    values = sorted(
        set(example[best_attr] for example in examples)
    )

    # Create a branch for each value
    for value in values:

        subset = [
            example
            for example in examples
            if example[best_attr] == value
        ]

        tree[best_attr][value] = learn_decision_tree(
            subset,
            remaining_attributes,
            target_attr
        )

    return tree