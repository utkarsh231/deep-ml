from collections import deque

def run_producer_consumer(capacity, producer_items, num_consumers):
    buffer = deque()
    consumed = []

    # Current position for each producer
    producer_idx = [0] * len(producer_items)

    remaining = sum(len(items) for items in producer_items)

    while remaining > 0:

        # Producers add items while there is space
        for p in range(len(producer_items)):
            if len(buffer) >= capacity:
                break

            if producer_idx[p] < len(producer_items[p]):
                item = producer_items[p][producer_idx[p]]
                buffer.append(item)

                producer_idx[p] += 1

        # Consumers remove available items
        for _ in range(num_consumers):
            if buffer:
                item = buffer.popleft()
                consumed.append(item)
                remaining -= 1

    return sorted(consumed)