class Queue:
    """
    Queue data structure.
    """

    def __init__(self, input_list: list[object]) -> None:
        self.queue = input_list

    def __repr__(self) -> str:
        return str(self.queue)

    def poll(self) -> object:
        # Removes first element of queue and returns it
        element = self.queue.pop(0)
        return element

    def add(self, value) -> None:
        # Adds an element to the end of the queue
        self.queue.append(value)

    def is_empty(self):
        # Checks if a queue is empty
        return self.queue == []
