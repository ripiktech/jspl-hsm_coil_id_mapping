from datetime import datetime

class OCRBuffer:
    def __init__(self, id_len=10):
        self.buffer = dict()
        self.id_len = id_len
        self.year = 25
    
    def _validate(self, id):
        try:
            int(id)
        except:
            return False
        if int(id[:2]) != self.year:
            return False
        return True

    def add(self, id):
        # print(id)
        if self._validate(id):
            if not id in self.buffer:
                self.buffer[id] = 0
            self.buffer[id] += 1

    # def get(self):
    #     id, occ = None, None
    #     items = self.buffer.items()
    #     print(self.buffer)
    #     if items:
    #         id, occ = max(self.buffer.items(), key=lambda x: x[1])
    #     return id, occ
    
    def get(self):
        # id, occ = max(self.buffer.items(), key=lambda x: x[1])
        # Initialize the result ID
        best_id = []
        # Process each position across all IDs
        for position in range(self.id_len):
            # Count character occurrences at this position
            char_counts = {}
            for id_str in self.buffer:
                if position < len(id_str):
                    char = id_str[position]
                    char_counts[char] = char_counts.get(char, 0) + 1
            # Find the most common character
            best_char = max(char_counts.items(), key=lambda x: x[1])[0]
            best_id.append(best_char)
        return "".join(best_id)

    def is_empty(self):
        return True if len(self.buffer) == 0 else False
    
    def empty(self):
        self.buffer = dict()