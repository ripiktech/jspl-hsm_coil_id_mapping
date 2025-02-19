from datetime import datetime

class OCRBuffer:
    """
    A buffer for processing and validating OCR-extracted identification numbers.
    
    This class stores and tracks frequency of detected ID numbers, performs validation,
    and can generate a consensus ID by selecting the most frequent character at each position.
    
    Attributes:
        buffer (dict): Dictionary mapping detected IDs to their occurrence count.
        id_len (int): Expected length of valid identification numbers.
        year (int): Two-digit year prefix that valid IDs must start with.
        
    Methods:
        add(id): Add an ID to the buffer after validation.
        get(): Return the most likely correct ID based on character frequency analysis.
        is_empty(): Check if the buffer contains any IDs.
        empty(): Clear all IDs from the buffer.
        _validate(id): Validate if an ID has the correct format and year prefix.
    """
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
    
    def get(self):
        if not self.buffer:
            return ""
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