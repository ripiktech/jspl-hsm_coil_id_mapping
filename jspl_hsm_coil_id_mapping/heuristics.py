from datetime import datetime

class OCRBuffer:
    def __init__(self, id_len=10):
        self.buffer = dict()
        self.id_len = id_len
        today = datetime.date.today()
        self.year = today.strftime("%Y")
    
    def _validate(self, id):
        if len(id) < self.id_len:
            return False
        if len(id) > self.id_len:
            return False
        try:
            int(id)
        except:
            return False
        if id[4:] != self.year:
            return False
        return True

    def add(self, id):
        if self._validate(id):
            if not id in self.buffer:
                self.buffer[id] = 0
            self.buffer[id] += 1
    
    def get(self):
        id, occ = max(self.buffer.items(), key=lambda x: x[1])
        return id, occ

    def is_empty(self):
        return True if len(self.buffer) == 0 else False
    
    def empty(self):
        self.buffer = dict()