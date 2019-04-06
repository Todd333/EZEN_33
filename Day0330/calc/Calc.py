class Calc:
    def __init__(self,first,second):
        self.first = first
        self.second = second


    def set_num(self,first,second):
        self.first=first
        self.second=second


    def sum(self):
        return self.first + self.second


    def mul(self):
        return self.first * self.second


    def minus(self):
        return self.first - self.second


    def div(self):
        return self.first / self.second
