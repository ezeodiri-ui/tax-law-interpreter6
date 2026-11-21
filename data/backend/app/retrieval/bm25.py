import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = [len(d) for d in docs]
        self.avg = sum(self.dl) / max(1, self.N)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in self.tf:
            df.update(d.keys())
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.inv = defaultdict(list)
        for i, d in enumerate(self.tf):
            for t in d:
                self.inv[t].append(i)

    def scores(self, q: list[str]) -> dict[int, float]:
        s: dict[int, float] = defaultdict(float)
        for t in set(q):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in self.inv[t]:
                f = self.tf[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return s
