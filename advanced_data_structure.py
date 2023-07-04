from typing import List
import sortedcontainers


class DisjointSet:
    def __init__(self):
        self.parent = [i for i in range(26)]

    def findRoot(self, e):
        if self.parent[e] == e:
            return e
        self.parent[e] = self.findRoot(self.parent[e])
        return self.parent[e]

    def union(self, e1, e2):
        r1 = self.findRoot(e1)
        r2 = self.findRoot(e2)
        if r1 != r2:
            self.parent[r1] = r2


class Solution:
    # 990
    def equationsPossible(self, equations: List[str]) -> bool:
        letterToIndex = lambda c: ord(c) - 97
        d = DisjointSet()
        neqList = []
        for eq in equations:
            if eq[1] == "=":
                d.union(letterToIndex(eq[0]), letterToIndex(eq[3]))
            else:
                neqList.append((eq[0], eq[3]))
        for c1, c2 in neqList:
            if d.findRoot(letterToIndex(c1)) == d.findRoot(letterToIndex(c2)):
                return False
        return True

    # 218
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        def sortFunc(point):
            x, isEntering, height = point
            if isEntering:
                return (x, 0, -height)
            else:
                return (x, 1, height)

        keyPoints = []  # (x, isEntering, height)
        for l, r, h in buildings:
            keyPoints.append((l, True, h))
            keyPoints.append((r, False, h))
        keyPoints.sort(key=sortFunc)

        skylines = []
        curRoofs = sortedcontainers.SortedList([0])
        for x, isEntering, height in keyPoints:
            if isEntering:
                if curRoofs[-1] < height:
                    skylines.append((x, height))
                curRoofs.add(height)
            else:
                curRoofs.remove(height)
                if curRoofs[-1] < height:
                    skylines.append((x, curRoofs[-1]))
        return skylines
