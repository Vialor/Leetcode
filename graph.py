import collections
import heapq
from typing import List, Optional


class Solution:
    # 785*
    def isBipartite(self, graph: List[List[int]]) -> bool:
        self.colorScheme = [-1] * len(graph)

        def DFS(cur: int, color: int) -> bool:
            if self.colorScheme[cur] != -1 and self.colorScheme[cur] != color:
                return False
            if self.colorScheme[cur] != -1:
                return True
            self.colorScheme[cur] = color
            for neighbor in graph[cur]:
                if not DFS(neighbor, 1 if color == 0 else 0):
                    return False
            return True

        for i in range(len(graph)):
            if self.colorScheme[i] == -1 and not DFS(i, 0):
                return False
        return True

    # 207
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for course in range(numCourses)]
        for prerequisite in prerequisites:
            graph[prerequisite[1]].append(prerequisite[0])

        path = set()

        def DFSHasLoop(nodeIndex: int) -> bool:
            path.add(nodeIndex)
            for i in graph[nodeIndex]:
                if i in path or DFSHasLoop(i):
                    return True
            path.remove(nodeIndex)
            graph[nodeIndex] = []
            return False

        for i in range(len(graph)):
            if DFSHasLoop(i):
                return False
        return True

    # 210*
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for course in range(numCourses)]
        for prerequisite in prerequisites:
            graph[prerequisite[1]].append(prerequisite[0])

        stack = []
        path = set()
        visited = set()

        def DFSHasLoop(nodeIndex: int) -> bool:
            if nodeIndex in visited:
                return False
            visited.add(nodeIndex)

            path.add(nodeIndex)
            for i in graph[nodeIndex]:
                if i in path or DFSHasLoop(i):
                    return True
            stack.append(nodeIndex)
            path.remove(nodeIndex)
            return False

        for i in range(len(graph)):
            if DFSHasLoop(i):
                return []
        return stack[::-1]

    # 684
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        disjointSets = [-1] * (len(edges) + 1)

        def findRoot(i: int) -> int:
            cur = i
            while disjointSets[cur] != -1:
                cur = disjointSets[cur]
            result = cur
            # optimization
            cur = i
            while disjointSets[cur] != -1:
                disjointSets[cur] = result
                cur = disjointSets[cur]

            return result

        for edge in edges:
            rooti = findRoot(edge[0])
            rootj = findRoot(edge[1])
            if rooti != rootj:
                disjointSets[rooti] = rootj
            else:
                return edge
        return [-1, -1]

    # 322
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = collections.defaultdict(list)
        for start, dest in tickets:
            heapq.heappush(graph[start], dest)

        path = []

        def DFS(start):
            destList = graph[start]
            while len(destList) > 0:
                DFS(heapq.heappop(destList))
            path.append(start)

        DFS("JFK")
        return path[::-1]

    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        graph = [[] for _ in range(n)]  # from: (to, possibility from from to to)
        for i in range(len(edges)):
            a, b = edges[i]
            graph[a].append((b, succProb[i]))
            graph[b].append((a, succProb[i]))

        dp = [0] * n
        dp[start] = 1
        maxHeap = [(-1, start)]  # (best possibility from start to the node, node)
        while maxHeap:
            prevProb, node = heapq.heappop(maxHeap)
            for toNode, nextProb in graph[node]:
                newProb = -prevProb * nextProb
                if newProb > dp[toNode]:
                    heapq.heappush(maxHeap, (-newProb, toNode))
                    dp[toNode] = newProb
        return dp[end]
