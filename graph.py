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

    # 1514
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

    # 743* Dijkstra
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # graph represented by adj list - {fromNode: [(cost, toNode)]}
        graph = [[] if i > 0 else None for i in range(n + 1)]
        for u, v, w in times:
            graph[u].append((w, v))

        minHeap = [(0, k)]
        processed = set()
        while minHeap:
            curCost, curNode = heapq.heappop(minHeap)
            processed.add(curNode)
            if len(processed) == n:
                return curCost
            for neighborCost, neighbor in graph[curNode]:
                newCost = curCost + neighborCost
                if neighbor not in processed:
                    heapq.heappush(minHeap, (newCost, neighbor))
        return -1

    # 1631*
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        m, n = len(heights), len(heights[0])
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        availablePaths = [(0, (0, 0))]  # (cost, dest)
        dp = [[float("inf")] * n for _ in range(m)]
        dp[0][0] = 0
        while True:
            cost, (i, j) = heapq.heappop(availablePaths)
            if i == m - 1 and j == n - 1:
                return dp[-1][-1]
            for a, b in neighbors:
                ci, cj = a + i, b + j
                if 0 <= ci < m and 0 <= cj < n:
                    newCost = max(cost, abs(heights[i][j] - heights[ci][cj]))
                    if newCost < dp[ci][cj]:
                        dp[ci][cj] = newCost
                        heapq.heappush(availablePaths, (newCost, (ci, cj)))

    # 787
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        graph = [[] for _ in range(n)]

        for fromNode, toNode, cost in flights:
            graph[fromNode].append((cost, toNode))

        dp = [float("inf") if i != src else 0 for i in range(n)]
        currentLevel = [src]
        while currentLevel and k >= 0:
            nextLevel = []
            nextdp = dp[:]
            for node in currentLevel:
                for newCost, newNode in graph[node]:
                    if dp[node] + newCost < nextdp[newNode]:
                        nextdp[newNode] = dp[node] + newCost
                        nextLevel.append(newNode)
            currentLevel = nextLevel
            dp = nextdp
            k -= 1
        return dp[dst] if dp[dst] != float("inf") else -1

    # 778
    def swimInWater(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[float("inf")] * n for _ in range(m)]
        solved = set()
        availablePaths = [(grid[0][0], 0, 0)]  # cost, i, j
        while availablePaths:
            cost, i, j = heapq.heappop(availablePaths)
            if (i, j) in solved:
                continue
            if i == m - 1 and j == n - 1:
                return cost
            solved.add((i, j))
            dp[i][j] = cost
            for ni, nj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in solved:
                    heapq.heappush(availablePaths, (max(cost, grid[ni][nj]), ni, nj))

    # 329*
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        dp = [[-1] * n for _ in range(m)]

        def increasingPathSearch(i, j):
            if dp[i][j] != -1:
                return
            neighborLengths = [0]
            for ni, nj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                    increasingPathSearch(ni, nj)
                    neighborLengths.append(dp[ni][nj])
            dp[i][j] = 1 + max(neighborLengths)

        for i in range(m):
            for j in range(n):
                increasingPathSearch(i, j)
        return max(sum(dp, []))

    # 834
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)

        ans = [0] * n
        size = [1] * n

        # calculate size and ans[0]
        unvisited = set(range(n))

        def DFS1(node) -> int:
            unvisited.remove(node)
            distanceSum = 0
            for neighbor in graph[node]:
                if neighbor in unvisited:
                    neighborDistanceSum = DFS1(neighbor)
                    distanceSum += neighborDistanceSum + size[neighbor]
                    size[node] += size[neighbor]
            return distanceSum

        ans[0] = DFS1(0)

        # calculate ans
        unvisited = set(range(n))

        def DFS2(node):
            unvisited.remove(node)
            for neighbor in graph[node]:
                if neighbor in unvisited:
                    ans[neighbor] = ans[node] + n - 2 * size[neighbor]
                    DFS2(neighbor)

        DFS2(0)
        return ans
