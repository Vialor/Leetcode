from functools import cache
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # 104*
    # not dp-optimized
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    # 110
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def getHeight(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            leftHeight, rightHeight = getHeight(node.left), getHeight(node.right)
            if leftHeight == -1 or rightHeight == -1 or abs(leftHeight - rightHeight) > 1:
                return -1
            return max(leftHeight, rightHeight) + 1

        return getHeight(root) != -1

    # 543
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # (diameter of subtree, height of subtree)
        def DFS(node: Optional[TreeNode]):
            if not node:
                return (0, 0)
            leftDiameter, leftHeight = DFS(node.left)
            rightDiameter, rightHeight = DFS(node.right)
            return (max(leftDiameter, rightDiameter, leftHeight + rightHeight), max(leftHeight, rightHeight) + 1)

        return DFS(root)[0]

    # 226
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root

    # 617
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1:
            return root2
        if not root2:
            return root1
        result = root1
        result.val += root2.val
        result.left = self.mergeTrees(root1.left, root2.left)
        result.right = self.mergeTrees(root1.right, root2.right)
        return result

    # 112
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return targetSum == root.val
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)

    # 437
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        if not root:
            return 0

        def pathSumWithRoot(curNode: Optional[TreeNode], targetSum: int) -> int:
            if not curNode:
                return 0
            return (
                pathSumWithRoot(curNode.left, targetSum - curNode.val)
                + pathSumWithRoot(curNode.right, targetSum - curNode.val)
                + (1 if curNode.val == targetSum else 0)
            )

        return (
            self.pathSum(root.left, targetSum)
            + self.pathSum(root.right, targetSum)
            + pathSumWithRoot(root.left, targetSum - root.val)
            + pathSumWithRoot(root.right, targetSum - root.val)
            + (1 if root.val == targetSum else 0)
        )

    # 572*
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSameTree(tree1, tree2):
            if not tree1 and not tree2:
                return True
            if not tree1 or not tree2:
                return False
            return (
                tree1.val == tree2.val and isSameTree(tree1.left, tree2.left) and isSameTree(tree1.right, tree2.right)
            )

        if not root:
            return subRoot is None
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot) or isSameTree(root, subRoot)

    # 101
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def isMirrorTree(tree1, tree2):
            if not tree1 and not tree2:
                return True
            if not tree1 or not tree2:
                return False
            return (
                tree1.val == tree2.val
                and isMirrorTree(tree1.left, tree2.right)
                and isMirrorTree(tree1.right, tree2.left)
            )

        if not root:
            return True
        return isMirrorTree(root.left, root.right)

    # 687*
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        self.pathLength = 0

        def DFS(node):
            if not node:
                return 0
            leftUnivalueHeight, rightUnivalueHeight = DFS(node.left), DFS(node.right)
            leftUnivalueHeight = leftUnivalueHeight + 1 if node.left and node.left.val == node.val else 0
            rightUnivalueHeight = rightUnivalueHeight + 1 if node.right and node.right.val == node.val else 0
            self.pathLength = max(self.pathLength, leftUnivalueHeight + rightUnivalueHeight)
            return max(leftUnivalueHeight, rightUnivalueHeight)

        DFS(root)
        return self.pathLength

    @cache
    def rob(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        robWithoutRoot = self.rob(root.left) + self.rob(root.right)

        robWithRoot = root.val
        robWithRoot += self.rob(root.left.left) + self.rob(root.left.right) if root.left else 0
        robWithRoot += self.rob(root.right.left) + self.rob(root.right.right) if root.right else 0

        return max(robWithRoot, robWithoutRoot)

    # 669*
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        def trim(node):
            return self.trimBST(node, low, high)

        if root is None:
            return root
        if low > root.val:
            return trim(root.right)
        if high < root.val:
            return trim(root.left)
        root.left = trim(root.left)
        root.right = trim(root.right)
        return root

    # 230
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.count = 0

        def DFS(node):
            if node is None:
                return -1
            left = DFS(node.left)
            if left != -1:
                return left
            self.count += 1
            if self.count == k:
                return node.val
            right = DFS(node.right)
            if right != -1:
                return right
            return -1

        return DFS(root)
