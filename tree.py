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
