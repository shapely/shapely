import unittest
from unittest.mock import patch
from shapely._version import render
from shapely._version import render_git_describe


class TestRenderFunction(unittest.TestCase):
    #testing render_git_describe()
    def setUp(self):
        self.pieces = {
            "error": None,
            "long": "abcdef123456",
            "dirty": False,
            "closest-tag": "v1.0.0",
            "distance": 0,
            "short": "abcdef1",
            "date": "2024-01-01T00:00:00"
        }


    def test_render_with_error(self):
        self.pieces["error"] = "some error"
        result = render(self.pieces, "pep440")
        expected = {"version": "unknown",
                    "full-revisionid": self.pieces["long"],
                    "dirty": None,
                    "error": self.pieces["error"],
                    "date": None}
        self.assertEqual(result, expected)

    def test_render_default_style(self):
        result = render(self.pieces, None)
        self.assertEqual(result["error"], None)

    @patch('shapely._version.render_pep440', return_value="v1.0.0")
    def test_render_pep440_style(self, mock_render_pep440):
        result = render(self.pieces, "pep440")
        self.assertEqual(result["version"], "v1.0.0")

    @patch('shapely._version.render_pep440_branch', return_value="v1.0.0-branch")
    def test_render_pep440_branch_style(self, mock_render_pep440_branch):
        result = render(self.pieces, "pep440-branch")
        self.assertEqual(result["version"], "v1.0.0-branch")


    @patch('shapely._version.render_pep440_pre', return_value="v1.0.0-pre")
    def test_render_pep440_pre_style(self, mock_render_pep440_pre):
        result = render(self.pieces, "pep440-pre")
        self.assertEqual(result["version"], "v1.0.0-pre")

    @patch('shapely._version.render_pep440_post', return_value="v1.0.0-post")
    def test_render_pep440_post_style(self, mock_render_pep440_post):
        result = render(self.pieces, "pep440-post")
        self.assertEqual(result["version"], "v1.0.0-post")

    @patch('shapely._version.render_pep440_post_branch', return_value="v1.0.0-post-branch")
    def test_render_pep440_post_branch_style(self, mock_render_pep440_post_branch):
        result = render(self.pieces, "pep440-post-branch")
        self.assertEqual(result["version"], "v1.0.0-post-branch")

    @patch('shapely._version.render_pep440_old', return_value="v1.0.0-old")
    def test_render_pep440_old_style(self, mock_render_pep440_old):
        result = render(self.pieces, "pep440-old")
        self.assertEqual(result["version"], "v1.0.0-old")

    @patch('shapely._version.render_git_describe', return_value="v1.0.0-describe")
    def test_render_git_describe_style(self, mock_render_git_describe):
        result = render(self.pieces, "git-describe")
        self.assertEqual(result["version"], "v1.0.0-describe")

    @patch('shapely._version.render_git_describe_long', return_value="v1.0.0-describe-long")
    def test_render_git_describe_long_style(self, mock_render_git_describe_long):
        result = render(self.pieces, "git-describe-long")
        self.assertEqual(result["version"], "v1.0.0-describe-long")

    def test_render_unknown_style(self):
        with self.assertRaises(ValueError) as context:
            render(self.pieces, "unknown-style")
        self.assertIn("unknown style", str(context.exception))

    #testing render()
    def test_tagged_version(self):
        pieces = {
            "closest-tag": "v1.0.0",
            "distance": 5,
            "short": "abcdef1",
            "dirty": False
        }
        result = render_git_describe(pieces)
        self.assertEqual(result, "v1.0.0-5-gabcdef1")

    def test_untagged_version(self):
        pieces = {
            "closest-tag": None,
            "distance": 0,
            "short": "abcdef1",
            "dirty": False
        }
        result = render_git_describe(pieces)
        self.assertEqual(result, "abcdef1")


    def test_dirty_tagged_version(self):
        pieces = {
            "closest-tag": "v1.0.0",
            "distance": 5,
            "short": "abcdef1",
            "dirty": True
        }
        result = render_git_describe(pieces)
        self.assertEqual(result, "v1.0.0-5-gabcdef1-dirty")


    def test_dirty_untagged_version(self):
        pieces = {
            "closest-tag": None,
            "distance": 0,
            "short": "abcdef1",
            "dirty": True
        }
        result = render_git_describe(pieces)
        self.assertEqual(result, "abcdef1-dirty")


    def test_no_distance_tagged_version(self):
        pieces = {
            "closest-tag": "v1.0.0",
            "distance": 0,
            "short": "abcdef1",
            "dirty": False
        }
        result = render_git_describe(pieces)
        self.assertEqual(result, "v1.0.0")
