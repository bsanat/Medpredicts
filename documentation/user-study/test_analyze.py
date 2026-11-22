import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analyze import FIELDS, SUS, analyze, read_rows, sus_summary, wilson


BASE = Path(__file__).resolve().parent
SYNTHETIC = BASE / "tests" / "responses-synthetic.csv"


class StudyAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.rows = read_rows(SYNTHETIC)

    def test_blank_template_and_small_n(self):
        empty = analyze(read_rows(BASE / "responses-template.csv"))["round_1"]
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["questions"]["q1"]["wilson_95_pct"])
        self.assertEqual(empty["threshold_status"], "provisional")
        small = analyze(self.rows[:7])["round_1"]
        self.assertEqual(small["questions"]["q1"]["correct_pct"], 100)
        self.assertEqual(small["questions"]["q1"]["status"], "provisional")

    def test_sus_formula_and_round_separation(self):
        extremes = [dict(zip(SUS, map(str, pattern * 5))) for pattern in ([5, 1], [1, 5], [3, 3])]
        self.assertEqual([sus_summary([row], 1, True)["mean"] for row in extremes], [100, 0, 50])
        result = analyze(self.rows)
        first, second = result["round_1"], result["round_2"]
        self.assertEqual((first["n"], second["n"]), (8, 2))
        self.assertEqual(first["sus"]["mean"], 68.75)
        self.assertEqual(first["sus"]["median"], 68.75)
        self.assertEqual(first["sus"]["status"], "met")
        self.assertEqual(second["sus"]["complete_n"], 1)
        self.assertEqual(second["sus"]["incomplete_n"], 1)
        self.assertEqual(second["threshold_status"], "descriptive")

    def test_wilson_and_exact_question_thresholds(self):
        lo, hi = wilson(8, 8)
        self.assertAlmostEqual(lo, 67.559, places=2)
        self.assertAlmostEqual(hi, 100, places=8)
        lo, hi = wilson(0, 8)
        self.assertAlmostEqual(lo, 0, places=8)
        self.assertAlmostEqual(hi, 32.441, places=2)
        first = analyze(self.rows)["round_1"]
        questions = first["questions"]
        self.assertEqual(questions["q3"]["correct_pct"], 75)
        self.assertEqual(questions["q3"]["status"], "not_met")
        self.assertEqual(questions["q4"]["correct_pct"], 87.5)
        self.assertEqual(questions["q4"]["status"], "provisional")
        self.assertEqual(questions["q7"]["status"], "met")
        self.assertEqual(first["threshold_status"], "not_met")

    def test_open_answers_agreement_and_consensus(self):
        questions = analyze(self.rows)["round_1"]["questions"]
        q2, q5 = questions["q2"], questions["q5"]
        self.assertEqual(q2["correct_n"], 6)
        self.assertEqual(q2["response_counts"]["partial"], 1)
        self.assertEqual(q2["rater_agreement"]["agreement_pct"], 87.5)
        self.assertAlmostEqual(q2["rater_agreement"]["cohen_kappa"], 23 / 31)
        self.assertEqual(q5["rater_agreement"]["unresolved_n"], 1)
        self.assertEqual(q5["status"], "provisional")
        self.rows[7]["q5_final"] = "partial"
        q5 = analyze(self.rows)["round_1"]["questions"]["q5"]
        self.assertEqual(q5["correct_n"], 7)
        self.assertEqual(q5["rater_agreement"]["disagreements_n"], 1)
        self.assertEqual(q5["status"], "met")
        agreement = analyze(self.rows)["round_2"]["questions"]["q5"]["rater_agreement"]
        self.assertEqual(agreement["agreement_pct"], 100)
        self.assertIsNone(agreement["cohen_kappa"])

    def test_step_time_excludes_technical_and_missing(self):
        step = analyze(self.rows)["round_1"]["steps"]["t1"]
        self.assertEqual(step["statuses"], {"unaided": 3, "assisted": 1, "not_completed": 1,
                                            "technical": 1, "missing": 2})
        self.assertEqual((step["attempted_n"], step["timed_n"]), (5, 4))
        self.assertEqual(step["median_seconds"], 41.25)
        self.assertEqual(step["errors_total"], 4)

    def test_csv_validation_and_cli_keep_raw_text_private(self):
        run = subprocess.run([sys.executable, str(BASE / "analyze.py"), str(SYNTHETIC)],
                             capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(run.stdout)["round_1"]["n"], 8)
        self.assertNotIn("SYNTHETIC FIXTURE", run.stdout)
        self.assertNotIn("P01", run.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "malformed-synthetic.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows([self.rows[0], {**self.rows[0], "round": "2"}])
            with self.assertRaisesRegex(ValueError, "unique across rounds"):
                read_rows(path)

            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow({**self.rows[0], "q2_final": "SYNTHETIC INVALID VALUE"})
            run = subprocess.run([sys.executable, str(BASE / "analyze.py"), str(path)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("CSV row 2: q2", run.stderr)
            self.assertNotIn("SYNTHETIC", run.stderr)


if __name__ == "__main__":
    unittest.main()
