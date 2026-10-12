"""
Automated tests for StackLens.      Run:  python -m unittest test_stacklens -v

Includes a randomised check: 400 random expressions are evaluated by StackLens
(using stacks) AND by an independent recursive-descent calculator written here
(using no stacks). The two must always agree.
"""
import io
import os
import random
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from decimal import Decimal, ROUND_HALF_EVEN, Context
from fractions import Fraction

import stacklens
from stacklens import Stack, StackLensError, evaluate, format_result, tokenize


def run(text):
    """What would appear in output.txt for this expression?"""
    try:
        return format_result(evaluate(text))
    except StackLensError as err:
        return "Error: %s" % err


# ---------- an independent calculator (no stacks) used as the referee ----------
def referee(expr):
    toks = re.findall(r"\d+\.?\d*|\.\d+|\*\*|[-+*/^()]", expr)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def nxt():
        t = peek(); pos[0] += 1; return t

    def e():
        v = t()
        while peek() in ("+", "-"):
            o = nxt(); r = t(); v = v + r if o == "+" else v - r
        return v

    def t():
        v = u()
        while peek() in ("*", "/"):
            o = nxt(); r = u(); v = v * r if o == "*" else v / r
        return v

    def u():
        if peek() == "-":
            nxt(); return -u()
        return p()

    def p():
        b = a()
        if peek() in ("^", "**"):
            nxt(); x = u(); return b ** int(x)
        return b

    def a():
        tok = nxt()
        if tok == "(":
            v = e(); nxt(); return v
        return Fraction(tok)

    return e()


def referee_text(fr):
    d = Context(prec=80).divide(Decimal(fr.numerator), Decimal(fr.denominator))
    d = d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN, context=Context(prec=120))
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def random_expression(rng, depth=0):
    kind = rng.random()
    if depth >= 3 or kind < 0.30:
        n = rng.choice([str(rng.randint(0, 20)), "%d.%d" % (rng.randint(0, 9), rng.randint(0, 99))])
        return n
    if kind < 0.45:
        return "(" + random_expression(rng, depth + 1) + ")"
    if kind < 0.52:
        return "-" + random_expression(rng, depth + 1)
    if kind < 0.60:
        return "%s ^ %d" % ("(" + random_expression(rng, depth + 1) + ")", rng.randint(0, 3))
    op = rng.choice(["+", "-", "*", "/"])
    return "%s %s %s" % (random_expression(rng, depth + 1), op, random_expression(rng, depth + 1))


class TestStack(unittest.TestCase):
    def test_last_in_first_out(self):
        s = Stack(); s.push(1); s.push(2); s.push(3)
        self.assertEqual([s.pop(), s.pop(), s.pop()], [3, 2, 1])

    def test_peek_leaves_stack_unchanged(self):
        s = Stack(); s.push("a")
        self.assertEqual(s.peek(), "a")
        self.assertEqual(s.size(), 1)

    def test_empty_stack_errors(self):
        s = Stack()
        self.assertTrue(s.is_empty())
        with self.assertRaises(IndexError): s.pop()
        with self.assertRaises(IndexError): s.peek()

    def test_snapshot_is_a_copy(self):
        s = Stack(); s.push(1)
        snap = s.snapshot(); snap.append(99)
        self.assertEqual(s.size(), 1)
        self.assertEqual(len(s), 1)


class TestTokenizer(unittest.TestCase):
    def test_positions_start_at_one(self):
        self.assertEqual([t.pos for t in tokenize("12 + 3")], [1, 4, 6])

    def test_negative_sign_versus_minus(self):
        self.assertEqual([t.kind for t in tokenize("-5 - -3")], ["neg", "num", "op", "neg", "num"])

    def test_double_star_is_power(self):
        self.assertEqual([t.value for t in tokenize("2 ** 3") if t.kind == "op"], ["^"])


class TestEvaluation(unittest.TestCase):
    def test_handout_examples(self):
        self.assertEqual(run("3 + 5 * 2"), "13")
        self.assertEqual(run("(8 / 4) + 7 * 2"), "16")      # the handout says 17, which is a typo
        self.assertEqual(run("10 - (2 + 3) * 4"), "-10")

    def test_left_to_right(self):
        self.assertEqual(run("100 / 4 / 5"), "5")
        self.assertEqual(run("10 - 4 - 3"), "3")

    def test_power_is_right_to_left(self):
        self.assertEqual(run("2 ^ 3 ^ 2"), "512")

    def test_negative_sign_and_power(self):
        self.assertEqual(run("-2 ^ 2"), "-4")
        self.assertEqual(run("(-2) ^ 2"), "4")
        self.assertEqual(run("2 ^ -2"), "0.25")

    def test_negative_numbers(self):
        self.assertEqual(run("-5 + 12"), "7")
        self.assertEqual(run("2 * -3"), "-6")
        self.assertEqual(run("-(2 + 3) * 2"), "-10")

    def test_exact_arithmetic(self):
        self.assertEqual(run("0.1 + 0.2"), "0.3")
        self.assertEqual(run("1 / 3 + 1 / 3 + 1 / 3"), "1")
        self.assertEqual(evaluate("1 / 3 + 1 / 3"), Fraction(2, 3))

    def test_brackets_and_spaces(self):
        self.assertEqual(run("((2 + 3) * (4 - 1)) / 5"), "3")
        self.assertEqual(run("   7*  6   "), "42")
        self.assertEqual(run("(((7)))"), "7")


class TestErrorMessagesGivePositions(unittest.TestCase):
    def test_division_by_zero(self):
        self.assertEqual(run("8 / (4 - 4)"), "Error: division by zero (the '/' at position 3)")

    def test_brackets(self):
        self.assertEqual(run("(3 + 4"), "Error: mismatched parentheses: '(' at position 1 is never closed")
        self.assertEqual(run("3 + 4)"), "Error: mismatched parentheses: unexpected ')' at position 6")
        self.assertEqual(run("()"), "Error: nothing to calculate before ')' at position 2")

    def test_syntax(self):
        self.assertEqual(run("3 + * 4"), "Error: unexpected '*' at position 5")
        self.assertEqual(run("3 4"), "Error: missing operator before number 4 at position 3")
        self.assertEqual(run("3(4)"), "Error: missing operator before '(' at position 2")
        self.assertEqual(run("5 -"), "Error: expression ends unexpectedly: a number is missing after '-' at position 3")

    def test_characters_and_numbers(self):
        self.assertEqual(run("7 $ 2"), "Error: invalid character '$' at position 3")
        self.assertEqual(run("1.2.3 + 4"), "Error: invalid number '1.2.3' at position 1")
        self.assertEqual(run(""), "Error: empty expression")
        self.assertEqual(run("+"), "Error: no numbers to calculate")       # found by the random test

    def test_power_limits(self):
        self.assertIn("whole number", run("2 ^ 0.5"))
        self.assertIn("division by zero", run("0 ^ -1"))
        self.assertIn("too large", run("2 ^ 5000"))
        self.assertIn("too large", run("((9 ^ 1000) ^ 1000) ^ 1000"))


class TestFormatting(unittest.TestCase):
    def test_whole_numbers_have_no_decimal_point(self):
        self.assertEqual(format_result(Fraction(13)), "13")
        self.assertEqual(format_result(Fraction(-4)), "-4")

    def test_four_decimal_places_trimmed(self):
        self.assertEqual(format_result(Fraction(10, 3)), "3.3333")
        self.assertEqual(format_result(Fraction(1, 8)), "0.125")
        self.assertEqual(format_result(Fraction(22, 7)), "3.1429")

    def test_tiny_negative_does_not_print_minus_zero(self):
        self.assertEqual(format_result(Fraction(-1, 100000)), "0")


class TestObserverAndFiles(unittest.TestCase):
    def test_observer_sees_every_step(self):
        steps = []
        evaluate("3 + 5 * 2", steps.append)
        self.assertEqual(steps[0]["values"], [])                      # start: empty stacks
        self.assertEqual(steps[-1]["values"][0][0], "13")             # end: the answer
        self.assertIn("apply 5 * 2 = 10", " ".join(s["note"] for s in steps))

    def test_file_round_trip_keeps_separators(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
            with open(src, "w") as f:
                f.write("3 + 5 * 2\n-----\n(8 / 4) + 7 * 2\n-----\n4 / 0\n\n-----\n2 ^ 10\n")
            answers, errors = stacklens.process_file(src, dst, os.path.join(d, "t.txt"), os.path.join(d, "v.html"))
            with open(dst) as f:
                lines = f.read().splitlines()
            self.assertEqual(lines, ["13", "-----", "16", "-----", "Error: division by zero (the '/' at position 3)", "", "-----", "1024"])
            self.assertEqual((answers, errors), (3, 1))
            with open(os.path.join(d, "v.html")) as f:
                html = f.read()
            self.assertIn("StackLens", html)
            self.assertIn('"expr": "3 + 5 * 2"', html)

    def test_command_line_single_expression(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = stacklens.main(["--expr", "3 + 5 * 2"])
        self.assertEqual(code, 0)
        self.assertIn("RESULT = 13", buf.getvalue())


class TestAgainstIndependentCalculator(unittest.TestCase):
    def test_400_random_expressions_agree(self):
        rng = random.Random(2026)
        checked = 0
        for _ in range(400):
            text = random_expression(rng)
            try:
                expected = referee(text)
            except ZeroDivisionError:
                self.assertIn("division by zero", run(text))
                continue
            self.assertEqual(run(text), referee_text(expected), msg=text)
            checked += 1
        self.assertGreater(checked, 300)

    def test_garbage_input_never_crashes(self):
        rng = random.Random(7)
        alphabet = "0123456789.+-*/^() $x"
        for _ in range(600):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
            try:
                evaluate(text)
            except StackLensError:
                pass            # a clean, reported error is fine; any other exception fails the test


if __name__ == "__main__":
    unittest.main()
