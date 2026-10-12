#!/usr/bin/env python3
"""
StackLens - see every push and pop.

DCOMP320 Data Structures and Algorithms - Individual Assignment: Stacks Implementation
Author: Abdulai Bah

WHAT IT DOES
  1. Reads maths expressions from input.txt (one per line, groups separated by -----).
  2. Evaluates each expression using two stacks (a value stack and an operator stack).
  3. Writes the answers to output.txt, keeping every separator line.

WHAT MAKES IT DIFFERENT
  * Exact arithmetic: numbers are fractions, so 1/3 + 1/3 + 1/3 is exactly 1.
  * Every error says WHERE the problem is (position in the line).
  * Supports + - * / ^ (power, also written **), brackets, decimals, negative numbers.
  * Writes stacklens.html: open it in any browser to WATCH the stacks change,
    step by step, for every expression in the file (animated, keyboard friendly).
  * Writes trace.txt: the same working as plain text.

USAGE
  python stacklens.py                      read input.txt, write output.txt (+ trace.txt, stacklens.html)
  python stacklens.py in.txt out.txt       use other file names
  python stacklens.py --expr "3 + 5 * 2"   evaluate one expression and print its working
  python stacklens.py --no-visual          skip trace.txt and stacklens.html
"""

import argparse
import json
import os
import sys
from collections import namedtuple
from fractions import Fraction


# ======================================================================
# PART 1 - The Stack (Last In, First Out)
# ======================================================================
class Stack:
    """A LIFO stack. The END of the internal list is the TOP of the stack."""

    def __init__(self):
        self._items = []

    def push(self, item):
        """Put an item on top."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item."""
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it."""
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def is_empty(self):
        return not self._items

    def size(self):
        return len(self._items)

    def snapshot(self):
        """A copy of the contents, bottom first (used by the trace and the visualizer)."""
        return list(self._items)

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return "Stack(%r)" % (self._items,)


class StackLensError(Exception):
    """Any problem with an expression. The message is written into output.txt."""


# ======================================================================
# PART 2 - Tokens: numbers, operators, brackets (each remembers its position)
# ======================================================================
Token = namedtuple("Token", "kind value pos text")
# kind: "num" | "op" | "neg" | "lpar" | "rpar"

PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "^": 4}
RIGHT_ASSOCIATIVE = {"^"}          # 2 ^ 3 ^ 2 means 2 ^ (3 ^ 2)
MAX_EXPONENT = 1000
MAX_BITS = 100000                  # stops absurdly huge results


def symbol(token):
    """What to show for a token on screen."""
    return "-" if token.kind == "neg" else token.text


def tokenize(text):
    """Split text into tokens. Positions start at 1."""
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit() or ch == ".":
            start = i
            while i < len(text) and (text[i].isdigit() or text[i] == "."):
                i += 1
            piece = text[start:i]
            if piece.count(".") > 1 or piece == ".":
                raise StackLensError("invalid number '%s' at position %d" % (piece, start + 1))
            tokens.append(Token("num", Fraction(piece), start + 1, piece))
        elif ch in "+-*/^":
            pos = i + 1
            if ch == "*" and text[i:i + 2] == "**":     # ** is accepted as power
                ch = "^"
                i += 1
            i += 1
            has_left_operand = bool(tokens) and tokens[-1].kind in ("num", "rpar")
            if has_left_operand:
                tokens.append(Token("op", ch, pos, ch))
            elif ch == "-":
                tokens.append(Token("neg", "neg", pos, "-"))      # negative sign
            elif ch == "+":
                pass                                               # leading + changes nothing
            else:
                tokens.append(Token("op", ch, pos, ch))            # the parser reports it
        elif ch == "(":
            tokens.append(Token("lpar", "(", i + 1, "("))
            i += 1
        elif ch == ")":
            tokens.append(Token("rpar", ")", i + 1, ")"))
            i += 1
        else:
            raise StackLensError("invalid character '%s' at position %d" % (ch, i + 1))
    return tokens


def check_brackets(text):
    """Stack of open-bracket positions: finds exactly where brackets go wrong."""
    opened = Stack()
    for pos, ch in enumerate(text, start=1):
        if ch == "(":
            opened.push(pos)
        elif ch == ")":
            if opened.is_empty():
                raise StackLensError("mismatched parentheses: unexpected ')' at position %d" % pos)
            opened.pop()
    if not opened.is_empty():
        raise StackLensError("mismatched parentheses: '(' at position %d is never closed" % opened.peek())


# ======================================================================
# PART 3 - Calculating (exact fractions)
# ======================================================================
def power(base, exponent, pos):
    if exponent.denominator != 1:
        raise StackLensError("exponent must be a whole number (the '^' at position %d)" % pos)
    n = int(exponent)
    if abs(n) > MAX_EXPONENT:
        raise StackLensError("exponent too large, the limit is %d (the '^' at position %d)" % (MAX_EXPONENT, pos))
    if base == 0 and n < 0:
        raise StackLensError("division by zero (0 raised to a negative power, the '^' at position %d)" % pos)
    if max(base.numerator.bit_length(), base.denominator.bit_length()) * abs(n) > MAX_BITS:
        raise StackLensError("result too large (the '^' at position %d)" % pos)
    return base ** n


def apply_top(operators, values, note=None):
    """Take the top operator and its number(s), calculate, push the answer back."""
    op = operators.pop()
    if op.kind == "neg":
        a = values.pop()
        values.push(-a)
        if note is not None:
            note.append("apply negative sign to %s = %s" % (show(a), show(-a)))
        return
    b = values.pop()                  # the second number comes off first
    a = values.pop()
    if op.value == "+":
        r = a + b
    elif op.value == "-":
        r = a - b
    elif op.value == "*":
        r = a * b
    elif op.value == "/":
        if b == 0:
            raise StackLensError("division by zero (the '/' at position %d)" % op.pos)
        r = a / b
    else:
        r = power(a, b, op.pos)
    values.push(r)
    if note is not None:
        note.append("apply %s %s %s = %s" % (show(a), op.value, show(b), show(r)))


def should_apply_first(top, incoming):
    """Must the operator on top of the stack be calculated before 'incoming' is pushed?"""
    p_top = PRECEDENCE[top.value]
    p_new = PRECEDENCE[incoming.value]
    if p_top != p_new:
        return p_top > p_new
    return incoming.value not in RIGHT_ASSOCIATIVE      # equal: left to right


# ======================================================================
# PART 4 - The evaluator (two stacks)
# ======================================================================
def evaluate(text, observer=None):
    """Evaluate one expression and return an exact Fraction.

    observer(step) is called after every token and every calculation, with a dict
    describing both stacks. It is how the trace and the visualizer are produced.
    """
    text = text.strip()
    if not text:
        raise StackLensError("empty expression")
    check_brackets(text)
    tokens = tokenize(text)
    if not tokens:
        raise StackLensError("no numbers to calculate")     # e.g. a line containing only '+'

    values = Stack()          # numbers and results
    operators = Stack()       # operator tokens waiting for their turn
    expect_operand = True     # True when the next token must be a number, "(" or a negative sign

    def emit(index, notes):
        if observer is not None:
            observer({"tok": index, "note": "; ".join(notes),
                      "values": [show_pair(v) for v in values.snapshot()],
                      "ops": [symbol(o) for o in operators.snapshot()]})

    emit(-1, ["start: both stacks are empty"])
    for index, tok in enumerate(tokens):
        notes = []
        if tok.kind == "num":
            if not expect_operand:
                raise StackLensError("missing operator before number %s at position %d" % (tok.text, tok.pos))
            values.push(tok.value)
            expect_operand = False
            notes.append("number %s: push on the value stack" % tok.text)
        elif tok.kind == "neg":
            operators.push(tok)
            notes.append("negative sign: push on the operator stack")
        elif tok.kind == "lpar":
            if not expect_operand:
                raise StackLensError("missing operator before '(' at position %d" % tok.pos)
            operators.push(tok)
            notes.append("'(': push on the operator stack")
        elif tok.kind == "rpar":
            if expect_operand:
                raise StackLensError("nothing to calculate before ')' at position %d" % tok.pos)
            while operators.peek().kind != "lpar":
                apply_top(operators, values, notes)
            operators.pop()
            notes.append("')': brackets done, remove '('")
        else:  # + - * / ^
            if expect_operand:
                raise StackLensError("unexpected '%s' at position %d" % (tok.text, tok.pos))
            while (not operators.is_empty() and operators.peek().kind != "lpar"
                   and should_apply_first(operators.peek(), tok)):
                apply_top(operators, values, notes)
            operators.push(tok)
            expect_operand = True
            notes.append("operator %s: push on the operator stack" % tok.text)
        emit(index, notes)

    if expect_operand:
        last = tokens[-1]
        raise StackLensError("expression ends unexpectedly: a number is missing after '%s' at position %d"
                             % (symbol(last), last.pos))
    notes = []
    while not operators.is_empty():
        apply_top(operators, values, notes)
    emit(len(tokens), notes or ["end of expression: nothing left to apply"])
    return values.pop()


# ======================================================================
# PART 5 - Showing results
# ======================================================================
def format_result(x):
    """Whole numbers have no decimal point; others show up to 4 decimals (exact, half-even)."""
    if x.denominator == 1:
        try:
            return str(x.numerator)
        except ValueError:
            raise StackLensError("result too large to display")
    scaled = round(Fraction(abs(x.numerator) * 10000, x.denominator))
    whole, part = divmod(scaled, 10000)
    try:
        text = ("%d.%04d" % (whole, part)).rstrip("0").rstrip(".")
    except ValueError:
        raise StackLensError("result too large to display")
    return "-" + text if (x < 0 and scaled != 0) else text


def show(x):
    try:
        return format_result(x)
    except StackLensError:
        return "(huge)"


def show_pair(x):
    """[display text, exact fraction or None] for the visualizer."""
    exact = None
    if x.denominator != 1 and x.numerator.bit_length() < 200:
        exact = "%d/%d" % (x.numerator, x.denominator)
    return [show(x), exact]


def is_separator(line):
    """A separator is a line made only of three or more dashes."""
    s = line.strip()
    return len(s) >= 3 and set(s) == {"-"}


def process_file(in_path, out_path, trace_path=None, html_path=None):
    """Evaluate every expression in in_path; write out_path (and optional trace/visualizer)."""
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    output, trace, records = [], [], []
    answers = errors = 0
    for line in lines:
        if is_separator(line) or not line.strip():
            output.append(line.strip())             # separators and blank lines pass through
            continue
        steps = []
        expression = line.strip()
        trace.append("Expression: " + expression)
        try:
            result = format_result(evaluate(expression, steps.append))
            answers += 1
            bad = False
        except StackLensError as err:
            result = "Error: %s" % err
            errors += 1
            bad = True
        output.append(result)
        for n, s in enumerate(steps):
            trace.append("  step %-2d values=%-26s operators=%-14s %s" % (
                n, "[" + ", ".join(v[0] for v in s["values"]) + "]", "[" + ", ".join(s["ops"]) + "]", s["note"]))
        trace.append("  " + ("ERROR: " + result[7:] if bad else "RESULT = " + result) + "\n")
        try:
            toks = [[symbol(t), t.kind] for t in tokenize(expression)]
        except StackLensError:
            toks = []
        records.append({"expr": expression, "result": result, "bad": bad, "tokens": toks, "steps": steps})

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output) + "\n")
    if trace_path:
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trace) + "\n")
    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(build_html(records))
    return answers, errors


# ======================================================================
# PART 6 - The visualizer page (self-contained HTML, no internet needed)
# ======================================================================
def build_html(records):
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", data)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StackLens</title>
<style>
:root{--bg:#f6f7fb;--card:#ffffff;--ink:#1b2437;--muted:#6a7490;--line:#dfe3ee;--accent:#3b5bdb;--accent-soft:#e7ecff;
--val:#0f9d7a;--val-soft:#dcf6ee;--op:#d9480f;--op-soft:#ffe9dc;--ok:#0f9d7a;--bad:#c92a2a;--bad-soft:#ffe3e3;--shadow:0 6px 24px rgba(30,40,80,.08)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10131c;--card:#181c28;--ink:#e8ebf5;--muted:#8d97b5;--line:#2a3042;--accent:#7f9bff;--accent-soft:#232c52;
--val:#3ddbae;--val-soft:#14352c;--op:#ffa066;--op-soft:#3b2417;--ok:#3ddbae;--bad:#ff8787;--bad-soft:#3b1c1c;--shadow:none}}
:root[data-theme="dark"]{--bg:#10131c;--card:#181c28;--ink:#e8ebf5;--muted:#8d97b5;--line:#2a3042;--accent:#7f9bff;--accent-soft:#232c52;
--val:#3ddbae;--val-soft:#14352c;--op:#ffa066;--op-soft:#3b2417;--ok:#3ddbae;--bad:#ff8787;--bad-soft:#3b1c1c;--shadow:none}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Tahoma,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:16px 24px;border-bottom:1px solid var(--line);background:var(--card)}
.logo{width:38px;height:38px;border-radius:10px;background:var(--accent);display:grid;place-items:center;color:#fff;font-weight:800}
h1{font-size:20px;margin:0;letter-spacing:.2px}
header p{margin:0;color:var(--muted);font-size:13px}
header .sp{flex:1}
button{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:7px 13px;cursor:pointer}
button:hover{border-color:var(--accent)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.wrap{display:grid;grid-template-columns:300px 1fr;gap:20px;padding:20px 24px;max-width:1200px;margin:0 auto}
@media(max-width:820px){.wrap{grid-template-columns:1fr;padding:14px 16px}}
.list{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);max-height:78vh;overflow:auto}
.list h2{font-size:12px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted);margin:0;padding:14px 16px 8px}
.item{display:flex;gap:10px;align-items:center;width:100%;text-align:left;border:0;border-radius:0;border-top:1px solid var(--line);padding:10px 16px}
.item:hover{background:var(--accent-soft)}
.item.sel{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
.item .n{color:var(--muted);font-size:12px;width:22px}
.item .e{flex:1;font-family:ui-monospace,Consolas,monospace;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{font-size:12px;padding:1px 8px;border-radius:99px;background:var(--val-soft);color:var(--ok);font-weight:600}
.badge.bad{background:var(--bad-soft);color:var(--bad)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:20px}
.strip{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 16px}
.tk{font-family:ui-monospace,Consolas,monospace;font-size:18px;min-width:38px;text-align:center;padding:6px 10px;border-radius:9px;border:1px solid var(--line);color:var(--muted);transition:all .2s}
.tk.done{color:var(--ink);background:var(--bg)}
.tk.now{background:var(--accent);color:#fff;border-color:var(--accent);transform:translateY(-3px);box-shadow:0 6px 14px rgba(59,91,219,.35)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}
.controls .sp{flex:1}
.controls label{font-size:13px;color:var(--muted);display:flex;gap:8px;align-items:center}
.note{min-height:48px;padding:10px 14px;border-radius:10px;background:var(--accent-soft);margin-bottom:16px}
.note small{color:var(--muted)}
.stacks{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.col h3{margin:0 0 8px;font-size:12px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted)}
.well{min-height:260px;border:2px dashed var(--line);border-top:0;border-radius:0 0 12px 12px;padding:10px;display:flex;flex-direction:column-reverse;gap:6px}
.blk{padding:9px 12px;border-radius:9px;font-family:ui-monospace,Consolas,monospace;font-size:17px;text-align:center;border:1px solid transparent}
.blk small{display:block;font-size:11px;opacity:.75}
.well.v .blk{background:var(--val-soft);color:var(--val);border-color:var(--val)}
.well.o .blk{background:var(--op-soft);color:var(--op);border-color:var(--op)}
.blk.new{animation:drop .35s ease-out}
@keyframes drop{from{transform:translateY(-26px);opacity:0}to{transform:none;opacity:1}}
@media(prefers-reduced-motion:reduce){.blk.new{animation:none}.tk{transition:none}}
.top{font-size:11px;color:var(--muted);text-align:center;margin-top:4px}
.empty{margin:auto;color:var(--muted);font-size:13px;align-self:center}
.result{margin-top:16px;padding:12px 16px;border-radius:10px;font-weight:600}
.result.ok{background:var(--val-soft);color:var(--ok)}
.result.bad{background:var(--bad-soft);color:var(--bad)}
.log{margin:16px 0 0;padding:0;list-style:none;font-size:13px;max-height:170px;overflow:auto;border-top:1px solid var(--line)}
.log li{padding:5px 2px;color:var(--muted);border-bottom:1px solid var(--line)}
.log li.cur{color:var(--ink);font-weight:600}
.hint{color:var(--muted);font-size:12px;margin-top:12px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:0 6px;font-size:12px}
</style>
</head>
<body>
<header>
  <div class="logo" aria-hidden="true">SL</div>
  <div><h1>StackLens</h1><p>See every push and pop &middot; DCOMP320 stacks assignment &middot; Abdulai Bah</p></div>
  <span class="sp"></span>
  <button id="theme" type="button" aria-label="Switch between light and dark theme">Theme</button>
</header>
<div class="wrap">
  <nav class="list" aria-label="Expressions"><h2>Expressions</h2><div id="items"></div></nav>
  <main class="panel" id="main" aria-live="polite">
    <div class="strip" id="strip" aria-label="Tokens of the expression"></div>
    <div class="controls">
      <button id="first" type="button" aria-label="First step">&laquo;</button>
      <button id="prev" type="button" aria-label="Previous step">&lsaquo; Back</button>
      <button id="next" type="button" class="primary" aria-label="Next step">Next &rsaquo;</button>
      <button id="last" type="button" aria-label="Last step">&raquo;</button>
      <button id="play" type="button">Play</button>
      <span class="sp"></span>
      <label>Speed <input id="speed" type="range" min="1" max="5" value="3"></label>
      <span id="count" style="color:var(--muted);font-size:13px"></span>
    </div>
    <div class="note" id="note"></div>
    <div class="stacks">
      <div class="col"><h3>Value stack</h3><div class="well v" id="vals"></div><div class="top">top is the highest block</div></div>
      <div class="col"><h3>Operator stack</h3><div class="well o" id="ops"></div><div class="top">top is the highest block</div></div>
    </div>
    <div id="result"></div>
    <ul class="log" id="log"></ul>
    <p class="hint">Keys: <kbd>&larr;</kbd> <kbd>&rarr;</kbd> step &middot; <kbd>Space</kbd> play/pause &middot; <kbd>Home</kbd> <kbd>End</kbd> jump &middot; <kbd>&uarr;</kbd> <kbd>&darr;</kbd> change expression</p>
  </main>
</div>
<script>
const DATA = __DATA__;
let cur = 0, step = 0, timer = null, prevLens = [0, 0];
const $ = id => document.getElementById(id);

function buildList() {
  $("items").innerHTML = "";
  DATA.forEach((r, i) => {
    const b = document.createElement("button");
    b.className = "item"; b.type = "button"; b.dataset.i = i;
    b.innerHTML = '<span class="n">' + (i + 1) + '</span><span class="e"></span><span class="badge' + (r.bad ? ' bad' : '') + '"></span>';
    b.querySelector(".e").textContent = r.expr;
    b.querySelector(".badge").textContent = r.bad ? "error" : r.result;
    b.onclick = () => select(i);
    $("items").appendChild(b);
  });
}
function select(i) {
  stop(); cur = i; step = 0; prevLens = [0, 0];
  document.querySelectorAll(".item").forEach(el => el.classList.toggle("sel", +el.dataset.i === i));
  const sel = document.querySelector(".item.sel"); if (sel) sel.scrollIntoView({block: "nearest"});
  render();
}
function blocks(el, list, prevLen) {
  el.innerHTML = "";
  if (!list.length) { const e = document.createElement("span"); e.className = "empty"; e.textContent = "empty"; el.appendChild(e); return; }
  list.forEach((v, k) => {
    const d = document.createElement("div"); d.className = "blk" + (k >= prevLen ? " new" : "");
    const main = Array.isArray(v) ? v[0] : v; d.textContent = main;
    if (Array.isArray(v) && v[1]) { const s = document.createElement("small"); s.textContent = "= " + v[1]; d.appendChild(s); }
    el.appendChild(d);
  });
}
function render() {
  const r = DATA[cur], steps = r.steps, s = steps[Math.min(step, steps.length - 1)];
  const strip = $("strip"); strip.innerHTML = "";
  r.tokens.forEach((t, i) => {
    const d = document.createElement("span"); d.textContent = t[0];
    d.className = "tk" + (i === s.tok ? " now" : (i < s.tok ? " done" : ""));
    strip.appendChild(d);
  });
  if (!r.tokens.length) { strip.textContent = r.expr; }
  blocks($("vals"), s.values, prevLens[0]); blocks($("ops"), s.ops, prevLens[1]);
  prevLens = [s.values.length, s.ops.length];
  $("note").innerHTML = "<strong>Step " + step + ":</strong> " + esc(s.note || "");
  $("count").textContent = "Step " + step + " of " + (steps.length - 1);
  const last = step >= steps.length - 1, res = $("result");
  if (last) { res.innerHTML = '<div class="result ' + (r.bad ? 'bad' : 'ok') + '">' + (r.bad ? esc(r.result) : "Result = " + esc(r.result)) + "</div>"; } else { res.innerHTML = ""; }
  const log = $("log"); log.innerHTML = "";
  steps.slice(0, step + 1).forEach((x, i) => { const li = document.createElement("li"); li.textContent = i + ". " + x.note; if (i === step) li.className = "cur"; log.appendChild(li); });
  log.scrollTop = log.scrollHeight;
}
function esc(t) { return String(t).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c])); }
function go(n) { const max = DATA[cur].steps.length - 1; step = Math.max(0, Math.min(max, n)); render(); }
function stop() { if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; } }
function play() {
  if (timer) { stop(); return; }
  if (step >= DATA[cur].steps.length - 1) { step = 0; }
  $("play").textContent = "Pause";
  timer = setInterval(() => { if (step >= DATA[cur].steps.length - 1) { stop(); return; } go(step + 1); }, 1900 - $("speed").value * 320);
}
$("next").onclick = () => { stop(); go(step + 1); }; $("prev").onclick = () => { stop(); go(step - 1); };
$("first").onclick = () => { stop(); go(0); }; $("last").onclick = () => { stop(); go(1e9); };
$("play").onclick = play; $("speed").oninput = () => { if (timer) { stop(); play(); } };
$("theme").onclick = () => { const d = document.documentElement, dark = getComputedStyle(d).getPropertyValue("--bg").trim() === "#10131c"; d.setAttribute("data-theme", dark ? "light" : "dark"); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type === "range") return;
  if (e.key === "ArrowRight") { stop(); go(step + 1); } else if (e.key === "ArrowLeft") { stop(); go(step - 1); }
  else if (e.key === " " && e.target.tagName !== "BUTTON") { e.preventDefault(); play(); }
  else if (e.key === "Home") { stop(); go(0); } else if (e.key === "End") { stop(); go(1e9); }
  else if (e.key === "ArrowDown") { e.preventDefault(); select(Math.min(DATA.length - 1, cur + 1)); }
  else if (e.key === "ArrowUp") { e.preventDefault(); select(Math.max(0, cur - 1)); }
});
buildList(); if (DATA.length) select(0);
</script>
</body>
</html>
"""


# ======================================================================
# PART 7 - Command line
# ======================================================================
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="StackLens: stack-based expression evaluator with a visual trace.")
    ap.add_argument("input", nargs="?", default=os.path.join(here, "input.txt"), help="input file (default: input.txt next to this program)")
    ap.add_argument("output", nargs="?", default=os.path.join(here, "output.txt"), help="output file (default: output.txt next to this program)")
    ap.add_argument("--expr", help="evaluate a single expression and print its working")
    ap.add_argument("--no-visual", action="store_true", help="do not write trace.txt and stacklens.html")
    args = ap.parse_args(argv)

    if args.expr is not None:
        steps = []
        try:
            result = format_result(evaluate(args.expr, steps.append))
        except StackLensError as err:
            result = None
            print("Error: %s" % err)
        for n, s in enumerate(steps):
            print("step %-2d values=%-26s operators=%-14s %s" % (
                n, "[" + ", ".join(v[0] for v in s["values"]) + "]", "[" + ", ".join(s["ops"]) + "]", s["note"]))
        if result is not None:
            print("RESULT = " + result)
        return 0 if result is not None else 1

    if not os.path.exists(args.input):
        print("Error: cannot find %s" % args.input)
        print("Put input.txt in the same folder as stacklens.py, or give the file name.")
        return 1
    out_dir = os.path.dirname(os.path.abspath(args.output))
    trace_path = None if args.no_visual else os.path.join(out_dir, "trace.txt")
    html_path = None if args.no_visual else os.path.join(out_dir, "stacklens.html")
    answers, errors = process_file(args.input, args.output, trace_path, html_path)
    print("StackLens")
    print("  input   : %s" % args.input)
    print("  output  : %s" % args.output)
    print("  results : %d answers, %d error lines" % (answers, errors))
    if html_path:
        print("  trace   : %s" % trace_path)
        print("  visual  : %s   (open it in a web browser)" % html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
