"""The loop: perceive, decide, ask the one question that matters, then explain.

The shape of the conversation is set by the engine, not the model.  After each answer
the engine recomputes what is still undecided and names the single fact whose value
would resolve the most benefit; that fact becomes the next question.  When no remaining
fact can change any outcome, the loop stops - which is usually after two or three
questions, not twenty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .engine import evaluate, next_question, summarise
from .explain import explain
from .facts import QUESTIONS, Facts
from .perceive import perceive


@dataclass
class Turn:
    question: str | None
    utterance: str
    facts_after: Facts
    resolves: list[str] = field(default_factory=list)


@dataclass
class Session:
    schemes: dict
    chat_fn: object
    lang: str = "hi"
    facts: Facts = field(default_factory=Facts)
    turns: list[Turn] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_questions: int = 6

    def hear(self, utterance: str, question: str | None = None,
             expect_fact: str | None = None,
             resolves: list[str] | None = None) -> Facts:
        """Take one thing the person said and fold it into what we know."""
        new, warns, _raw = perceive(utterance, self.chat_fn, question, expect_fact)
        self.warnings.extend(warns)
        for name, value in new.known().items():
            # earlier answers win: a person correcting themselves says so explicitly,
            # and a later extraction inventing a value should not overwrite a stated one
            if getattr(self.facts, name) is None:
                self.facts = self.facts.with_(name, value)
        self.turns.append(Turn(question, utterance, self.facts, resolves or []))
        return self.facts

    def ask(self) -> tuple[str | None, str | None, list[str]]:
        """The next question to put to the person, or (None, ...) when done."""
        if len(self.turns) > self.max_questions:
            return None, None, []
        fact, diag = next_question(self.facts, self.schemes)
        if fact is None:
            return None, None, []
        text = QUESTIONS[fact][self.lang if self.lang in QUESTIONS[fact] else "en"]
        return fact, text, diag["resolves"].get(fact, [])

    def result(self):
        return summarise(evaluate(self.facts, self.schemes))

    def message(self) -> tuple[str, list[str]]:
        s = self.result()
        return explain(s["eligible"], s["uncertain"], self.schemes,
                       self.chat_fn, self.lang)
