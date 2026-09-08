"""Deterministic capability-boundary fixture, not an LLM or secure sandbox."""
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Call:
    tool: str
    argument: str
    key: str


@dataclass(frozen=True)
class Finish:
    text: str


@dataclass(frozen=True)
class Observation:
    tool: str
    text: str
    trusted_as_instruction: bool = False


@dataclass
class Environment:
    documents: dict[str, str]
    allow_drafts: bool = False
    drafts: list[str] = field(default_factory=list)
    receipts: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    timeout_after_commit_once: bool = False

    def execute(self, call: Call) -> str:
        if not all(isinstance(value, str) for value in (call.tool, call.argument, call.key)):
            raise ValueError("call fields must be strings")
        if call.tool not in {"read_document", "save_draft"}:
            raise PermissionError("tool not allowed")
        if not call.key or not isinstance(call.argument, str):
            raise ValueError("invalid call schema")
        if call.tool == "save_draft" and not self.allow_drafts:
            raise PermissionError("draft capability not granted")
        if call.tool == "read_document" and call.argument not in self.documents:
            raise KeyError("document is not in the currently authorized store")
        # Authorize BEFORE replay: a revoked capability stays revoked.
        if call.key in self.receipts:
            tool, argument, result = self.receipts[call.key]
            if (tool, argument) != (call.tool, call.argument):
                raise ValueError("idempotency key reused with different request")
            return result
        if call.tool == "read_document":
            result = self.documents[call.argument]
        else:
            self.drafts.append(call.argument)
            result = f"draft:{len(self.drafts)}"
        self.receipts[call.key] = (call.tool, call.argument, result)
        if call.tool == "save_draft" and self.timeout_after_commit_once:
            self.timeout_after_commit_once = False
            raise TimeoutError("reply lost after commit")
        return result


@dataclass(frozen=True)
class Result:
    status: str
    observations: tuple[Observation, ...]
    answer: str = ""
    tool_attempts: int = 0


def run(policy: Callable, environment: Environment, *, max_steps=8,
        max_tool_attempts=4, max_observation_chars=2000) -> Result:
    """Bound decisions and tool attempts; retry unknown delivery with the SAME key.

    This in-memory, single-process fixture has no remote-call timeout enforcement,
    durable storage, concurrency, real model tokens, or semantic answer verifier.
    """
    if min(max_steps, max_tool_attempts, max_observation_chars) <= 0:
        raise ValueError("budgets must be positive")
    observations = []
    attempts = 0
    pending = None
    for _ in range(max_steps):
        decision = pending if pending is not None else policy(tuple(observations))
        if isinstance(decision, Finish):
            return Result("finished", tuple(observations), decision.text, attempts)
        if not isinstance(decision, Call):
            return Result("invalid_action", tuple(observations), tool_attempts=attempts)
        if attempts >= max_tool_attempts:
            return Result("budget_exhausted", tuple(observations), tool_attempts=attempts)
        attempts += 1
        try:
            value = environment.execute(decision)
        except TimeoutError:
            pending = decision
            continue
        except (PermissionError, ValueError, KeyError) as error:
            return Result(type(error).__name__, tuple(observations), tool_attempts=attempts)
        pending = None
        observations.append(Observation(decision.tool, value[:max_observation_chars]))
    return Result("budget_exhausted", tuple(observations), tool_attempts=attempts)


if __name__ == "__main__":
    environment = Environment({"policy": "Drafts need review; never send automatically."})
    def policy(observations):
        if not observations:
            return Call("read_document", "policy", "read-1")
        return Finish("I found the policy; no message was sent.")
    result = run(policy, environment)
    print(result.status, result.tool_attempts, len(environment.drafts))
