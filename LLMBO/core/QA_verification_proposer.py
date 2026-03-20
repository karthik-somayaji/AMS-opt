import os
from typing import Optional

from core.proposer import LLMProposer


def _escape_langchain_braces(text: str) -> str:
    # LangChain PromptTemplate treats `{...}` as variables.
    return text.replace("{", "{{").replace("}", "}}")


class QAValidationLLMProposer(LLMProposer):
    """LLM proposer variant that injects a QA paragraph instead of related-circuit KGs."""

    def generate_prefix(self, prompt_input, args):
        prefix = ""

        if getattr(args, "history", 0):
            paragraph: Optional[str] = getattr(args, "qa_paragraph", None)
            if paragraph:
                paragraph = _escape_langchain_braces(str(paragraph))
                circuit = getattr(args, "circuit", None) or getattr(args, "target_id", None) or "unknown"
                prefix += f"\n\n[QA_PARAGRAPH circuit={circuit}]\n" + paragraph + "\n\n"

        # Keep the original task context and examples framing.
        prefix += self.task_context
        prefix += "**Examples**\n"
        return prefix
