from agent.client import Agent
from core.config import settings


def main() -> None:
    if not settings.anthropic_api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running this example.")

    agent = Agent()
    for token in agent.stream("What is the current directory? Use the PowerShell tool."):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
