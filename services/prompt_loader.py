from pathlib import Path
import yaml
from typing import Dict

class PromptLoader:

  @staticmethod
  def _load_prompts() -> Dict[str, str]:
    """Load prompt templates from YAML configuration file"""
    try:
        project_root = Path(__file__).parent.parent
        prompts_path = project_root / "config" / "prompts.yaml"

        with open(prompts_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading prompts from YAML: {e}")
        return {}