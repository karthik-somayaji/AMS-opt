"""Parse prune.txt files to extract removable components."""
import os
import json
import re
from typing import List, Optional


class PruneParser:
    """Parse <cir>_prune.txt to extract removable components."""
    
    def __init__(self, prune_file: str):
        """
        Args:
            prune_file: path to <cir>_prune.txt or <cir>_prune.json
        """
        if not os.path.exists(prune_file):
            raise FileNotFoundError(f"Prune file not found: {prune_file}")
        
        self.prune_file = prune_file
        self.components = self._parse()
    
    def _parse(self) -> List[str]:
        """
        Extract component IDs from prune file.
        
        Supports:
        - JSON array: ["M0", "M1", "R1"]
        - JSON object with 'components' key: {"components": ["M0", "M1"]}
        - Plain text list (one per line, optionally with JSON formatting)
        
        Returns:
            List of component IDs
        """
        with open(self.prune_file, 'r') as f:
            content = f.read().strip()
        
        components = []
        
        try:
            # Try JSON parsing
            data = json.loads(content)
            if isinstance(data, list):
                components = data
            elif isinstance(data, dict):
                if 'components' in data:
                    components = data['components']
                else:
                    components = list(data.keys())
        except json.JSONDecodeError:
            # Fall back to line-by-line parsing
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Try to extract component name
                    # Common formats: "M0", '"M0"', '"M0", "M1"'
                    matches = re.findall(r'"([^"]+)"', line)
                    if matches:
                        components.extend(matches)
                    elif line.isalnum() or (line[0] in ('M', 'R', 'C') and line[1:].isalnum()):
                        components.append(line)
        
        # Clean up: ensure all are strings, no duplicates
        components = list(set(str(c).strip('"') for c in components))
        return sorted(components)
    
    def get_components(self) -> List[str]:
        """Get list of all removable components."""
        return self.components.copy()
    
    def get_random_subset(self, max_fraction: float = 1.0, min_count: int = 1) -> List[str]:
        """
        Get a random subset of components to delete.
        
        Args:
            max_fraction: maximum fraction of components to delete (0.0 to 1.0)
            min_count: minimum number of components to delete
        
        Returns:
            List of component IDs to delete
        """
        import random
        
        if not self.components:
            return []
        
        max_deletable = max(min_count, int(len(self.components) * max_fraction))
        max_deletable = min(max_deletable, len(self.components))
        
        num_to_delete = random.randint(min_count, max_deletable)
        return random.sample(self.components, num_to_delete)
